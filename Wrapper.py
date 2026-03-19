from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from Segmenter import HSPPlannerConfig, Segmenter
from PageEncoder import PageEncoder
from QueryPlanner import QueryPlanner


@dataclass
class HSPWrapperConfig:
    page_size: int = 64
    flow_window: int = 4
    flash_top_k: int = 22
    anchor_pages: int = 4
    pad_token_id: Optional[int] = None
    newline_token_ids: Optional[Tuple[int, ...]] = None
    newline_token_id: int = 198
    sentence_boundary_ids: Optional[Tuple[int, ...]] = None

    allow_implicit_query: bool = False

    query_block_size: int = 64
    use_dynamic_token_weights: bool = True
    min_length_for_dynamic_weights: int = 256

    use_query_multitoken_semantic: bool = True
    min_query_tokens_for_multi: int = 4
    max_query_tokens_for_multi: int = 32
    identity_mean_weight: float = 0.7
    identity_max_weight: float = 0.3
    use_lexical_overlap_score: bool = True
    semantic_score_weight: float = 0.7
    lexical_score_weight: float = 0.3


class HSPBlackBoxWrapper:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: HSPWrapperConfig,
        idf_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.wrap_cfg = cfg

        if device is None:
            device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device
        self.model.to(self.device)

        if self.wrap_cfg.pad_token_id is None:
            if self.tokenizer.pad_token_id is not None:
                self.wrap_cfg.pad_token_id = int(self.tokenizer.pad_token_id)
            else:
                self.wrap_cfg.pad_token_id = 0
        else:
            if (
                self.tokenizer.pad_token_id is not None
                and int(self.wrap_cfg.pad_token_id) != int(self.tokenizer.pad_token_id)
            ):
                raise ValueError(
                    f"pad_token_id mismatch: wrapper cfg={self.wrap_cfg.pad_token_id}, tokenizer={self.tokenizer.pad_token_id}."
                )
        if self.wrap_cfg.newline_token_ids is None:
            auto_newline_ids: List[int] = []
            for s in ["\n", "\r\n", "\n\n"]:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = int(ids[0])
                    if tid not in auto_newline_ids:
                        auto_newline_ids.append(tid)
            if auto_newline_ids:
                self.wrap_cfg.newline_token_ids = tuple(auto_newline_ids)
                self.wrap_cfg.newline_token_id = auto_newline_ids[0]
        if self.wrap_cfg.sentence_boundary_ids is None:
            boundary_ids: List[int] = []
            if self.wrap_cfg.newline_token_ids is not None:
                boundary_ids.extend(int(x) for x in self.wrap_cfg.newline_token_ids)

            punct_candidates = ["。", "！", "？", ".", "!", "?"]
            for s in punct_candidates:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    tid = int(ids[0])
                    if tid not in boundary_ids:
                        boundary_ids.append(tid)

            if boundary_ids:
                self.wrap_cfg.sentence_boundary_ids = tuple(boundary_ids)
        hsp_cfg = HSPPlannerConfig(
            page_size=cfg.page_size,
            flow_window=cfg.flow_window,
            flash_top_k=cfg.flash_top_k,
            anchor_pages=cfg.anchor_pages,
            pad_token_id=cfg.pad_token_id,
            newline_token_id=cfg.newline_token_id,
            newline_token_ids=cfg.newline_token_ids,
            sentence_boundary_ids=cfg.sentence_boundary_ids,
            identity_mean_weight=cfg.identity_mean_weight,
            identity_max_weight=cfg.identity_max_weight,
            lambda_semantic=cfg.semantic_score_weight,
            lambda_lexical=cfg.lexical_score_weight,
            min_query_tokens_for_multi=cfg.min_query_tokens_for_multi,
            max_query_tokens_for_multi=cfg.max_query_tokens_for_multi,
        )
        self.hsp_cfg = hsp_cfg
        boundary_ids = (
            hsp_cfg.sentence_boundary_ids
            if hsp_cfg.sentence_boundary_ids is not None
            else (
                hsp_cfg.newline_token_ids
                if (hsp_cfg.newline_token_ids is not None and len(hsp_cfg.newline_token_ids) > 0)
                else ((hsp_cfg.newline_token_id,) if hsp_cfg.newline_token_id is not None else tuple())
            )
        )
        self.segmenter = Segmenter(
            cfg=hsp_cfg,
            query_block_size=cfg.query_block_size,
            boundary_ids=boundary_ids,
            align_explicit_query_pos=False,
        )

        hidden_dim = model.config.hidden_size
        self.page_encoder = PageEncoder(hsp_cfg, hidden_dim, idf_weights=idf_weights)
        self.query_planner = QueryPlanner(hsp_cfg, query_dim=hidden_dim)

        self.to(self.device)

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        self.segmenter.to(device)
        self.page_encoder.to(device)
        self.query_planner.to(device)
        return self

    # Token weights (tokenizer-dependent)
    def _compute_local_token_weights(
        self,
        input_ids: torch.Tensor,          # [B, L]
        attention_mask: torch.Tensor,     # [B, L]
    ) -> torch.Tensor:
        if not self.wrap_cfg.use_dynamic_token_weights:
            return attention_mask.float()

        B, L = input_ids.shape
        pad_id = self.wrap_cfg.pad_token_id
        device = input_ids.device

        weights = torch.zeros_like(input_ids, dtype=torch.float, device=device)

        for b in range(B):
            valid = attention_mask[b].bool() if attention_mask is not None else (input_ids[b] != pad_id)
            ids_valid = input_ids[b, valid]
            L_b = ids_valid.numel()

            if L_b == 0 or L_b < self.wrap_cfg.min_length_for_dynamic_weights:
                weights[b, valid] = 1.0
                continue

            uniq, counts = ids_valid.unique(return_counts=True)
            counts = counts.float()
            w = torch.log1p(L_b / (1.0 + counts))

            w_pos = torch.zeros_like(ids_valid, dtype=torch.float, device=device)
            for uid, wv in zip(uniq.tolist(), w.tolist()):
                mask = (ids_valid == uid)
                if mask.any():
                    w_pos[mask] = wv

            maxv = float(w_pos.max().item())
            if maxv > 0:
                w_pos = w_pos / maxv

            weights[b, valid] = w_pos

        weights = weights + 1e-6
        return weights

    # Build tokenized inputs (tokenizer-dependent)
    def _build_inputs_from_texts(
        self,
        contexts: List[str],
        questions: Optional[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = len(contexts)
        if questions is None:
            questions = [""] * B
        else:
            assert len(contexts) == len(questions)

        ctx_enc = [self.tokenizer.encode(c, add_special_tokens=False) for c in contexts]

        implicit_mode = False
        if self.wrap_cfg.allow_implicit_query:
            no_query_flags = [((q is None) or (str(q).strip() == "")) for q in questions]
            if all(no_query_flags):
                implicit_mode = True

        if implicit_mode:
            q_enc = [[] for _ in range(B)]
        else:
            q_enc = [self.tokenizer.encode((q if q is not None else ""), add_special_tokens=False) for q in questions]

        input_ids_list = []
        explicit_qp_list = []

        for i in range(B):
            ids_ctx = ctx_enc[i]
            ids_q = q_enc[i]
            all_ids = ids_ctx if implicit_mode else (ids_ctx + ids_q)
            if len(all_ids) == 0:
                all_ids = [self.wrap_cfg.pad_token_id]
                qp = 0
            else:
                qp = len(ids_ctx)
            input_ids_list.append(all_ids)
            if not implicit_mode:
                explicit_qp_list.append(qp)

        max_len = max(len(x) for x in input_ids_list)
        pad_id = self.wrap_cfg.pad_token_id

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

        for i in range(B):
            L_i = len(input_ids_list[i])
            input_ids[i, :L_i] = torch.tensor(input_ids_list[i], dtype=torch.long, device=self.device)
            attention_mask[i, :L_i] = 1

        explicit_qp = None if implicit_mode else torch.tensor(explicit_qp_list, dtype=torch.long, device=self.device)
        return input_ids, attention_mask, explicit_qp

    # Compress inputs for prefill
    @torch.no_grad()
    def compress_inputs_for_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        explicit_query_pos: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        B, L = input_ids.shape
        aligned_qp, split_info, layout = self.segmenter.build_layout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            explicit_query_pos=explicit_query_pos,
        )
        if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
            embed = self.model.get_input_embeddings()
            hidden = embed(input_ids)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            hidden = self.model.model.embed_tokens(input_ids)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            hidden = self.model.transformer.wte(input_ids)
        else:
            raise RuntimeError("Cannot locate input embedding layer for wrapped model.")
        token_level_weights = self._compute_local_token_weights(input_ids, attention_mask)
        block_repr = self.page_encoder(
            hidden_states=hidden,
            layout=layout,
            input_ids=input_ids,
            token_level_weights=token_level_weights,
        )
        token_valid = layout.token_valid
        query_hidden_list = []
        pos_idx = torch.arange(L, device=self.device)

        w_mean = getattr(self.hsp_cfg, "identity_mean_weight", 0.7)
        w_max = getattr(self.hsp_cfg, "identity_max_weight", 0.3)
        w_sum = w_mean + w_max
        if w_sum <= 0:
            w_mean_eff, w_max_eff = 1.0, 0.0
        else:
            w_mean_eff, w_max_eff = w_mean / w_sum, w_max / w_sum

        has_idf = getattr(self.page_encoder, "idf_weights", None) is not None

        for b in range(B):
            split_b = split_info[b]
            qs, qe = int(split_b.query_start), int(split_b.query_end)
            if qe < qs:
                qp = int(aligned_qp[b].item())
                query_hidden_list.append(hidden[b, qp])
                continue
            span_mask = (pos_idx >= qs) & (pos_idx <= qe) & token_valid[b]
            if not span_mask.any():
                qp = int(aligned_qp[b].item())
                query_hidden_list.append(hidden[b, qp])
                continue
            h_b = hidden[b]
            ids_b = input_ids[b]
            if has_idf and input_ids is not None:
                idf_vec = self.page_encoder.idf_weights
                idf_b = idf_vec[ids_b]
                weights = idf_b * span_mask.float()
                w_sum_b = float(weights.sum().item())
                if w_sum_b > 1e-6:
                    mean_b = (h_b * weights.unsqueeze(-1)).sum(dim=0) / w_sum_b
                else:
                    mean_b = h_b[span_mask].mean(dim=0)
            else:
                mean_b = h_b[span_mask].mean(dim=0)
            max_b, _ = h_b[span_mask].max(dim=0)
            pooled_b = w_mean_eff * mean_b + w_max_eff * max_b
            query_hidden_list.append(pooled_b)

        query_hidden = torch.stack(query_hidden_list, dim=0)
        query_token_hidden_list = []
        query_token_weight_list = []

        min_q = getattr(self.hsp_cfg, "min_query_tokens_for_multi", 4)
        max_q = getattr(self.hsp_cfg, "max_query_tokens_for_multi", 32)
        use_multi = self.wrap_cfg.use_query_multitoken_semantic and min_q > 0

        for b in range(B):
            split_b = split_info[b]
            qs, qe = int(split_b.query_start), int(split_b.query_end)
            if qe < qs or not use_multi:
                query_token_hidden_list.append(None)
                query_token_weight_list.append(None)
                continue
            span_mask = (pos_idx >= qs) & (pos_idx <= qe) & token_valid[b]
            idx_span = pos_idx[span_mask]
            Tq = idx_span.numel()
            if Tq < min_q:
                query_token_hidden_list.append(None)
                query_token_weight_list.append(None)
                continue
            h_b = hidden[b]
            w_b = token_level_weights[b]
            w_span = w_b[idx_span]
            top_k = min(Tq, max_q)
            vals, indices = torch.topk(w_span, k=top_k, largest=True, sorted=True)
            idx_top = idx_span[indices]
            h_top = h_b[idx_top]
            w_top = vals / (vals.sum() + 1e-6)
            query_token_hidden_list.append(h_top)
            query_token_weight_list.append(w_top)
        keep_pages = self.query_planner(
            block_repr=block_repr,
            layout=layout,
            query_hidden=query_hidden,
            query_pos=aligned_qp,
            input_ids=input_ids,
            token_level_weights=token_level_weights,
            split_results=split_info,
            query_token_hidden_list=query_token_hidden_list,
            query_token_weight_list=query_token_weight_list,
        )
        token2page = layout.token2page               # [B, L]
        token_valid = layout.token_valid             # [B, L] bool
        keep_pages_bool = keep_pages.bool()          # [B, P] bool
        token2page_clamped = token2page.clamp(min=0) # avoid -1 gather; token_valid will mask anyway
        keep_token = token_valid & keep_pages_bool.gather(1, token2page_clamped)
        boundary_ids = (
            self.hsp_cfg.sentence_boundary_ids
            if self.hsp_cfg.sentence_boundary_ids is not None
            else (
                self.hsp_cfg.newline_token_ids
                if (self.hsp_cfg.newline_token_ids is not None and len(self.hsp_cfg.newline_token_ids) > 0)
                else ((self.hsp_cfg.newline_token_id,) if self.hsp_cfg.newline_token_id is not None else tuple())
            )
        )
        kept_context_token_indices: List[List[int]] = []
        context_lens: List[int] = []
        for b in range(B):
            qp = int(aligned_qp[b].item())
            qp = max(0, min(qp, L))
            context_lens.append(qp)
            if qp <= 0:
                kept_context_token_indices.append([])
                continue
            keep_ctx = keep_token[b, :qp].clone()
            valid_ctx = token_valid[b, :qp]
            if boundary_ids is not None and len(boundary_ids) > 0:
                ids_slice = input_ids[b, :qp]
                is_boundary = torch.zeros(qp, dtype=torch.bool, device=input_ids.device)
                for bid in boundary_ids:
                    is_boundary |= (ids_slice == int(bid))
                boundary_pos = torch.nonzero(is_boundary, as_tuple=False).flatten().tolist()
                start = 0
                for p in boundary_pos:
                    end = int(p) + 1
                    if end <= start:
                        continue
                    if keep_ctx[start:end].any():
                        keep_ctx[start:end] = valid_ctx[start:end]
                    start = end
                if start < qp and keep_ctx[start:qp].any():
                    keep_ctx[start:qp] = valid_ctx[start:qp]
            kept_idx = torch.nonzero(keep_ctx & valid_ctx, as_tuple=False).flatten().detach().cpu().tolist()
            kept_context_token_indices.append([int(x) for x in kept_idx])
        compressed = self.segmenter.compress(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layout=layout,
            keep_pages=keep_pages,
            query_pos=aligned_qp,
        )
        L_comp = compressed["input_ids"].size(1)
        stats = {
            "aligned_query_pos": [int(x) for x in aligned_qp.detach().cpu().tolist()],
            "context_len": context_lens,
            "kept_context_token_indices": kept_context_token_indices,
            "original_len": int(L),
            "compressed_len": int(L_comp),
            "compression_ratio": float(L_comp / max(L, 1)),
        }
        return compressed, stats

    # Batch generation (dense or BEAVER)
    @torch.no_grad()
    def generate_batch(
        self,
        contexts: List[str],
        questions: List[str],
        max_new_tokens: int = 128,
        use_hsp: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        assert len(contexts) == len(questions)
        self.model.eval()

        input_ids, attention_mask, explicit_qp = self._build_inputs_from_texts(contexts, questions)
        prompt_lens = attention_mask.sum(dim=1).tolist()

        if not use_hsp:
            t0 = time.time()
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **gen_kwargs,
            )
            t1 = time.time()

            texts: List[str] = []
            for i in range(outputs.size(0)):
                pl = int(prompt_lens[i]) if i < len(prompt_lens) else 0
                answer_tokens = outputs[i, pl:]
                texts.append(self.tokenizer.decode(answer_tokens, skip_special_tokens=True))

            meta = {
                "mode": "dense",
                "time_total": float(t1 - t0),
                "original_len": int(input_ids.size(1)),
                "compressed_len": int(input_ids.size(1)),
                "compression_ratio": 1.0,
            }
            return {"outputs": texts, "meta": meta}
        t0 = time.time()
        compressed_inputs, stats = self.compress_inputs_for_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            explicit_query_pos=explicit_qp,
        )
        t1 = time.time()

        comp_attn = compressed_inputs["attention_mask"]
        comp_prompt_lens = comp_attn.sum(dim=1).tolist()
        gen_outputs = self.model.generate(
            input_ids=compressed_inputs["input_ids"],
            attention_mask=compressed_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **gen_kwargs,
        )
        t2 = time.time()

        texts: List[str] = []
        for i in range(gen_outputs.size(0)):
            pl = int(comp_prompt_lens[i]) if i < len(comp_prompt_lens) else 0
            answer_tokens = gen_outputs[i, pl:]
            texts.append(self.tokenizer.decode(answer_tokens, skip_special_tokens=True))

        meta = {
            "mode": "hsp",
            "time_prefill": float(t1 - t0),
            "time_total": float(t2 - t0),
            "original_len": stats["original_len"],
            "compressed_len": stats["compressed_len"],
            "compression_ratio": stats["compression_ratio"],
        }
        return {"outputs": texts, "meta": meta}