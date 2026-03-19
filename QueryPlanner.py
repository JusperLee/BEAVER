# QueryPlanner.py
# BEAVER query planning module.
# Scores pages with semantic + lexical signals and selects pages via Anchor / Flow / Flash.

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Segmenter import HSPPlannerConfig, SegmentPageLayout, QuerySplitResult


class QueryPlanner(nn.Module):
    """Plan kept pages based on semantic and lexical relevance."""

    def __init__(self, cfg: HSPPlannerConfig, query_dim: int):
        super().__init__()
        self.cfg = cfg

        self.lambda_sem = float(getattr(cfg, "lambda_semantic", 0.7))
        self.lambda_lex = float(getattr(cfg, "lambda_lexical", 0.3))
        self.min_q_multi = int(getattr(cfg, "min_query_tokens_for_multi", 4))
        self.max_q_multi = int(getattr(cfg, "max_query_tokens_for_multi", 32))

    def forward(
        self,
        block_repr: torch.Tensor,       # [B, N, D]
        layout: SegmentPageLayout,
        query_hidden: torch.Tensor,     # [B, D]
        query_pos: torch.Tensor,        # [B]
        input_ids: Optional[torch.Tensor] = None,                    # [B, L]
        token_level_weights: Optional[torch.Tensor] = None,          # [B, L]
        split_results: Optional[Tuple[QuerySplitResult, ...]] = None,
        query_token_hidden_list: Optional[List[Optional[torch.Tensor]]] = None,  # len=B, [K_b, D]
        query_token_weight_list: Optional[List[Optional[torch.Tensor]]] = None,  # len=B, [K_b]
    ) -> torch.Tensor:
        B, N, Dp = block_repr.shape
        device = block_repr.device

        common_dtype = block_repr.dtype
        if query_hidden.dtype != common_dtype:
            query_hidden = query_hidden.to(common_dtype)

        if query_token_hidden_list is not None:
            new_list: List[Optional[torch.Tensor]] = []
            for qt in query_token_hidden_list:
                new_list.append(None if qt is None else qt.to(common_dtype))
            query_token_hidden_list = new_list

        segment_ids = layout.segment_ids
        page_valid_any = layout.page_valid.any(dim=-1)
        token2page = layout.token2page
        token_valid = layout.token_valid

        query_page = torch.gather(token2page, dim=1, index=query_pos.view(B, 1)).squeeze(1)
        query_page = query_page.clamp(min=0)

        query_seg = torch.gather(segment_ids, dim=1, index=query_page.view(B, 1)).squeeze(1)

        scores_sem = block_repr.new_full((B, N), -1e4)
        k_vec = F.normalize(block_repr, dim=-1)

        use_multi = (query_token_hidden_list is not None and self.min_q_multi > 0)

        for b in range(B):
            if use_multi and query_token_hidden_list[b] is not None:
                q_tok = query_token_hidden_list[b]
                if q_tok.numel() == 0:
                    continue
                q_vec = F.normalize(q_tok, dim=-1)
                k_b = k_vec[b]
                scores_bn = torch.matmul(q_vec, k_b.t())

                if query_token_weight_list is not None and query_token_weight_list[b] is not None:
                    w_q = query_token_weight_list[b]
                    if w_q.numel() == scores_bn.size(0):
                        w_q = w_q / (w_q.sum() + 1e-6)
                        scores_sem[b] = (scores_bn * w_q.unsqueeze(-1)).sum(dim=0)
                    else:
                        scores_sem[b] = scores_bn.mean(dim=0)
                else:
                    scores_sem[b] = scores_bn.mean(dim=0)
            else:
                q = query_hidden[b:b+1]
                q_vec = F.normalize(q, dim=-1)
                k_b = k_vec[b:b+1]
                scores_bn = torch.einsum("bd,bnd->bn", q_vec, k_b)
                scores_sem[b] = scores_bn[0]

        scores_lex = block_repr.new_zeros((B, N))
        use_lex = (
            self.lambda_lex > 0
            and input_ids is not None
            and token_level_weights is not None
            and split_results is not None
        )

        if use_lex:
            for b in range(B):
                sr = split_results[b]
                qs = int(sr.query_start)
                qe = int(sr.query_end)
                ids_b = input_ids[b]
                valid_b = token_valid[b]
                if qe < qs:
                    continue
                L = ids_b.size(0)
                qs = max(0, min(qs, L - 1))
                qe = max(0, min(qe, L - 1))

                span_mask = torch.zeros_like(valid_b, dtype=torch.bool)
                span_mask[qs:qe+1] = True
                span_mask &= valid_b
                if not span_mask.any():
                    continue

                q_ids = ids_b[span_mask]
                w_b = token_level_weights[b]

                for n in range(N):
                    idx_n = layout.page_indices[b, n]
                    valid_n = layout.page_valid[b, n]
                    if not valid_n.any():
                        continue
                    pos_n = idx_n[valid_n]
                    tok_n = ids_b[pos_n]
                    mask_in_q = torch.isin(tok_n, q_ids)
                    if mask_in_q.any():
                        w_page = w_b[pos_n]
                        scores_lex[b, n] = (w_page[mask_in_q]).sum()

        page_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
        causal_mask = (page_idx <= query_page.view(B, 1))
        valid_mask = page_valid_any & (segment_ids >= 0) & causal_mask

        def _norm_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            out = scores.clone()
            for bb in range(scores.size(0)):
                m = mask[bb]
                if not m.any():
                    continue
                v = out[bb, m]
                v_min = v.min()
                v_max = v.max()
                if float(v_max - v_min) < 1e-6:
                    out[bb, m] = 0.0
                else:
                    out[bb, m] = (v - v_min) / (v_max - v_min)
            out = out.masked_fill(~mask, 0.0)
            return out

        scores_sem_norm = _norm_scores(scores_sem, valid_mask)
        scores_mix = scores_sem_norm

        if use_lex:
            scores_lex_norm = _norm_scores(scores_lex, valid_mask)
            if self.lambda_sem + self.lambda_lex > 0:
                lam_s = self.lambda_sem
                lam_l = self.lambda_lex
                scores_mix = (lam_s * scores_sem_norm + lam_l * scores_lex_norm) / (lam_s + lam_l)

        scores_final = scores_mix.masked_fill(~valid_mask, -1e4)

        anchor = torch.zeros_like(valid_mask)
        is_seg0 = (segment_ids == 0) & valid_mask
        for b in range(B):
            idx_seg0 = torch.nonzero(is_seg0[b], as_tuple=False).flatten()
            if idx_seg0.numel() > 0:
                k = min(self.cfg.anchor_pages, idx_seg0.numel())
                anchor[b, idx_seg0[:k]] = True

        if self.cfg.flow_window >= 0:
            lower = (query_page.view(B, 1) - self.cfg.flow_window).clamp(min=0)
            upper = query_page.view(B, 1)
            flow = (page_idx >= lower) & (page_idx <= upper) & valid_mask
        else:
            flow = valid_mask.clone()

        flow.scatter_(1, query_page.view(B, 1), True)

        base_keep = anchor | flow
        candidate = valid_mask & (~base_keep)

        scores_candidate = scores_final.masked_fill(~candidate, -1e4)
        effective_k = min(self.cfg.flash_top_k, N)
        if effective_k > 0:
            _, topk_idx = torch.topk(scores_candidate, k=effective_k, dim=1)
            flash = torch.zeros_like(candidate)
            flash.scatter_(1, topk_idx, True)
            flash = flash & candidate
        else:
            flash = torch.zeros_like(candidate)

        keep_pages = base_keep | flash
        return keep_pages