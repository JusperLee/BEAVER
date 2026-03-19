# Segmenter.py
# BEAVER segmentation module.
# Components: QueryLocator, SegmentPager, HSPPrefillCompressor, Segmenter.
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shared config and layout dataclasses
@dataclass
class HSPPlannerConfig:
    page_size: int = 64
    identity_mean_weight: float = 0.7
    identity_max_weight: float = 0.3
    flow_window: int = 4
    flash_top_k: int = 22
    anchor_pages: int = 4
    pad_token_id: int = 0
    newline_token_id: Optional[int] = None
    newline_token_ids: Optional[Tuple[int, ...]] = None
    sentence_boundary_ids: Optional[Tuple[int, ...]] = None
    lambda_semantic: float = 0.7
    lambda_lexical: float = 0.3
    min_query_tokens_for_multi: int = 4
    max_query_tokens_for_multi: int = 32


@dataclass
class SegmentPageLayout:
    """Intermediate segment/page layout tensors."""
    token_valid: torch.Tensor
    page_indices: torch.Tensor
    page_valid: torch.Tensor
    segment_ids: torch.Tensor
    token2page: torch.Tensor


@dataclass
class QuerySplitResult:
    """Per-sample context/query split ranges (token indices)."""
    ctx_start: int
    ctx_end: int
    query_start: int
    query_end: int


 
# Query locator: split context/query using an explicit query position (no alignment by default)
class QueryLocator(nn.Module):
    """Split context/query at an explicit query position (no boundary alignment by default)."""

    def __init__(
        self,
        pad_token_id: int,
        special_sep_ids: Optional[Tuple[int, ...]] = None,
        query_block_size: int = 128,
        sentence_boundary_ids: Optional[Tuple[int, ...]] = None,
        align_explicit_query_pos: bool = False,
        implicit_query_len: Optional[int] = None,
    ):
        super().__init__()
        self.pad_token_id = int(pad_token_id)
        self.special_sep_ids = special_sep_ids or ()
        self.query_block_size = int(query_block_size)
        self.implicit_query_len = int(implicit_query_len) if implicit_query_len is not None else None

        if sentence_boundary_ids is not None:
            self.sentence_boundary_ids = tuple(int(x) for x in sentence_boundary_ids)
            self._sentence_boundary_set = set(self.sentence_boundary_ids)
        else:
            self.sentence_boundary_ids = tuple()
            self._sentence_boundary_set = set()

        self.align_explicit_query_pos = bool(align_explicit_query_pos)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,                    # [B, L]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or None
        explicit_query_pos: Optional[torch.Tensor] = None,  # [B]
    ) -> Tuple[torch.Tensor, Tuple[QuerySplitResult, ...]]:
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            token_valid = attention_mask.bool()
        else:
            token_valid = (input_ids != self.pad_token_id)

        if explicit_query_pos is None:
            qp = torch.zeros((B,), device=device, dtype=torch.long)
            fallback_len = self.implicit_query_len if self.implicit_query_len is not None else max(0, self.query_block_size)

            for b in range(B):
                valid_b = token_valid[b]
                if valid_b.any():
                    last_idx = torch.nonzero(valid_b, as_tuple=False).flatten()[-1].item()
                else:
                    last_idx = 0

                qp0 = int(max(0, (int(last_idx) + 1) - int(fallback_len)))
                qp0 = min(qp0, L - 1) if L > 0 else 0
                qp[b] = qp0

            do_align = (self.query_block_size > 0 and bool(self._sentence_boundary_set))
        else:
            qp = explicit_query_pos.clone().to(device=device, dtype=torch.long)
            qp = qp.clamp(min=0, max=L - 1)

            do_align = (
                self.align_explicit_query_pos
                and self.query_block_size > 0
                and bool(self._sentence_boundary_set)
            )
        if do_align:
            for b in range(B):
                q0 = int(qp[b].item())
                if not token_valid[b].any():
                    continue
                window_left = max(0, q0 - self.query_block_size)
                window_right = q0
                if window_right <= window_left:
                    continue

                ids_slice = input_ids[b, window_left:window_right]
                valid_slice = token_valid[b, window_left:window_right]

                found = False
                for offset in range(ids_slice.size(0) - 1, -1, -1):
                    if not bool(valid_slice[offset]):
                        continue
                    tok_id = int(ids_slice[offset].item())
                    if tok_id in self._sentence_boundary_set:
                        qp[b] = window_left + offset + 1
                        found = True
                        break
                if not found:
                    qp[b] = window_left

        split_results: List[QuerySplitResult] = []
        for b in range(B):
            valid_b = token_valid[b]
            if valid_b.any():
                last_idx = torch.nonzero(valid_b, as_tuple=False).flatten()[-1].item()
            else:
                last_idx = 0
            qpb = int(qp[b].item())
            split_results.append(
                QuerySplitResult(
                    ctx_start=0,
                    ctx_end=qpb,
                    query_start=qpb,
                    query_end=last_idx,
                )
            )

        return qp, tuple(split_results)


# Segment pager: build segment-to-page layout on token ids
class SegmentPager(nn.Module):
    """Build a segment-to-page layout using token ids and an attention mask."""

    def __init__(self, cfg: HSPPlannerConfig):
        super().__init__()
        self.cfg = cfg

        if getattr(cfg, "newline_token_ids", None) is not None and len(cfg.newline_token_ids) > 0:
            newline_ids_list = [int(x) for x in cfg.newline_token_ids]
        elif getattr(cfg, "newline_token_id", None) is not None:
            newline_ids_list = [int(cfg.newline_token_id)]
        else:
            newline_ids_list = []

        self.register_buffer(
            "newline_token_ids_tensor",
            torch.tensor(newline_ids_list, dtype=torch.long),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,              # [B, L]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or None
        query_pos: Optional[torch.Tensor] = None,       # [B] or None
    ) -> SegmentPageLayout:
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            token_valid = attention_mask.bool()
        else:
            token_valid = (input_ids != self.cfg.pad_token_id)

        all_indices = torch.arange(L, device=device)

        batch_page_indices = []
        batch_page_valid = []
        batch_segment_ids = []
        max_pages = 0

        for b in range(B):
            tokens_b = input_ids[b]
            valid_b = token_valid[b]
            if query_pos is not None:
                qp_b = int(query_pos[b].item())
                qp_b = max(0, min(qp_b, L))
            else:
                qp_b = None

            if not valid_b.any():
                page_idx = torch.full((1, self.cfg.page_size), -1, dtype=torch.long, device=device)
                page_valid = torch.zeros((1, self.cfg.page_size), dtype=torch.bool, device=device)
                seg_ids = torch.full((1,), -1, dtype=torch.long, device=device)

                batch_page_indices.append(page_idx)
                batch_page_valid.append(page_valid)
                batch_segment_ids.append(seg_ids)
                max_pages = max(max_pages, 1)
                continue

            if self.newline_token_ids_tensor.numel() > 0:
                is_nl = torch.zeros_like(tokens_b, dtype=torch.bool)
                for tid in self.newline_token_ids_tensor.tolist():
                    is_nl |= (tokens_b == tid)
            else:
                is_nl = torch.zeros_like(tokens_b, dtype=torch.bool)

            nl_pos = torch.nonzero(is_nl, as_tuple=False).flatten()
            split_points = [0]
            if nl_pos.numel() > 0:
                split_points += (nl_pos + 1).tolist()

            if qp_b is not None and 0 < qp_b < L:
                split_points.append(qp_b)

            if L not in split_points:
                split_points.append(L)

            split_points = sorted(set(int(x) for x in split_points))

            pages_this = []
            valid_this = []
            seg_ids_this = []

            cur_idx_chunks = []
            cur_val_chunks = []
            cur_len = 0
            cur_seg_id = None

            seg_id = 0
            for i in range(len(split_points) - 1):
                s, e = split_points[i], split_points[i + 1]
                if s >= e:
                    seg_id += 1
                    continue

                if qp_b is not None and s == qp_b and cur_len > 0:
                    page_idx = torch.cat(cur_idx_chunks, dim=0)
                    page_val = torch.cat(cur_val_chunks, dim=0)
                    pad_len = self.cfg.page_size - page_idx.numel()
                    if pad_len > 0:
                        page_idx = F.pad(page_idx, (0, pad_len), value=-1)
                        page_val = F.pad(page_val, (0, pad_len), value=False)
                    pages_this.append(page_idx)
                    valid_this.append(page_val)
                    seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)
                    cur_idx_chunks, cur_val_chunks = [], []
                    cur_len = 0
                    cur_seg_id = None

                indices_seg = all_indices[s:e]
                valid_seg = valid_b[s:e]
                seg_len = indices_seg.numel()
                offset = 0

                while offset < seg_len:
                    len_left = seg_len - offset
                    remaining = self.cfg.page_size - cur_len

                    if len_left > self.cfg.page_size:
                        if cur_len > 0:
                            page_idx = torch.cat(cur_idx_chunks, dim=0)
                            page_val = torch.cat(cur_val_chunks, dim=0)
                            pad_len = self.cfg.page_size - page_idx.numel()
                            if pad_len > 0:
                                page_idx = F.pad(page_idx, (0, pad_len), value=-1)
                                page_val = F.pad(page_val, (0, pad_len), value=False)
                            pages_this.append(page_idx)
                            valid_this.append(page_val)
                            seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)
                            cur_idx_chunks, cur_val_chunks = [], []
                            cur_len = 0
                            cur_seg_id = None
                            remaining = self.cfg.page_size

                        full_chunks = len_left // self.cfg.page_size
                        for _ in range(full_chunks):
                            chunk_idx = indices_seg[offset : offset + self.cfg.page_size]
                            chunk_val = valid_seg[offset : offset + self.cfg.page_size]
                            pages_this.append(chunk_idx)
                            valid_this.append(chunk_val)
                            seg_ids_this.append(seg_id)
                            offset += self.cfg.page_size
                        continue

                    if len_left <= remaining:
                        cur_idx_chunks.append(indices_seg[offset : offset + len_left])
                        cur_val_chunks.append(valid_seg[offset : offset + len_left])
                        if cur_seg_id is None:
                            cur_seg_id = seg_id
                        cur_len += len_left
                        offset += len_left

                        if cur_len == self.cfg.page_size:
                            page_idx = torch.cat(cur_idx_chunks, dim=0)
                            page_val = torch.cat(cur_val_chunks, dim=0)
                            pages_this.append(page_idx)
                            valid_this.append(page_val)
                            seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)
                            cur_idx_chunks, cur_val_chunks = [], []
                            cur_len = 0
                            cur_seg_id = None
                        continue
                    else:
                        if cur_len > 0:
                            page_idx = torch.cat(cur_idx_chunks, dim=0)
                            page_val = torch.cat(cur_val_chunks, dim=0)
                            pad_len = self.cfg.page_size - page_idx.numel()
                            if pad_len > 0:
                                page_idx = F.pad(page_idx, (0, pad_len), value=-1)
                                page_val = F.pad(page_val, (0, pad_len), value=False)
                            pages_this.append(page_idx)
                            valid_this.append(page_val)
                            seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)
                            cur_idx_chunks, cur_val_chunks = [], []
                            cur_len = 0
                            cur_seg_id = None

                        cur_idx_chunks.append(indices_seg[offset : offset + len_left])
                        cur_val_chunks.append(valid_seg[offset : offset + len_left])
                        cur_seg_id = seg_id
                        cur_len = len_left
                        offset += len_left

                        if cur_len == self.cfg.page_size:
                            page_idx = torch.cat(cur_idx_chunks, dim=0)
                            page_val = torch.cat(cur_val_chunks, dim=0)
                            pages_this.append(page_idx)
                            valid_this.append(page_val)
                            seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)
                            cur_idx_chunks, cur_val_chunks = [], []
                            cur_len = 0
                            cur_seg_id = None
                        continue

                seg_id += 1

            if cur_len > 0:
                page_idx = torch.cat(cur_idx_chunks, dim=0)
                page_val = torch.cat(cur_val_chunks, dim=0)
                pad_len = self.cfg.page_size - page_idx.numel()
                if pad_len > 0:
                    page_idx = F.pad(page_idx, (0, pad_len), value=-1)
                    page_val = F.pad(page_val, (0, pad_len), value=False)
                pages_this.append(page_idx)
                valid_this.append(page_val)
                seg_ids_this.append(cur_seg_id if cur_seg_id is not None else -1)

            if pages_this:
                page_idx = torch.stack(pages_this, dim=0)
                page_valid = torch.stack(valid_this, dim=0)
                seg_ids = torch.tensor(seg_ids_this, device=device, dtype=torch.long)
            else:
                page_idx = torch.full((1, self.cfg.page_size), -1, dtype=torch.long, device=device)
                page_valid = torch.zeros((1, self.cfg.page_size), dtype=torch.bool, device=device)
                seg_ids = torch.full((1,), -1, dtype=torch.long, device=device)

            batch_page_indices.append(page_idx)
            batch_page_valid.append(page_valid)
            batch_segment_ids.append(seg_ids)
            max_pages = max(max_pages, page_idx.size(0))

        page_indices = torch.full((B, max_pages, self.cfg.page_size), -1, dtype=torch.long, device=device)
        page_valid = torch.zeros((B, max_pages, self.cfg.page_size), dtype=torch.bool, device=device)
        segment_ids = torch.full((B, max_pages), -1, dtype=torch.long, device=device)

        for b in range(B):
            n = batch_page_indices[b].size(0)
            page_indices[b, :n] = batch_page_indices[b]
            page_valid[b, :n] = batch_page_valid[b]
            segment_ids[b, :n] = batch_segment_ids[b]

        token2page = torch.full((B, L), -1, dtype=torch.long, device=device)
        for b in range(B):
            idx_b = page_indices[b]   # [N, P]
            valid_b = page_valid[b]   # [N, P]
            N = idx_b.size(0)

            flat_idx = idx_b.view(-1)
            flat_valid = valid_b.view(-1)
            mask = (flat_idx >= 0) & flat_valid

            if mask.any():
                page_ids = torch.arange(N, device=device).view(N, 1).expand(N, self.cfg.page_size)
                flat_page_ids = page_ids.reshape(-1)
                token2page[b].scatter_(0, flat_idx[mask], flat_page_ids[mask])

        return SegmentPageLayout(
            token_valid=token_valid,
            page_indices=page_indices,
            page_valid=page_valid,
            segment_ids=segment_ids,
            token2page=token2page,
        )


# Prefill compressor: apply keep_pages to tokens and run sentence smoothing on [0, query_pos)
class HSPPrefillCompressor(nn.Module):
    def __init__(self, cfg: HSPPlannerConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.sentence_boundary_ids is not None:
            boundary_ids = tuple(int(x) for x in cfg.sentence_boundary_ids)
        elif getattr(cfg, "newline_token_ids", None) is not None and len(cfg.newline_token_ids) > 0:
            boundary_ids = tuple(int(x) for x in cfg.newline_token_ids)
        else:
            boundary_ids = tuple()

        self.sentence_boundary_ids = boundary_ids
        self._sentence_boundary_set = set(boundary_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,              # [B, L]
        attention_mask: Optional[torch.Tensor],
        layout: SegmentPageLayout,
        keep_pages: torch.Tensor,             # [B, N] bool
        query_pos: torch.Tensor,              # [B]
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is not None:
            token_valid = attention_mask.bool()
        else:
            token_valid = layout.token_valid

        token2page = layout.token2page
        token2page_clamped = token2page.clamp(min=0)
        page_mask_for_token = torch.gather(keep_pages, dim=1, index=token2page_clamped)  # [B, L]
        keep_token = token_valid & page_mask_for_token

        query_pos_tensor = query_pos.to(device=device, dtype=torch.long).clone()
        query_pos_tensor.clamp_(min=0, max=L - 1)

        kept_lengths: List[int] = []
        kept_ids: List[torch.Tensor] = []
        kept_pos: List[torch.Tensor] = []

        for b in range(B):
            mask_b = keep_token[b].clone()
            valid_b = token_valid[b]
            ids_b = input_ids[b]

            qp = int(query_pos_tensor[b].item())
            qp = max(0, min(qp, L - 1))

            if qp < L:
                mask_b[qp:] |= valid_b[qp:]

            if self._sentence_boundary_set and qp > 0:
                ctx_end = qp
                boundary_mask = valid_b.clone().fill_(False)
                for tok_id in self.sentence_boundary_ids:
                    boundary_mask |= (ids_b == tok_id)
                boundary_mask &= valid_b
                boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).flatten()
                if boundary_positions.numel() > 0:
                    boundary_positions = boundary_positions[boundary_positions < ctx_end]

                sentence_starts = []
                sentence_ends = []
                start = 0
                for pos in boundary_positions.tolist():
                    end = pos + 1
                    if end > ctx_end:
                        end = ctx_end
                    if start < end:
                        sentence_starts.append(start)
                        sentence_ends.append(end)
                        start = end
                if start < ctx_end:
                    sentence_starts.append(start)
                    sentence_ends.append(ctx_end)

                for s, e in zip(sentence_starts, sentence_ends):
                    span_valid = valid_b[s:e]
                    if not span_valid.any():
                        continue
                    span_keep = mask_b[s:e]
                    if span_keep.any():
                        mask_b[s:e] = span_keep | span_valid

            idx_b = torch.nonzero(mask_b, as_tuple=False).flatten()
            if idx_b.numel() == 0:
                if valid_b.any():
                    idx_b = torch.nonzero(valid_b, as_tuple=False).flatten()[-1:]
                else:
                    idx_b = torch.tensor([0], device=device, dtype=torch.long)

            kept_lengths.append(int(idx_b.numel()))
            kept_ids.append(input_ids[b, idx_b])
            kept_pos.append(torch.arange(idx_b.numel(), device=device, dtype=torch.long))

        max_len = max(kept_lengths)
        input_ids_comp = input_ids.new_full((B, max_len), self.cfg.pad_token_id)
        attn_comp = input_ids.new_zeros((B, max_len), dtype=torch.long)
        pos_comp = torch.zeros((B, max_len), device=device, dtype=torch.long)

        for b in range(B):
            Lk = kept_lengths[b]
            input_ids_comp[b, :Lk] = kept_ids[b]
            attn_comp[b, :Lk] = 1
            pos_comp[b, :Lk] = kept_pos[b]

        return {
            "input_ids": input_ids_comp,
            "attention_mask": attn_comp,
            "position_ids": pos_comp,
        }


# Public API: build_layout + compress
class Segmenter(nn.Module):
    """Public interface that wires QueryLocator + SegmentPager + HSPPrefillCompressor."""

    def __init__(
        self,
        cfg: HSPPlannerConfig,
        query_block_size: int = 128,
        boundary_ids: Optional[Tuple[int, ...]] = None,
        align_explicit_query_pos: bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        if boundary_ids is None:
            if cfg.sentence_boundary_ids is not None:
                boundary_ids = cfg.sentence_boundary_ids
            elif cfg.newline_token_ids is not None and len(cfg.newline_token_ids) > 0:
                boundary_ids = cfg.newline_token_ids
            elif cfg.newline_token_id is not None:
                boundary_ids = (cfg.newline_token_id,)
            else:
                boundary_ids = tuple()

        self.query_locator = QueryLocator(
            pad_token_id=cfg.pad_token_id,
            special_sep_ids=None,
            query_block_size=query_block_size,
            sentence_boundary_ids=boundary_ids,
            align_explicit_query_pos=align_explicit_query_pos,
            implicit_query_len=cfg.page_size,
        )
        self.segment_pager = SegmentPager(cfg)
        self.compressor = HSPPrefillCompressor(cfg)

    @torch.no_grad()
    def build_layout(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        explicit_query_pos: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[QuerySplitResult, ...], SegmentPageLayout]:
        aligned_qp, split_info = self.query_locator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            explicit_query_pos=explicit_query_pos,
        )
        layout = self.segment_pager(input_ids=input_ids, attention_mask=attention_mask, query_pos=aligned_qp)
        return aligned_qp, split_info, layout

    @torch.no_grad()
    def compress(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layout: SegmentPageLayout,
        keep_pages: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.compressor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layout=layout,
            keep_pages=keep_pages,
            query_pos=query_pos,
        )