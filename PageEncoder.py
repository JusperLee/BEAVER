# PageEncoder.py
# BEAVER page encoding module.
# Pools token embeddings into per-page representations with optional weighting.
from typing import Optional

import torch
import torch.nn as nn

from Segmenter import HSPPlannerConfig, SegmentPageLayout


class PageEncoder(nn.Module):
    """Encode pages by pooling token embeddings under a segment/page layout."""

    def __init__(
        self,
        cfg: HSPPlannerConfig,
        hidden_dim: int,
        idf_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = int(hidden_dim)

        if idf_weights is not None:
            assert idf_weights.dim() == 1
            self.register_buffer("idf_weights", idf_weights.float())
        else:
            self.idf_weights = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout: SegmentPageLayout,
        input_ids: Optional[torch.Tensor] = None,
        token_level_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape
        device = hidden_states.device

        page_indices = layout.page_indices
        page_valid = layout.page_valid
        B2, N, P = page_indices.shape
        assert B2 == B

        x_paged = hidden_states.new_zeros(B, N, P, D)
        for b in range(B):
            idx_b = page_indices[b]
            flat_idx = idx_b.view(-1)
            mask = (flat_idx >= 0)
            if mask.any():
                x_b = hidden_states[b, flat_idx[mask]]
                x_paged[b].view(-1, D)[mask] = x_b

        mask_exp = page_valid.unsqueeze(-1)  # [B, N, P, 1]

        x_sum = (x_paged * mask_exp).sum(dim=2)      # [B, N, D]
        count = mask_exp.sum(dim=2).clamp(min=1)     # [B, N, 1]
        x_mean_uniform = x_sum / count

        weights_eff = page_valid.float()

        if self.idf_weights is not None and input_ids is not None:
            idx_clamped = page_indices.clamp(min=0)
            idx_flat = idx_clamped.view(B, -1)
            tokens_flat = input_ids.gather(1, idx_flat)
            tokens_paged = tokens_flat.view(B, N, P)
            idf = self.idf_weights[tokens_paged]
            weights_eff = weights_eff * idf

        if token_level_weights is not None:
            idx_clamped = page_indices.clamp(min=0)
            idx_flat = idx_clamped.view(B, -1)
            w_flat = token_level_weights.gather(1, idx_flat)
            w_paged = w_flat.view(B, N, P)
            weights_eff = weights_eff * w_paged

        if (self.idf_weights is not None and input_ids is not None) or (token_level_weights is not None):
            w_sum = weights_eff.sum(dim=2, keepdim=True)  # [B, N, 1]
            thr = 1e-4
            low = (w_sum < thr)
            w_sum_safe = w_sum.clone()
            w_sum_safe[w_sum_safe < 1e-6] = 1.0
            x_weighted = (x_paged * weights_eff.unsqueeze(-1)).sum(dim=2)  # [B, N, D]
            x_mean_weighted = x_weighted / w_sum_safe
            low_expand = low.expand(-1, -1, D)
            x_mean = torch.where(low_expand, x_mean_uniform, x_mean_weighted)
        else:
            x_mean = x_mean_uniform

        neg_inf = hidden_states.new_full((), -1e4)
        x_for_max = x_paged.masked_fill(~mask_exp, neg_inf)
        x_max = x_for_max.max(dim=2).values

        w_mean = float(getattr(self.cfg, "identity_mean_weight", 0.7))
        w_max = float(getattr(self.cfg, "identity_max_weight", 0.3))
        s = w_mean + w_max
        if s <= 0:
            block_repr = x_mean
        else:
            w_mean = w_mean / s
            w_max = w_max / s
            block_repr = w_mean * x_mean + w_max * x_max

        return block_repr.to(hidden_states.dtype)