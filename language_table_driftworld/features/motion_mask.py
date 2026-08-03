"""
Token-space motion weighting for the drifting loss
"""
import torch
import logging

log = logging.getLogger(__name__)


class TokenMasker:
    def __init__(self, motion_mask_alpha: float = 2.5, motion_mask_lambda=None,
                 motion_mask_qhi: float = 0.98, motion_mask_tau: float = 0.35) -> None:
        """
        Token-space motion weighting: amplify the drifting loss on
        the tokens that move between the past frame o_t and the future frame y_pos.
        """
        self.motion_mask_alpha = float(motion_mask_alpha)
        self.motion_mask_lambda = None if motion_mask_lambda is None else float(motion_mask_lambda)
        self.motion_mask_qhi = float(motion_mask_qhi)
        self.motion_mask_tau = float(motion_mask_tau)
        self.motion_masking_enabled = (self.motion_mask_lambda is not None
                                       and self.motion_mask_lambda > 0.0)
        if self.motion_masking_enabled:
            log.info(f"Token-space motion masking ENABLED (soft): "
                     f"alpha={self.motion_mask_alpha}, lambda={self.motion_mask_lambda}, "
                     f"qhi={self.motion_mask_qhi}, tau={self.motion_mask_tau} "
                     f"(per-token weight = 1 + lambda*tanh(alpha*relu(norm-tau)/(1-tau))*gate, "
                     f"norm = motion magnitude / per-frame quantile(qhi), "
                     f"gate = per-frame peakedness in [0, 1])")

    def _group_weight(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Turn a per-frame motion magnitude `delta` (B, T) into a per-token weight (B, T).
        Returns 1 + motion_mask_lambda * tanh(motion_mask_alpha * n) * g.
        """
        scale = torch.quantile(delta, self.motion_mask_qhi, dim=1, keepdim=True)  # (B, 1)
        norm = torch.clamp(delta / (scale + 1e-6), 0.0, 1.0)                      # (B, T)
        median = torch.median(delta, dim=1, keepdim=True).values                  # (B, 1)
        gate = torch.clamp(1.0 - median / (scale + 1e-6), 0.0, 1.0)               # (B, 1)

        tau = self.motion_mask_tau
        n = torch.clamp(norm - tau, min=0.0) / max(1.0 - tau, 1e-6)               # (B, T)
        return 1.0 + self.motion_mask_lambda * torch.tanh(self.motion_mask_alpha * n) * gate

    @torch.no_grad()
    def motion_token_weights(self, pos_act: dict, past_act: dict) -> dict:
        """
        Token-space motion weighting for the drifting loss.

        Measures how much each spatial latent cell changed between the past frame o_t
        (past_act) and the future frame y_pos (pos_act) and returns a per-token loss
        multiplier for every spatial drift-field key (`_latent*`).

        Args:
            pos_act:  dict[str, (B, T, D)] activations of the ground-truth future frame
            past_act: dict[str, (B, T, D)] activations of the past frame o_t
        Returns:
            dict[str, (B, T)] of per-token weights, one entry per spatial key that is present
            in both dicts with T > 1. Empty if motion weighting is disabled.
        """
        if not self.motion_masking_enabled:
            return {}
        groups: dict = {}
        for k, pos in pos_act.items():
            if not k.startswith("_latent") or k not in past_act:
                continue
            past = past_act[k]
            if pos.dim() != 3 or pos.shape[1] <= 1:
                continue
            delta = torch.linalg.vector_norm(
                pos.detach().float() - past.detach().float(), dim=-1)  # (B, T)
            groups.setdefault(pos.shape[1], ([], []))
            groups[pos.shape[1]][0].append(k)
            groups[pos.shape[1]][1].append(delta)
        if not groups:
            return {}

        weights = {}
        for _T, (keys, deltas) in groups.items():
            delta = torch.stack(deltas, dim=0).mean(dim=0)  # (B, T)
            weight = self._group_weight(delta)
            for k in keys:
                weights[k] = weight
        return weights
