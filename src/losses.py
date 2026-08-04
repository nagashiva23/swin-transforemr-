"""Loss utilities matching the main project style."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BiomedicalTermAwareFocalLoss(nn.Module):
    def __init__(self, word_weights: torch.Tensor, gamma: float = 2.0, pad_id: int = 0, coverage_weight: float = 0.1, coverage_threshold: float = 1.0):
        super().__init__()
        self.register_buffer("word_weights", word_weights)
        self.gamma = gamma
        self.pad_id = pad_id
        self.coverage_weight = coverage_weight
        self.coverage_threshold = coverage_threshold

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, attention: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probabilities = target_log_probs.exp()
        valid = targets.ne(self.pad_id).float()
        weights = self.word_weights[targets]
        focal = -weights * (1.0 - probabilities).pow(self.gamma) * target_log_probs
        caption_loss = (focal * valid).sum() / valid.sum().clamp(min=1)

        total_attention = attention.sum(dim=1)
        coverage_loss = F.relu(total_attention - self.coverage_threshold).pow(2).mean()
        total_loss = caption_loss + self.coverage_weight * coverage_loss
        return total_loss, {"caption": float(caption_loss.detach()), "coverage": float(coverage_loss.detach())}
