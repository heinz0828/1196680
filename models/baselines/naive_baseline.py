import torch
import torch.nn as nn


class NaiveBaseline(nn.Module):
    """Naive baseline: predict return = 0 (price stays the same).

    No trainable parameters. Serves as the absolute performance floor.
    """

    def __init__(self, horizon: int = 1):
        super().__init__()
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, F) — ignored
        Returns:
            (B, horizon) — all zeros (predict no change)
        """
        B = x.shape[0]
        return torch.zeros(B, self.horizon, device=x.device)
