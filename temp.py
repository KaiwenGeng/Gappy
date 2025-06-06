import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth‑L1 (Huber) loss with optional per‑sample weights.

    Forward signature
    -----------------
        loss = crit(input, target, weight=None)

    Arguments
    ----------
    input   : (N, 1) or (N,)
    target  : (N,)
    weight  : (N,) or None          – non‑negative weights
    beta    : float  – threshold between L1 and L2 regions
    reduction : "mean" | "sum" | "none"
    """
    def __init__(self, beta: float = 1.0, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.beta = float(beta)
        self.reduction = reduction
        self.eps = eps        # avoids /0 when weight.sum()==0

    def forward(
        self,
        input:  torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None
    ) -> torch.Tensor:

        # No weighting requested → behave like nn.SmoothL1Loss
        if weight is None:
            return F.smooth_l1_loss(
                input, target, beta=self.beta, reduction=self.reduction
            )

        # Flatten to (N,)
        input  = input.view(-1)
        target = target.view(-1)
        weight = weight.view(-1).to(input.dtype)

        # Per‑element Smooth‑L1 (Huber) error
        diff = torch.abs(input - target)
        per_elem = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        # Apply weights (still 1‑to‑1 with samples here)
        loss_per_sample = weight * per_elem

        if self.reduction == "none":
            return loss_per_sample          # (N,)

        if self.reduction == "sum":
            return loss_per_sample.sum()

        # self.reduction == "mean"
        return loss_per_sample.sum() / (weight.sum() + self.eps)


#  conda install cudatoolkit==11.8 -c nvidia
#  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# !pip install causal_conv1d==1.4.0
# !pip install mamba_ssm==2.2.2