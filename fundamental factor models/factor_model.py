import torch.nn as nn
class LinearFactorModel(nn.Module):
    """
    r_hat =  B  ·  beta ,  shared across all stocks.
    B : (B, K, F)   -> r_hat : (B, K)
    """
    def __init__(self, n_factors):
        super().__init__()
        self.beta = nn.Linear(n_factors, 1, bias=False) # warning: the bias term MUST BE FALSE

    def forward(self, B):
        out = self.beta(B)            # (B, K, 1)
        return out.squeeze(-1)        # (B, K)