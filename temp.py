class UniPatchInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 2)   # 5 exogenous → 2 channels
        self.glu  = nn.GLU(dim=-1)    # → 1 channel (gated)
    def forward(self, x):
        endo = x[..., 5:6]            # endogenous
        exo  = x[..., :5]             # exogenous
        exo_mix = self.glu(self.proj(exo))
        uni     = endo + exo_mix      # final [B, 3000, 121, 1]
        return uni
