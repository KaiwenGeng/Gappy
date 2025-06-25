class TokenEmbedding(nn.Module):
    """
    x  shape  ⟨B, L, N_exo + 1⟩   (last column = endogenous return)
    out shape ⟨B, L, d_model⟩
    """
    def __init__(self, n_exo: int, d_model: int):
        super().__init__()
        self.endo_conv = nn.Conv1d(1,          d_model, 3, padding=1,
                                   padding_mode='circular', bias=False)
        self.exo_conv  = nn.Conv1d(n_exo,      d_model, 3, padding=1,
                                   padding_mode='circular', bias=False)

        # *feature-wise* gating that learns when to use the exogenous path
        self.gate = nn.GLU(dim=-1)             #  ➜ keeps code tiny

        # optional projection back to d_model
        self.project = nn.Linear(2*d_model, d_model, bias=False)

        for m in self.modules():                       # Kaiming init
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                         nonlinearity='leaky_relu')

    def forward(self, x):
        exo, endo = x[..., :-1], x[..., -1:]           # split features
        # Conv along time dim
        endo_emb = self.endo_conv(endo.permute(0, 2, 1)).transpose(1, 2)
        exo_raw  = self.exo_conv (exo .permute(0, 2, 1)).transpose(1, 2)

        # GLU:  exo_raw = [value, gate];   value * σ(gate)
        exo_emb  = self.gate(torch.cat([exo_raw, exo_raw], dim=-1))

        y = torch.cat([endo_emb, exo_emb], dim=-1)     # [B,L,2d]
        return self.project(y)                         # [B,L,d]
