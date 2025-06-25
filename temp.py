class TokenEmbedding_GLU(nn.Module):
    """
    x  shape  ⟨B, L, N_exo + 1⟩   (last column = endogenous return)
    out shape ⟨B, L, d_model⟩
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.endo_conv = nn.Conv1d(1,          d_model, 3, padding=1,
                                   padding_mode='circular', bias=False)
        self.exo_conv  = nn.Conv1d(c_in,      2*d_model, 3, padding=1, 
                                   padding_mode='circular', bias=False)

        # *feature-wise* gating that learns when to use the exogenous path
        self.gate = nn.GLU(dim=-1)             #  ➜ keeps code tiny

        for m in self.modules():                       # Kaiming init
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                         nonlinearity='leaky_relu')

    def forward(self, x):
        _, endo = x[..., :-1], x[..., -1:]           # split features
        # Conv along time dim
        endo_emb = self.endo_conv(endo.permute(0, 2, 1)).transpose(1, 2)
        exo_raw  = self.exo_conv (x .permute(0, 2, 1)).transpose(1, 2)

        # GLU:  exo_raw = [value, gate];   value * σ(gate)
        exo_emb = self.gate(exo_raw)  

        y = endo_emb + exo_emb
        return y  
