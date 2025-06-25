class TokenEmbedding(nn.Module):
    """
    Endogenous token  : last column   (…, -1)
    Exogenous tokens  : all others    (…, :-1)
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        c_exo = c_in - 1                       # everything except the return itself

        # ① separate convolutions
        self.endo_conv = nn.Conv1d(1, d_model // 2,
                                   kernel_size=3, padding=1, bias=False)
        self.exo_conv  = nn.Conv1d(c_exo, d_model // 2,
                                   kernel_size=3, padding=1,
                                   groups=c_exo, bias=False)  # depth-wise -> keeps each
                                                               # variable independent

        # ② feature-wise gating network (1 value per exogenous channel)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),           # (B, c_exo, T) -> (B, c_exo, 1)
            nn.Flatten(1),                     # (B, c_exo)
            nn.Linear(c_exo, c_exo, bias=True),
            nn.Sigmoid()
        )

        # kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_in",
                                        nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (B, T, C)   with C = c_exo + 1
        endo = x[..., -1:].permute(0, 2, 1)      # (B, 1, T)
        exo  = x[..., :-1].permute(0, 2, 1)      # (B, c_exo, T)

        # ③ learn a relevance weight for every exogenous variable
        weights = self.gate(exo).unsqueeze(-1)   # (B, c_exo, 1)
        exo = exo * weights                     # “squeeze-and-excite” style gating

        # ④ encode each path separately, then concatenate
        h_endo = self.endo_conv(endo)            # (B, d_model/2, T)
        h_exo  = self.exo_conv(exo)              # (B, d_model/2, T)

        h = torch.cat([h_endo, h_exo], dim=1)    # (B, d_model, T)
        return h.transpose(1, 2)                 # back to (B, T, d_model)
