class VariableSelection(nn.Module):
    def __init__(self, in_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = in_dim * hidden_mult

        # feature branch – sees only x
        self.x_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )

        # mask branch – sees only the (0/1) mask
        # A much smaller network is enough because the signal is binary
        self.m_net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),  # learnable per‑feature bias
        )

    def forward(self, x, mask):
        """
        x, mask: shape (B, T, D); mask == 1 for observed values
        """
        # logits from each branch
        logits_x = self.x_net(x)              # (B, T, D)
        logits_m = self.m_net(mask.float())   # (B, T, D)

        logits = logits_x + logits_m          # sum, no concat
        logits = logits.masked_fill(mask == 0, torch.finfo(x.dtype).min)

        # masked soft‑max, safe when every feature is missing
        w = torch.softmax(logits, dim=-1) * mask
        w = w / (w.sum(-1, keepdim=True) + 1e-8)

        return x * w   
