class VariableSelection(nn.Module):
    def __init__(self, in_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = in_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )

    def forward(self, x, mask):
        # x, mask: (B, T, in_dim) ; mask = 1 if value is observed
        logits = self.net(x)
        logits = logits.masked_fill(mask == 0, -1e9)   # forbid missing feats
        w = torch.softmax(logits, dim=-1)               # (B,T,in_dim)
        return x * w 
