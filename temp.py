class VariableSelection(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim)
        )
    def forward(self, x):                       # (B, T, 6)
        w = torch.softmax(self.net(x), dim=-1)  # (B, T, 6)
        return x * w, w  
