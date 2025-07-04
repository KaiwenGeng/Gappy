class SEBlock(nn.Module):
    """
    Squeeze‑and‑Excitation for 1‑D sequences.
    x shape expected: [batch, length, channels]
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)          # squeeze over time
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [B, L, C] -> [B, C, L] for pooling
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)   # [B, C]
        y = self.fc(y).unsqueeze(1)                        # [B, 1, C]
        return x * y    
