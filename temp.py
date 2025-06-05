import torch, torch.nn as nn, torch.nn.functional as F

class StockLSTMAttention(nn.Module):
    def __init__(self, nvars=6, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=nvars,
            hidden_size=hidden,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)          # scores e_t
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):                  # x: [B, T, F]
        h, _ = self.lstm(x)               # h: [B, T, 2H]
        w = self.attn(h).squeeze(-1)      # [B, T]
        α = torch.softmax(w, dim=-1).unsqueeze(-1)
        context = (h * α).sum(dim=1)      # [B, 2H]
        out = self.mlp(self.norm(context))# [B, 1]
        return out
