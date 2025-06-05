import torch
import torch.nn as nn
import torch.nn.functional as F

class LN_LSTMCell(nn.Module):
    """Layer‑norm LSTM cell (Ba et al., 2016)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ln = nn.LayerNorm(4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        gates = self.ln(self.ih(x) + self.hh(h))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class LN_LSTM(nn.Module):
    """Stacked layer‑norm LSTM (uses LN_LSTMCell)."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(LN_LSTMCell(
                input_size if l == 0 else hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        B, T, _ = x.shape
        h = x
        for cell in self.layers:
            h_t, c_t = torch.zeros(B, cell.hidden_size, device=x.device), \
                       torch.zeros(B, cell.hidden_size, device=x.device)
            outs = []
            for t in range(T):
                h_t, c_t = cell(h[:, t], (h_t, c_t))
                outs.append(h_t)
            h = torch.stack(outs, dim=1)  # [B, T, H]
            h = self.dropout(h)
        return h  # final layered output sequence

class AttnPool(nn.Module):
    """Bahdanau attention pooling across timeline."""
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
        self.W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h):
        # h: [B, T, H]
        scores = torch.tanh(self.W(h)) @ self.q      # [B, T]
        α = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (α * h).sum(dim=1)                    # [B, H]

class StockLSTM(nn.Module):
    def __init__(
        self,
        n_feat=6,
        proj_dim=64,
        hidden_dim=128,
        num_layers=2,
        ffn_dim=64,
        dropout=0.1
    ):
        super().__init__()
        self.proj = nn.Linear(n_feat, proj_dim)
        self.encoder = LN_LSTM(proj_dim, hidden_dim, num_layers, dropout)
        self.attnpool = AttnPool(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [N_d, T, 6]
        x = self.proj(x)
        h = self.encoder(x)
        z = self.attnpool(h)          # [N_d, hidden_dim]
        z = z + self.head(z)          # residual
        return self.out(z).unsqueeze(1)  # ➜ [N_d, 1, 1]
