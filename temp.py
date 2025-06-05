class StockFeatureMixer(nn.Module):
    """
    Collapse F features → 1 scalar per stock, separately for every stock.
    Input : (batch, T, S*F)
    Output: (batch, T, S)
    """
    def __init__(self, num_stocks: int, num_features: int, bias: bool = True):
        super().__init__()
        self.S = num_stocks
        self.F = num_features
        self.mixer = nn.Conv1d(
            in_channels = num_stocks * num_features,
            out_channels = num_stocks,          # 1 value per stock
            kernel_size = 1,
            groups = num_stocks,                # ⇐ independence
            bias = bias
        )

    def forward(self, x):
        # x: (B, T, S*F)  -->  (B, S*F, T)
        x = x.permute(0, 2, 1)
        # grouped 1×1 conv: (B, S*F, T) -> (B, S, T)
        x = self.mixer(x)
        # back to (B, T, S)
        return x.permute(0, 2, 1)