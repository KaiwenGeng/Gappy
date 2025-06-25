class ChannelMixer(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.mix = nn.Conv2d(C, 1, kernel_size=1, bias=True)  # or nn.Linear(C,1)

    def forward(self, x):
        # x : [B, C, P, D]
        y = self.mix(x).squeeze(1)        # -> [B, P, D]
        return y
