class MultiScaleTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, k_list=[1,3,5]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(c_in, d_model // len(k_list),
                      kernel_size=k, padding=k//2, bias=False)
            for k in k_list])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                 # x: [B, L, C]
        x = x.permute(0,2,1)              # [B, C, L] for conv
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)        # concat channel-wise
        return self.norm(x.transpose(1,2))
