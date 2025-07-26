with torch.no_grad():


self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
self.beta  = nn.Parameter(torch.zeros(1, 1, d_model))

h = self.encoder(x_enc)                        # still z‑scored
h = self.gamma * h * torch.sqrt(var + eps) + self.beta + mean
