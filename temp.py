class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embedding_kernel):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=embedding_kernel, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x





import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from seq2seq_alpha.layers.Embed import DataEmbedding_wo_pos  # note: there's no positional embedding 
class MambaLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.mamba_d_state,
            d_conv = configs.mamba_d_conv,  
            dt_scale = configs.mamba_dt_scale,
            dt_min=0.693 / configs.seq_len,
            dt_max=0.693 / 1,
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        # self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        # residual = x
        # x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        # x = residual + x
        return x
    

class MambaModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(configs) for _ in range(configs.e_layers)
        ])
        # self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.norm(x)
        return x
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embedkernel, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = MambaModel(configs)
        self.projection = nn.Linear(configs.d_model, configs.c_out)


    def forecast(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out  = self.encoder(enc_out)

        dec_out = self.projection(enc_out)


        return dec_out

    def forward(self, x_enc, x_mark_enc = None, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc)
        res =  dec_out[:, -1:, :]  # [B, L, D]
        return res
