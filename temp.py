Sps I'm training a model, where the dataset is very noisy. ie, it's very unlikely to get sth that's even close to 0.1 IC

I use a day's universe  (changes slightly everyday)  as a batch, and lookback is 126 days (can change). My input is therefore [3000,126,6], 6 means 6 features per stock (log residual log return, volume, sell side volume, buy side volume, sell short volume).

My target is next 10 day return (a scaler). All these data are revisualized already and has mean 0 and std 1. Also these are processedby ranking, not original value. we also have market cap as sample weight.  we are using rolling window for training, ie typically 5 - 10 years to train and another year to val.

I designed a mamba ssm model like this. I start with embedding like:
class TokenEmbedding(nn.Module):

def init(self, c_in, d_model, embedding_kernel):

super(TokenEmbedding, self).init()

self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,

kernel_size=embedding_kernel, padding_mode='zeros', bias=False)

for m in self.modules():

if isinstance(m, nn.Conv1d):

nn.init.kaiming_normal_(

m.weight, mode='fan_in', nonlinearity='leaky_relu')

def forward(self, x):

x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

return x

and the mamba model looks like this:
import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from mamba_ssm import Mamba

from seq2seq_alpha.layers.Embed import DataEmbedding_wo_pos  # note: there's no positional embedding

class MambaLayer(nn.Module):

def init(self, configs):

super().init()

self.mamba = Mamba(

d_model = configs.d_model,

d_state = configs.mamba_d_state,

d_conv = configs.mamba_d_conv,

dt_scale = configs.mamba_dt_scale,

dt_min=0.693 / configs.seq_len,

dt_max=0.693 / 1,

)

self.dropout = nn.Dropout(p=configs.dropout)

self.norm = nn.LayerNorm(configs.d_model)

def forward(self, x):

residual = x

x = self.norm(x)

x = self.mamba(x)

x = self.dropout(x)

x = residual + x

return x

class MambaModel(nn.Module):

def init(self, configs):

super().init()

self.layers = nn.ModuleList([

MambaLayer(configs) for _ in range(configs.e_layers)

])

self.norm = nn.LayerNorm(configs.d_model)

def forward(self, x):

for layer in self.layers:

x = layer(x)

x = self.norm(x)

return x

class Model(nn.Module):

def init(self, configs):

super(Model, self).init()

self.pred_len = configs.pred_len

Embedding

self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embedkernel, configs.embed, configs.freq,

configs.dropout)

Encoder

self.encoder = MambaModel(configs)

self.projection = nn.Linear(configs.d_model, configs.c_out)

def forecast(self, x_enc, x_mark_enc):

Embedding

enc_out = self.enc_embedding(x_enc, x_mark_enc)

enc_out  = self.encoder(enc_out)

dec_out = self.projection(enc_out)

return dec_out

def forward(self, x_enc, x_mark_enc = None, mask=None):

dec_out = self.forecast(x_enc, x_mark_enc)

res =  dec_out[:, -1:, :]  # [B, L, D]

return res

you can change the parameter listed here:
def init(

self,

d_model,

d_state=16,

d_conv=4,

expand=2,

dt_rank="auto",

dt_min=0.001,

dt_max=0.1,

dt_init="random",

dt_scale=1.0,

dt_init_floor=1e-4,

conv_bias=True,

bias=False,

use_fast_path=True,  # Fused kernel options

layer_idx=None,

device=None,

dtype=None,

):

Currently:
I have a mamba:
d_model = 48 (embedding 6 features to 48 via a con1d) and d_state = 16, dt_min = 0.693 / 126, dt_max = 0.693 / 2, , embedding_kernel = 3, dt_rank = 4,  the total number of parameters is around 20k
now I wanna switch to multiscale, what should I give to x and y?
class MultiScaleTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, k_list=[x,y]): 
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

if I want my model to perform better on large cap, should we make x, y larger or smaller?
