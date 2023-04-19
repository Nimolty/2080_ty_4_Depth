import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import os
import glob
from torch import batch_norm, einsum
from einops import rearrange, repeat
import math

class MHCA_ein(nn.Module):
    def __init__(self, num_heads, inp_dim,hid_dim, n, pos_embed=True):
        super().__init__()
        self.hid_dim = hid_dim
        # self.v_dim = v_dim
        self.inp_dim = inp_dim
        self.n_heads = num_heads
        self.n = n
        self.pos_embed_bool = pos_embed
        
        assert self.hid_dim % self.n_heads == 0
        
        self.w_q = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_k = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        self.w_v = nn.Linear(self.inp_dim, self.hid_dim, bias=False)
        
        self.fc = nn.Linear(self.hid_dim, self.inp_dim)
        self.scale = math.sqrt(self.hid_dim // self.n_heads)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_heads, self.n, self.n))
    
    def forward(self, query, key, value):
        b, n, m, h = *value.shape, self.n_heads
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        #print(Q.shape)
        Q = rearrange(Q, "b n (h d) -> b h n d", h=h)
        K = rearrange(K, "b n (h d) -> b h n d", h=h)
        V = rearrange(V, "b n (h d) -> b h n d", h=h)
        energy = einsum("b h i d, b h j d -> b h i j", Q, K) / self.scale
        # print("pos_embed_bool", self.pos_embed_bool)
        
        if self.pos_embed is not None and self.pos_embed_bool: 
            #print(energy.shape)
            energy = energy + self.pos_embed
        attn = torch.softmax(energy, dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, V)
        out = rearrange(out, "b h n d -> b n (h d)", b=b)
        #print("out.shape", out.shape)
        out = self.fc(out)
        return out

def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, \
                                  num_layers)
        self.num_layers = num_layers
        
    def forward(self, query, key, value):
        output = query
        for layer in self.layers:
            output = layer(output, key, value)
        
        return output
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_inp,d_model, d_out, d_ffn=1024,
                 dropout=0.1, n_k=21,
                 n_heads=1, pos_embed=True):
        # d_out = d_model * n_heads
        super().__init__()
        self.d_model = d_model
        self.d_inp = d_inp
        self.d_out = d_out
        self.d_ffn = d_ffn
        self.n_k = n_k
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_hid = self.d_model * self.n_heads
        
        # cross attention
        self.cross_attn = MHCA_ein(self.n_heads, self.d_inp, self.d_hid,self.n_k, pos_embed=pos_embed)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_inp)
        
        # ffn
        self.linear1 = nn.Linear(self.d_inp, self.d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.d_ffn, self.d_inp)
        self.dropout4 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_inp)
        
        # fcs
        self.fc1 = nn.Linear(self.d_inp * self.n_k, self.d_ffn)
        self.fc2 = nn.Linear(self.d_ffn, self.d_out)
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, query, key, value):
        # cross-attention
        tgt = self.cross_attn(query, key, value)
        query = tgt + self.dropout1(query)
        query = self.norm1(query)
        
        # ffn
        query = self.forward_ffn(query)
        
        # fc
        query = self.fc2(F.gelu(self.fc1(query.flatten(1))))
        
        
        return query
        
if __name__ == "__main__":
    transformer = TransformerEncoder(
                TransformerEncoderLayer(d_inp=3, d_out=7, d_model=64, n_k=21, n_heads=1, pos_embed=True), num_layers=1
                 )
    a = torch.randn(10, 21, 3)
    b = transformer(a, a, a)
    print("b.shape", b.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    