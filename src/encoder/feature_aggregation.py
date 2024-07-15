import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Feature_Aggregation(nn.Module):
    def __init__(self, n_embd):
        super(Feature_Aggregation, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(256, 128)
        self.gelu = nn.GELU()
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc2 = nn.Linear(2048, 512)

    def forward(self, x, b, f):
        x = self.ln1(x)
        x = rearrange(x, '(b f) l c -> b f c l',b=b,f=f)
        x = self.fc1(x)
        x = self.gelu(x)
        x = rearrange(x, 'b f c l -> b (f l) c')
        x = self.ln2(x) # [b, f*128, 2048]
        x = self.fc2(x) # [b, f*128, 512]
        return x