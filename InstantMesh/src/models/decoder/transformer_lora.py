# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import loralib as lora


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.in_proj = lora.Linear(embed_dim, embed_dim*3, bias=False, r=128)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections
        QKV = self.in_proj(torch.cat([x, x, x], dim=-1))  # (batch_size, seq_len, 3 * embed_dim)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)  # Split Q, K, V

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # Final linear layer
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.q_proj = lora.Linear(embed_dim, embed_dim, bias=False, r=128)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=False)
        self.v_proj = lora.Linear(self.vdim, embed_dim, bias=False, r=128)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, context):
        batch_size, seq_len, embed_dim = x.size()
        context_len = context.size(1)

        # Linear projections
        Q = self.q_proj(x)          # (batch_size, seq_len, embed_dim)
        K = self.k_proj(context)      # (batch_size, context_len, embed_dim)
        V = self.v_proj(context)    # (batch_size, context_len, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, context_len, head_dim)
        V = V.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, context_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, context_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, context_len)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # Final linear layer
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        return output

        
class BasicTransformerBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(
        self, 
        inner_dim: int, 
        cond_dim: int, 
        num_heads: int, 
        eps: float,
        attn_drop: float = 0., 
        attn_bias: bool = False,
        mlp_ratio: float = 4., 
        mlp_drop: float = 0.,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(inner_dim)
        self.cross_attn = CrossAttention(embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.self_attn = SelfAttention(embed_dim=inner_dim, num_heads=num_heads)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class TriplaneTransformer(nn.Module):
    """
    Transformer with condition that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(
        self, 
        inner_dim: int, 
        image_feat_dim: int,
        triplane_low_res: int, 
        triplane_high_res: int, 
        triplane_dim: int,
        num_layers: int, 
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, image_feats):
        # image_feats: [N, L_cond, D_cond]

        N = image_feats.shape[0]
        H = W = self.triplane_low_res
        L = 3 * H * W

        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        for layer in self.layers:
            x = layer(x, image_feats)
        x = self.norm(x)

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        return x
