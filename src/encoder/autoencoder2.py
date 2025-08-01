
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TransBlock
from src.layers import get_2d_sincos_pos_embed


class PatchEmbed2D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=1, embed_dim=768, flatten=True, bias=True):
        super().__init__()
        assert isinstance(img_size, tuple)
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

        self.num_h = img_size[0] // patch_size
        self.num_w = img_size[1] // patch_size
        self.num_patches = self.num_h * self.num_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape # batch, channel, voxels
        assert H == self.img_size[0]
        assert W == self.img_size[1]

        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous() # NCHW -> NLC

        return x # [B, L, C]


class fMRI_Encoder(nn.Module):
    def __init__(self, config):
        super(fMRI_Encoder, self).__init__()
        patch_size = config.patch_size
        image_size = tuple(config.image_size)

        embed_dim = config.embed_dim
        num_head = config.num_heads
        drop_p = config.drop_rate
        in_chans = config.in_chans
        # img_dim = 2048

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed2D(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       proj_drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.encoder_depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.clip_norm = nn.LayerNorm(embed_dim)
        
        # self.pred = nn.Linear(embed_dim, img_dim, bias=True)
        # --------------------------------------------------------------------------
        self.initialize_weights()
        self.in_chans = in_chans

        self.aggregation = nn.Linear(64*6, 4, bias=True)


    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.patch_embed.num_h, self.patch_embed.num_w), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        h = self.patch_embed.num_h
        w = self.patch_embed.num_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        return imgs


    def forward_encoder_wo_pred_clip(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        i=0
        # apply Transformer blocks
        for blk in self.blocks:
            i+=1
            x = blk(x)
            if i==12:
                fmri_feature = self.clip_norm(x[:,:1,:])
        x = self.norm(x)

        return x, fmri_feature
    
    def forward_encoder_only_clip(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        i=0
        # apply Transformer blocks
        for blk in self.blocks:
            i+=1
            x = blk(x)
            if i==12:
                return self.clip_norm(x[:,:1,:])
            
    def forward_encoder_wo_pred(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_encoder_w_pred(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:,1:,:])
        z = self.pred(x)
        return z

    def forward_encoder_w_pred_all(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = self.pred(x)
        return z
    
    def forward_encoder(self, x):
        # x: [B, C, H ,W]
        
        x = self.patch_embed(x) # [B, K, C] (B, 16*16, 1024)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
  
        z = self.pred(x[:, :1])
        return z


    def forward(self, fmri, mask=None):
        # fmri: [B, 1, K]
        latent = self.forward_encoder(fmri)
        return latent

class fMRI_Encoder_1D(nn.Module):
    def __init__(self, config):
        super(fMRI_Encoder_1D, self).__init__()
        embed_dim = config.embed_dim
        num_head = config.num_heads
        drop_p = config.drop_rate

        self.linear_in = nn.Linear(6*11249, 1024)
        self.embedding = nn.Linear(1, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024 + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       proj_drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.encoder_depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.clip_norm = nn.LayerNorm(embed_dim)
        
        # self.pred = nn.Linear(embed_dim, img_dim, bias=True)
        # --------------------------------------------------------------------------
        self.initialize_weights()

        self.aggregation = nn.Linear(256, 4, bias=True)


    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (16, 16), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.embedding.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder_wo_pred_clip(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.linear_in(x)
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, d_model)
        x = x + self.pos_embed

        i=0
        # apply Transformer blocks
        for blk in self.blocks:
            i+=1
            x = blk(x)
            if i==12:
                fmri_feature = self.clip_norm(x[:,:1,:])
        x = self.norm(x)

        return x, fmri_feature
   
       
    def forward_encoder_wo_pred(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.linear_in(x)
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, d_model)
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def forward_encoder_only_clip(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.linear_in(x)
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, d_model)
        x = x + self.pos_embed

        i=0
        # apply Transformer blocks
        for blk in self.blocks:
            i+=1
            x = blk(x)
            if i==12:
                return self.clip_norm(x[:,:1,:])