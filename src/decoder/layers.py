import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GELU, CausalSelfAttention


class GPTPrediction(nn.Module):
    def __init__(self, embed_dim):
        super(GPTPrediction, self).__init__()
        n_embd = embed_dim
        self.ln1 = nn.LayerNorm(n_embd)
        self.dense = nn.Linear(n_embd, n_embd)
        self.gelu = GELU()
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.ln2(x)
        return x
    

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.x_fc1 = nn.Linear(D_features, D_hidden_features)
        self.x_fc2 = nn.Linear(D_hidden_features, D_features)
        # self.alpha = nn.Parameter(torch.ones([]) * 0.1)

    def forward(self, x, fmri=None):
        # x is (BT, HW+1, D)
        xs = self.x_fc1(x)
        xs = self.act(xs)
        xs = self.x_fc2(xs) # * self.alpha
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, cfg, embed_dim, n_head, adapter=False):
        super().__init__()
        dropout = cfg['3d_model']['stage2_model_kwargs']['mlp_drop']
        n_embd = embed_dim
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(embed_dim=n_embd, n_head=n_head, cfg=cfg)
        
        if adapter:
            self.adapter = Adapter(embed_dim)
        else:
            self.adapter = False
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # nn.GELU(),  # nice, GELU is not valid in torch<1.6
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, fmri=None):        # x: 
        x = x + self.attn(self.ln1(x), mask)   # mask: tril mask
        if self.adapter:
            x = self.adapter(x, fmri)
        x = x + self.mlp(self.ln2(x))
        return x



class Transformer_adapter(nn.Module):
    def __init__(self, cfg):
        super(Transformer_adapter, self).__init__()
        print("FAST_transformer_builder_baseline No drop Quant single")
        fast_transformer_kwargs = cfg['3d_model']['fast_transformer_kwargs']
        self.attention_type = fast_transformer_kwargs['attention_type']

        transformer_layer = cfg['3d_model']['stage2_model_kwargs']['transformer_layer']
        transformer_embed_dim = cfg['3d_model']['stage2_model_kwargs']['transformer_embed_dim']
        transformer_n_head = cfg['3d_model']['stage2_model_kwargs']['transformer_n_head']

        self.blocks = nn.ModuleList([
            Block(
                cfg, embed_dim=transformer_embed_dim,
                n_head=transformer_n_head, adapter=(_%4==0)
            ) for _ in range(transformer_layer)
        ])

        self.dec = GPTPrediction(embed_dim=transformer_embed_dim)

        self.init_with_vqvae = cfg['3d_model']['stage2_model_kwargs']['init_with_vqvae']

        # Embedding
        self.vocab_size = cfg['3d_model']['quantizer_kwargs']['embedding_num']

        if not self.init_with_vqvae:
            embed_dim = cfg['3d_model']['stage2_model_kwargs']['stage2_embed_dim']   # e.g 768
            self.emb_proj = None
            self.dec_proj = None
        else:
            embed_dim = cfg['3d_model']['quantizer_kwargs']['embedding_dim']        # e.g 32
            self.emb_proj = nn.Linear(embed_dim, cfg['stage2_model_kwargs']['stage2_embed_dim'])    # 32 --> 768
            self.dec_proj = nn.Linear(cfg['3d_model']['stage2_model_kwargs']['stage2_embed_dim'], embed_dim)    # 768 --> 32
            
        self.emb = nn.Embedding(self.vocab_size, embed_dim)       # 1024: End Token   1025: Padding Token
        position_num = cfg['3d_model']['stage2_model_kwargs']['position_num']

        self.pos_emb = nn.Embedding(position_num + 1, embed_dim)
        self.value_start = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, embed_dim)))  # Value Start Token

        # value dec 
        self.z_value_dec = nn.Linear(self.emb.weight.size(1), self.emb.weight.size(0), bias=False)    # 768 --> 1024 + 1 + 1
        self.z_value_dec.weight = self.emb.weight
        self.bias_value = nn.Parameter(torch.zeros(self.emb.weight.size(0)))

        self.sequence_length = cfg['3d_model']['stage2_model_kwargs']['sequence_length']  # 1024 + 1 = 1025

        # clip project:
        clip_dim = cfg['3d_model']['clip']['clip_embed_dim']
        self.clip_project = nn.Linear(clip_dim, embed_dim)

        self.apply(self._init_weights)
        self.config = cfg
        self.drop = nn.Dropout(cfg['3d_model']['stage2_model_kwargs']['embed_drop_p'])
        


    def forward(self, plane_index, clip_embedding):
        # Forwar the Fast Transformer Joint Model
        clip_embedding = self.clip_project(clip_embedding).unsqueeze(1)
        embedding = self.emb(plane_index)
        embedding = torch.cat([clip_embedding, self.value_start.unsqueeze(0).repeat(plane_index.shape[0], 1, 1), embedding], dim=1)     # [B, L+1, C]
        bs = embedding.shape[0]
        pos_embedding = self.pos_emb(torch.arange(embedding.shape[1], device=embedding.device).reshape(1,-1)).repeat(bs, 1, 1)
        assert embedding.shape[1] <= self.sequence_length, "Cannot Forward Sequence Length Error"
        x = embedding + pos_embedding
        mask = torch.ones(bs, 1, x.shape[1], x.shape[1])
        mask = torch.tril(mask).to(x.device)
        # Origin Transformer:
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.dec(x)
        logits_value = self.z_value_dec(x) + self.bias_value
        logits_value = logits_value[:,1:-1,:]
        return logits_value


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx