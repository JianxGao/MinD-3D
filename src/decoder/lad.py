import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from src.decoder.encoder import encoder_dict
from src.decoder.conv_onet import models as VQVAE
from src.decoder.layers import Transformer_adapter


from PIL import Image


class Latent_Adapted_Decoder(nn.Module):
    def __init__(self, cfg, device=None):
        super(Latent_Adapted_Decoder, self).__init__()

        self._device = device
        self.config = cfg
        self.name = 'Latent Adapted Decoder'
        print("Using 3D Decoder: ",self.name)
        self.autoregressive_steps = cfg['3d_model']['stage2_model_kwargs']['sequence_length'] - 1  # 1025 - 1 = 1024
        self.reso = cfg['3d_model']['encoder_kwargs']['plane_resolution']  # 64
        self.stage1_reduce_dim = cfg['3d_model']['quantizer_kwargs']['reduce_dim']

        # Get Occupancy Network
        decoder = cfg['3d_model']['decoder']
        encoder = cfg['3d_model']['encoder']
        dim = cfg['data']['dim']
        c_dim = cfg['3d_model']['c_dim']
        decoder_kwargs = cfg['3d_model']['decoder_kwargs']
        encoder_kwargs = cfg['3d_model']['encoder_kwargs']
        padding = cfg['data']['padding']

        # VQVAE Quantizer
        quantizer = cfg['3d_model']['quantizer']
        embed_num = cfg['3d_model']['quantizer_kwargs']['embedding_num']
        embed_dim = cfg['3d_model']['quantizer_kwargs']['embedding_dim']
        self.codebook_dim = embed_dim
        self.embed_num = embed_num
        beta = cfg['3d_model']['quantizer_kwargs']['beta']
        unet2d_kwargs = cfg['3d_model']['encoder_kwargs']['unet_kwargs']
        c_dim = cfg['3d_model']['c_dim']
        quantizer = VQVAE.quantizer_dict[quantizer](cfg, n_e=embed_num,e_dim=embed_dim,beta=beta,c_dim=c_dim,unet_kwargs=unet2d_kwargs)

        # VQVAE Decoder
        decoder = VQVAE.decoder_dict[decoder](dim=dim, c_dim=c_dim, padding=padding,**decoder_kwargs).to(device)

        # VQVAE Encoder
        encoder = encoder_dict[encoder](dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs).to(device)

        self.vqvae_model = VQVAE.ConvolutionalOccupancyNetwork(
            decoder, quantizer, encoder, device=device
        )

        self.vqvae_model.eval()

        self.transformer_model = Transformer_adapter(cfg).to(device)

        self.nll_loss_value = CrossEntropyLoss()  
        
    def forward(self):
        raise NotImplementedError
    

    def get_fmri2shape_loss(self, fmri_embedding, plane_index):
        """Get 3Plane Joint Losses  (Value,Position) """
        nll_loss = 0
        logits_value = self.transformer_model(plane_index, fmri_embedding)
        _, _, value_dim = logits_value.shape
        # Value Loss:
        nll_value_loss = self.nll_loss_value(logits_value.reshape(-1, value_dim), plane_index.reshape(-1))
        return nll_value_loss
    

    @torch.no_grad()
    def Stage1_encode_inputs(self, inputs):
        '''
            Stage1 VQVAE Encoder --> Encode inputs
        '''
        c = self.vqvae_model.encode_inputs(inputs)
        return c


    @torch.no_grad()
    def Stage1_decode(self, p, c, **kwargs):
        '''
            Stage1 VQVAE Decoder --> returns occupancy probabilities for the sampled point
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_r = self.vqvae_model.decode(p, c, **kwargs)
        return p_r
    

    @torch.no_grad()
    def Stage1_quantize(self, inputs):
        '''
            Stage1 VQVAE Quantize ---> Quantize the Feature Grid
        '''
        c_quantize, loss_dict, info_dict = self.vqvae_model.quantize(inputs)
        return c_quantize, loss_dict, info_dict
    

    @torch.no_grad()
    def Stage1_quantize_get_index(self, inputs):
        '''
            Stage1 Quantize Without Unet ---> Get index
        '''
        _, _, info_dict = self.vqvae_model.quantize_get_index(inputs)
        return _, _, info_dict

    
    def backward(self, loss=None):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def restore_from_stage1(self):
        load_path = self.config['stage1_load_path']
        if os.path.exists(load_path):
            print('=> Loading Stage1 Model From {}\n'.format(load_path))
            state_dict = torch.load(load_path, map_location='cpu')
            self.vqvae_model.load_state_dict(state_dict['model'])
    
    
    @torch.no_grad()
    def batch_predict_cond_from_fmri_embedding(self, fmri_embedding, bs, device,top_k=250,greed=False, gt = None):
        index = self.batch_predict_cond_from_prompt(fmri_embedding, bs, device=device, top_k=top_k, greed=greed, gt = gt)
        feature_plane_shape = (bs, 32, 32, self.stage1_reduce_dim)
        plane_xz, plane_xy, plane_yz = self.vqvae_model.quantizer.get_codebook_entry(index, feature_plane_shape)
        return plane_xz, plane_xy, plane_yz


    @torch.no_grad()
    def batch_predict_cond_from_prompt(self, embedding, bs, device, temperature=1, top_k=250,greed=False, gt = None):
        steps = 1024
        if gt is not None:
            return gt.to(device)
        x = torch.zeros(bs, 1024, dtype=torch.long).to(device)
        for i in tqdm(range(steps)):
            logits = self.transformer_model(x[:,:i+1], embedding)
            logits = logits[:,i,:] / temperature
            if top_k:
                logits = self.top_k_logits(logits, top_k)
            
            probs_value = F.softmax(logits, dim=-1)
            if not greed:
                ix = torch.multinomial(probs_value, num_samples=1)

            else:
                _, ix = torch.topk(probs_value, k=1, dim=-1)
            ix = ix.squeeze(1)
            x[:, i] = ix
        x = x.reshape(bs, 1024)
        return x


    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

                



        
        


        
        

