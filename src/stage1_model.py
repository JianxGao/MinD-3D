import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import trimesh
from utils import libmcubes
from utils.libmise import MISE
from utils.libsimplify import simplify_mesh
from src.common import make_3d_grid, normalize_coord
from omegaconf import OmegaConf
# fmri encoder
from src.encoder.autoencoder import fMRI_Encoder
from src.encoder.feature_aggregation import Feature_Aggregation, Clip_Header
# diffusion prior
from src.diffusion.fbdm import DiffusionNetwork, FBDM
# # 3d decoder
# from src.decoder.lad import Latent_Adapted_Decoder
import clip
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from torch import distributed as dist


class MinD3D(nn.Module):
    def __init__(self, config, device=None, dataset=None):
        super(MinD3D, self).__init__()
        self.device = device
        # fMRI encoder
        self.fmri_encoder = fMRI_Encoder(config).to(device)
        self.fa_module = Feature_Aggregation(2048).to(device)
        self.clip_header = Clip_Header(2048).to(device)
        
        # Diffusion prior
        d_prior_path = config['diff_prior_config_path']
        self.d_prior = OmegaConf.load(d_prior_path)
        self.prior_network = DiffusionNetwork(**self.d_prior["diffusion_prior"])
        self.diffusion_prior = FBDM(
            net = self.prior_network,
            timesteps = self.d_prior["timesteps"],
            sample_timesteps = self.d_prior["sample_timesteps"],
            cond_drop_prob = self.d_prior["cond_drop_prob"],
        ).to(device)
        
        # Code for training
        self.clip_path = config['3d_model']['clip']['model_name']
        print("Loading CLIP Model from", self.clip_path)
        self.CLIP_Model, self.img_transform = clip.load(self.clip_path, device=device) 
        self.CLIP_Model.float()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [self.logit_scale]},
            {'params': [p for n, p in self.clip_header.named_parameters()]},
            {'params': [p for n, p in self.fmri_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.fmri_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fa_module.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.fa_module.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        for name, param in self.CLIP_Model.named_parameters():
            param.requires_grad = False
        self.opt = AdamW(params=optimizer_parameters,
            lr=float(config["training"]["lr"]),
            betas=(0.9, 0.95)
        )

        self.sche = get_linear_schedule_with_warmup(
            self.opt, 
            num_warmup_steps=config["training"]['warmup_iters'],
            num_training_steps=config["training"]["max_iters"]
        )

        # Generation Parameter
        self.points_batch_size = 100000
        self.threshold=config['test']['threshold']
        self.resolution0=config['generation']['resolution_0']
        self.upsampling_steps=config['generation']['upsampling_steps']
        self.sample=config['generation']['use_sampling']
        self.refinement_step=config['generation']['refinement_step']
        self.simplify_nfaces=config['generation']['simplify_nfaces']
        self.input_type=config['data']['input_type']
        self.padding=config['data']['padding']
        self.vol_bound = None
        self.with_normals = False
        self.world_size = dist.get_world_size()
    
    @torch.autocast("cuda", dtype=torch.float16)  # fp16 is option
    def get_loss(self, meta):
        x = meta["fmri"].cuda()
        b,f,w,h = x.size()

        # Encode the fMRI frames
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        x = self.fmri_encoder.forward_encoder_w_pred_all(x)
        x_cls,x = x[:,:1,:], x[:,1:,:]
        fmri_embed = self.fa_module(x, b, f)         # [B, 768, 512]
        fmri_feature = self.clip_header(x_cls, b, f) # [B,   1, 512]
        
        # calculate diffusion loss
        img = meta["img"].cuda()
        b,f_i,c,w,h = img.size()
        self.CLIP_Model.eval()
        with torch.no_grad(): 
            img = img.view(b*f_i, 3, 224, 224).contiguous()
            clip_img_feature = self.CLIP_Model.encode_image(img)
            clip_img_feature = clip_img_feature.view(b,f_i,512).contiguous()
            image_features = torch.mean(clip_img_feature, dim=1).view(b,512).contiguous()

        diff_loss, pred = self.diffusion_prior(fmri_embed = fmri_embed, image_embed = image_features)
        # contrastive loss
        clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.logit_scale.exp())
        return diff_loss, clip_loss, self.logit_scale


    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.float16)  # fp16 is option
    def get_test_metric(self, meta):
        x = meta["fmri"].cuda()
        b,f,w,h = x.size()

        # Encode the fMRI frames
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        x = self.fmri_encoder.forward_encoder_w_pred_all(x)
        x_cls,x = x[:,:1,:], x[:,1:,:]
        fmri_embed = self.fa_module(x, b, f)         # [B, 768, 512]
        fmri_feature = self.clip_header(x_cls, b, f) # [B,   1, 512]
        
        # calculate diffusion loss
        img = meta["img"].cuda()
        b,f_i,c,w,h = img.size()
        img = img.view(b*f_i, 3, 224, 224).contiguous()
        clip_img_feature = self.CLIP_Model.encode_image(img)
        clip_img_feature = clip_img_feature.view(b, f_i, 512).contiguous()
        image_features = torch.mean(clip_img_feature,dim=1).view(b,512).contiguous()

        diff_loss, pred = self.diffusion_prior(fmri_embed = fmri_feature, image_embed = image_features)
        # contrastive loss
        clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.logit_scale.exp())

        return diff_loss, clip_loss


    def cal_clip_loss(self, image_features, fmri_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_fmri = self.get_logits(image_features, fmri_features.squeeze(1), logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_fmri, labels)
        ) / 2
        return total_loss

    def get_logits(self, image_features, fmri_features, logit_scale):
        logits_per_image = logit_scale * image_features @ fmri_features.T
        logits_per_fmri = logit_scale * fmri_features @ image_features.T
        return logits_per_image, logits_per_fmri

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()