import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
from src.zero123plus.pipeline import RefOnlyNoisedUNet
from src.encoder.autoencoder2 import fMRI_Encoder
from src.layers import torch_init_model
from peft import LoraConfig, get_peft_model

import importlib
from torch import distributed as dist
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
import open_clip


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVDiffusion(nn.Module):
    def __init__(
        self,
        args,
        stable_diffusion_config,
        drop_cond_prob=0.1,
        fmri_encoder_config=None,
        logdir=None
    ):
        super(MVDiffusion, self).__init__()

        self.drop_cond_prob = drop_cond_prob

        self.register_schedule()

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline = pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(self.pipeline.unet, train_sched, self.pipeline.scheduler)
        
        self.train_scheduler = train_sched      # use ddpm scheduler during training

        self.unet = pipeline.unet

        print('Loading custom white-background unet ...')
        state_dict = torch.load("./pretrained_models/diffusion_pytorch_model.bin", map_location='cpu')
        new_ckpt = {"unet."+key: value for key, value in state_dict.items()}
        self.unet.load_state_dict(new_ckpt, strict=True)
        # Configuration for LoRA
        lora_config = LoraConfig(
            r=64,  # The rank of the update matrices
            lora_alpha=64,  # Alpha scaling factor
            lora_dropout=0.1,  # Dropout applied to LoRA weights during training
            target_modules=["attn1.to_q", "attn1.to_v", "attn2.to_q", "attn2.to_v"],  # Specify which modules to add LoRA adapters to
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        self.fmri_encoder = fMRI_Encoder(fmri_encoder_config)
        self.pipeline.fmri_encoder = self.fmri_encoder

        self.i_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.t_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # validation output buffer
        self.validation_step_outputs = []
        fmri_encoder_ckpt = torch.load("./pretrained_models/checkpoint_120000_encoder.pth", map_location='cpu')["model"]
        new_ckpt = {key.replace("module.transformer.", "") if key.startswith("module.") else key: value for key, value in fmri_encoder_ckpt.items()}
        torch_init_model(self.fmri_encoder, new_ckpt)

        lr = args.learning_rate
        autoencoder_params = []
        autoencoder_params += list(self.unet.parameters())
        autoencoder_params += list([p for n, p in self.fmri_encoder.named_parameters()])
        autoencoder_params += list([self.i_logit_scale])
        autoencoder_params += list([self.t_logit_scale])
        # 获取模型中所有带有 LoRA 权重的参数
        # lora_params = [param for name, param in self.unet.named_parameters() if 'lora' in name]
        # autoencoder_params += lora_params
        self.opt = AdamW(autoencoder_params, lr=lr, betas=(0.9, 0.95))
        self.sche = get_linear_schedule_with_warmup(
            self.opt, 
            num_warmup_steps=300,
            num_training_steps=100000
        )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.global_step = 0
        self.global_rank = self.rank
        self.device = f"cuda:{self.rank}"
        self.logdir = logdir
        self.on_fit_start()

        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k"
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
    
    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def get_clip_feature(self, image, text):
        dtype = torch.bfloat16
        image_features = self.clip_model.encode_image(image.cuda().to(dtype))
        text_features = self.clip_model.encode_text(self.tokenizer(text).cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
    
    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        dtype = next(self.fmri_encoder.parameters()).dtype
        cond_fmri = batch['fmri']      # (B, C, H, W)
        cond_fmri = cond_fmri.to(self.device).to(dtype)

        target_imgs = batch['target_imgs']  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(target_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
        target_imgs = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)    # (B, C, 3H, 2W)
        target_imgs = target_imgs.to(self.device).to(dtype)

        return cond_fmri, target_imgs

    def encode_embed_fmri_condition_fmri(self, fmri):
        dtype = next(self.fmri_encoder.parameters()).dtype
        B = fmri.shape[0]
        latents, fmri_feature = self.fmri_encoder.forward_encoder_wo_pred_clip(fmri.to(dtype).view(-1, 1, 256, 256)) # [B*6, 257, 1024]
        embed_fmri, latents = latents[:,:1,:], latents[:,1:,:]
        latents = latents.view(B, 6, 16, 16, 4, 4, 64).permute(0,1,2,4,3,5,6)
        latents = latents.reshape(B, 6, 16 * 4, 16 * 4, 64)
        latents = latents.permute(0, 2, 3, 1, 4).reshape(B, 64, 64, 64 * 6)
        latents = self.fmri_encoder.aggregation(latents).permute(0, 3, 1, 2)

        embed_fmri = embed_fmri.view(B, 6, 1, 1024)
        global_embeds = torch.mean(embed_fmri,dim=1)

        fmri_feature = fmri_feature.view(B, 6, 1, 1024)
        fmri_feature = torch.mean(fmri_feature,dim=1)

        encoder_hidden_states = self.pipeline._encode_prompt("", self.device, 1, False)[0]
        ramp = global_embeds.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        return fmri_feature, encoder_hidden_states, latents # should be [B, 4, 64, 64]
    
    
    def forward_encoder_wo_pred_clip(self, fmri):
        dtype = next(self.fmri_encoder.parameters()).dtype
        B = fmri.shape[0]
        fmri_feature = self.fmri_encoder.forward_encoder_only_clip(fmri.to(dtype).view(-1, 1, 256, 256)) # [B*6, 257, 1024]
        fmri_feature = fmri_feature.view(B, 6, 1, 1024)
        fmri_feature = torch.mean(fmri_feature,dim=1)
        return fmri_feature
    
    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    
    def forward_unet(self, latents, t, prompt_embeds, cond_latents):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_loss(self, batch):
        # get input
        self.pipeline.unet.training=True
        cond_fmri, target_imgs = self.prepare_batch_data(batch)

        # sample random timestep
        B = cond_fmri.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            fmri_feature = self.forward_encoder_wo_pred_clip(cond_fmri)
            image_features, text_features = self.get_clip_feature(batch["img"],batch["text"])
            i_clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.i_logit_scale.exp())
            t_clip_loss = self.cal_clip_loss(text_features, fmri_feature, self.t_logit_scale.exp())
            clip_loss = i_clip_loss + t_clip_loss

            prompt_embeds = self.pipeline._encode_prompt([""]*B, self.device, 1, False)
            _, _, fmri_cond_latents = self.encode_embed_fmri_condition_fmri(torch.zeros_like(cond_fmri))
        else:
            fmri_feature, prompt_embeds, fmri_cond_latents = self.encode_embed_fmri_condition_fmri(cond_fmri) # [2, 6, 64, 64]
            image_features, text_features = self.get_clip_feature(batch["img"],batch["text"])
            i_clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.i_logit_scale.exp())
            t_clip_loss = self.cal_clip_loss(text_features, fmri_feature, self.t_logit_scale.exp())
            clip_loss = i_clip_loss + t_clip_loss

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        
        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, fmri_cond_latents)
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)
        self.global_step+=1

        return loss, clip_loss
        
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
    
    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def validation_step(self, batch):
        self.pipeline.unet.training=False
        # get input
        cond_fmri, target_imgs = self.prepare_batch_data(batch)
        latent = self.pipeline(cond_fmri, num_inference_steps=75, output_type='latent').images
        images = unscale_image(self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
        images = (images * 0.5 + 0.5).clamp(0, 1)
        images = torch.cat([target_imgs, images], dim=0)        

        grid = make_grid(images, nrow=8, normalize=True, value_range=(0, 1))
        save_image(grid, os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png'))

