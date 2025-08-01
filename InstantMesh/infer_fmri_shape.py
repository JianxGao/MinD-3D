import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video
from src.encoder.autoencoder2 import fMRI_Encoder

def torch_init_model(model, state_dict, dist=False):
    # state_dict = torch.load(init_checkpoint, map_location='cpu')[key]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(model, prefix='')
    print("  --> missing keys:{}".format(missing_keys))
    print('  --> unexpected keys:{}'.format(unexpected_keys))
    print('  --> error msgs:{}'.format(error_msgs))


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--fmri_dir', type=str, help='')
parser.add_argument('--gt_image_dir', type=str, help='')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--unet_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--save_name', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True
config_name = config_name + args.save_name
device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path = "~/.cache/huggingface/hub/models--sudo-ai--zero123plus-v1.2/snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd", 
    custom_pipeline="src/zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)


fmri_encoder = fMRI_Encoder(config.model.params.fmri_encoder_config).to(torch.float16).cuda()
pipeline.fmri_encoder = fmri_encoder

# load custom white-background UNet
print('Loading custom white-background unet ...')
unet_ckpt_path = args.unet_path
state_dict = torch.load(unet_ckpt_path, map_location='cpu')

fmri_ckpt = {}
for n in list(state_dict["model"].keys()):
    if "fmri_encoder" in n:
        fmri_ckpt[n.replace("fmri_encoder.","")] = state_dict["model"][n]

unet_ckpt = {}
for n in list(state_dict["model"].keys()):
    if "unet.base_model.model.unet" in n:
        unet_ckpt[n.replace("unet.base_model.model.unet.","")] = state_dict["model"][n]

torch_init_model(pipeline.fmri_encoder, fmri_ckpt)
torch_init_model(pipeline.unet, unet_ckpt)


pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
model_ckpt_path = infer_config.model_path
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
input_files = np.genfromtxt(args.input_path, dtype=str, delimiter="\n")

print(f'Total number of input images: {len(input_files)}')

###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()
outputs = []
for idx, test_file in enumerate(input_files):
    name = test_file.split("/")[-1]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')
    input_image = np.load("{}/npy_data/{}.npy".format(args.fmri_dir, test_file))[2:8]
    input_image = torch.from_numpy(input_image).unsqueeze(0).to(torch.float16).cuda()
    print(input_image.size())
    # sampling
    output_image = pipeline(
        input_image, 
        num_inference_steps=args.diffusion_steps, 
    ).images[0]

    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs.append({'name': name, 'images': images})
    gt_image_path = os.path.join(args.gt_image_dir, test_file+"/0.png")
    os.system("cp {} {}/{}_gt.png".format(gt_image_path,image_path,name))
# delete pipeline to save memory
del pipeline

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")
