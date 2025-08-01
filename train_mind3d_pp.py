import os
import math
import torch
import numpy as np
import argparse
import shutil
import time
from src.utils import load_config
from src.mvdiffusion import MVDiffusion
from src.utils import torch_init_model, set_random_seed, CheckpointIO
from src.data.texture_dataset import fMRI_Objaverse
# from src.data.texture_dataset import fMRI_Shape
from einops import rearrange
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from contextlib import nullcontext
from omegaconf import OmegaConf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D mesh based on image input')
    parser.add_argument('--config', type=str, help='Path to config file.', default="./configs/mind3d.yaml")
    # Training
    parser.add_argument('--sub_id', type=str, default="0001")
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default="stage1_model")
    parser.add_argument('--check_point_path', type=str, default="mind3d_30k.pt")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    
    # # For slurm
    # import subprocess
    # from time import sleep
    # while not os.environ.get("MASTER_ADDR", ""):
    #     try:
    #         os.environ["MASTER_ADDR"] = subprocess.check_output(
    #             "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" %
    #             os.environ['SLURM_NODELIST'],
    #             shell=True,
    #         ).decode().strip()
    #     except:
    #         pass
    #     sleep(1)
    # os.environ["MASTER_PORT"] = str(int(12579)+1)
    # os.environ["RANK"] = os.environ["SLURM_PROCID"]
    # os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    # os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    # os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]

    args = parser.parse_args()
    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
    cfg = OmegaConf.load(args.config)

    # initialize random seed
    seed=42
    set_random_seed(seed)

    # Output directory and copy the config file
    out_dir = os.path.join("./output", args.out_dir)
    if args.ddp and rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
    elif not args.ddp:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
    
    # Setup device
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    # load dataset
    train_dataset = fMRI_Objaverse(validation=False)
    train_sampler = None
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        num_workers=12,
        drop_last=True,
        sampler=train_sampler,
    )

    test_dataset = fMRI_Objaverse(validation=True)
    sample_iterator = test_dataset.create_iterator(8)

    # load model
    model_full = MVDiffusion(
        cfg, 
        cfg.model.params.stable_diffusion_config, 
        fmri_encoder_config=cfg.model.params.fmri_encoder_config,
        logdir=out_dir
    ).cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.ddp:
        model_full_ddp = torch.nn.parallel.DistributedDataParallel(model_full, device_ids=[local_rank], output_device=local_rank)
        model_full = model_full_ddp.module

    # define checkpoint IO
    # state_dict = torch.load("/mnt/petrelfs/gaojianxiong/MinD-3D/output/sd_white_24l_encoder_shape_sub7/model_30000.pt", map_location='cpu')["model"]
    # torch_init_model(model_full, state_dict)
    checkpoint_io = CheckpointIO(out_dir, model=model_full)
    
    if args.ddp:
        if rank==0:
            logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    else:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    print_every = cfg.training.print_every
    backup_every = cfg.training.backup_every
    validate_every = cfg.training.validate_every
    visualize_every = cfg.training.visualize_every

    epoch_it=0
    it=0
    vis_count = 0
    t0 = time.time()
    while True:
        epoch_it += 1
        if args.ddp:
            train_loader.sampler.set_epoch(epoch_it)
        for train_item in train_loader:
            model_full.train()
            it+=1
            mcontext = model_full_ddp.no_sync if rank != -1 and it % args.accumulation_steps != 0 else nullcontext

            with mcontext():
                diff_loss, clip_loss = model_full.get_loss(train_item)
                loss = diff_loss + clip_loss
                loss = loss / args.accumulation_steps
                loss.backward()

            if it % args.accumulation_steps == 0:
                model_full.opt.step() 
                model_full.opt.zero_grad()
                model_full.sche.step()

            if rank == 0:
                logger.add_scalar('train/diff_loss', diff_loss, it)
                logger.add_scalar('train/clip_loss', clip_loss, it)
                logger.add_scalar('lr', model_full.sche.get_lr()[0], it)

                if print_every > 0 and (it % print_every) == 0:
                    t = time.time() - t0
                    print('[Epoch %02d] it=%03d, Train: diff_loss=%.4f, clip_loss=%.4f, lr=%.8f, time: %.0fm %0.2fs'
                    % (epoch_it, it, diff_loss, clip_loss, model_full.sche.get_lr()[0], t // 60, t % 60))

                # validate model
                if visualize_every > 0 and (it % visualize_every) == 0:
                    model_full.eval()
                    items = next(sample_iterator)
                    model_full.validation_step(items)
                    model_full.train()
                # save model
                if (backup_every > 0 and (it % backup_every) == 0):
                    checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it)
            dist.barrier()