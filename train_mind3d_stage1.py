import os
import math
import torch
import numpy as np
import argparse
import shutil
import time
from src.utils import load_config
from src.stage1_model import MinD3D
from src.utils import torch_init_model, set_random_seed, CheckpointIO
from src.data.fmri_shape import fmri_shape_object
from einops import rearrange
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from contextlib import nullcontext


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D mesh based on image input')
    parser.add_argument('--config', type=str, help='Path to config file.', default="./configs/mind3d.yaml")
    # 3D decoder
    parser.add_argument('--transformer_embed_dim', type=int, default=3072)
    parser.add_argument('--transformer_n_head', type=int, default=24)
    parser.add_argument('--transformer_layer', type=int, default=32)
    parser.add_argument('--topk', type=int, default=250)

    # Training
    parser.add_argument('--sub_id', type=str, default="0001")
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default="stage1_model")
    parser.add_argument('--check_point_path', type=str, default="mind3d_30k.pt")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

    args = parser.parse_args()
    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        rank = dist.get_rank()

    cfg = load_config(args.config, 'configs/default.yaml')

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
    train_dataset = fmri_shape_object(sub_id=args.sub_id, mode="train")
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

    test_dataset = fmri_shape_object(sub_id=args.sub_id, mode="test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        num_workers=4,
        drop_last=False,
    )

    # load model
    model_full = MinD3D(cfg, device=device).cuda().to(torch.float16)
    if args.ddp:
        model_full_ddp = torch.nn.parallel.DistributedDataParallel(model_full, device_ids=[args.local_rank], output_device=args.local_rank)
        model_full = model_full_ddp.module

    # define checkpoint IO
    # state_dict = torch.load(args.check_point_path, map_location='cpu')["model"]
    # torch_init_model(model_full, state_dict)
    checkpoint_io = CheckpointIO(out_dir, model=model_full)
    
    if args.ddp:
        if rank==0:
            logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    else:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))


    print_every = cfg['training']['print_every']
    backup_every = cfg['training']['backup_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    epoch_it=0
    it=0
    vis_count = 0
    t0 = time.time()
    while True:
        epoch_it += 1
        model_full.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch_it)
        for train_item in train_loader:
            it+=1
            mcontext = model_full_ddp.no_sync if rank != -1 and it % args.accumulation_steps != 0 else nullcontext

            with mcontext():
                diff_loss, clip_loss, logit_scale = model_full.get_loss(train_item)
                # model_full.backward(clip_loss)
                loss = clip_loss + diff_loss
                loss = loss / args.accumulation_steps
                loss.backward()

            if it % args.accumulation_steps == 0:
                model_full.opt.step() 
                model_full.opt.zero_grad()
                model_full.sche.step()

            with torch.no_grad():
                model_full.logit_scale.clamp_(0, math.log(100))

            if rank == 0:
                logger.add_scalar('train/diff_loss', diff_loss, it)
                logger.add_scalar('train/clip_loss', clip_loss, it)
                logger.add_scalar('train/logit_scale', logit_scale, it)
                logger.add_scalar('lr', model_full.sche.get_lr()[0], it)

                if print_every > 0 and (it % print_every) == 0:
                    t = time.time() - t0
                    print('[Epoch %02d] it=%03d, Train: diff_loss=%.4f, clip_loss=%.4f, logit_scale=%.4f, lr=%.8f, time: %.0fm %0.2fs'
                    % (epoch_it, it, diff_loss, clip_loss, logit_scale, model_full.sche.get_lr()[0], t // 60, t % 60))

                # validate model
                if validate_every > 0 and (it % validate_every) == 0:
                    model_full.eval()
                    for test_item in test_loader:
                        test_diff_loss, test_clip_loss = model_full.get_test_metric(test_item)
                    print('[Epoch %02d] it=%03d, Test: diff_loss=%.4f, clip_loss=%.4f' % (
                        epoch_it, it, test_diff_loss, test_clip_loss,
                    ))
                    logger.add_scalar('test/diff_loss', test_diff_loss, it)
                    logger.add_scalar('test/clip_loss', test_clip_loss, it)

                # save model
                if (backup_every > 0 and (it % backup_every) == 0):
                    checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it)
            dist.barrier()