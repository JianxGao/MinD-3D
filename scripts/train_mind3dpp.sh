CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25644 \
 python train_sd.py --ddp \
 --config ./configs/mind3d_pp.yaml \
 --out_dir mind3dpp_fmri_shape_subject1_rank_64 --batchsize 8
