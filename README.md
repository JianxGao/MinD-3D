# [ECCV 2024] MinD-3D: Reconstruct High-quality 3D objects in Human Brain

[Jianxiong Gao](https://jianxgao.github.io/), [Yuqian Fu](http://yuqianfu.com/), Yun Wang, [Xuelin Qian](https://naiq.github.io/), [Jianfeng Feng](https://www.dcs.warwick.ac.uk/~feng/) and [Yanwei Fu†](http://yanweifu.github.io/)

[![ArXiv](https://img.shields.io/badge/ArXiv-2312.07485-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.07485)
[![HomePage](https://img.shields.io/badge/HomePage-Visit-blue.svg?logo=homeadvisor&logoColor=f5f5f5)](https://jianxgao.github.io/MinD-3D/)
[![Dataset](https://img.shields.io/badge/Dataset-fMRI_Shape-faa035.svg?logo=Huggingface)](https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape)
[![Dataset](https://img.shields.io/badge/Dataset-fMRI_Objaverse-3aa0a5.svg?logo=Huggingface)](https://huggingface.co/datasets/Fudan-fMRI/fMRI-Objaverse)


# Introduction
<img src='./imgs/teaser.png' width="100%">

# Project Updates
- 🔥 **News**: We have released the training code!
-  **News**: We sample a new dataset, fMRI-Objaverse: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Objaverse.
-  **Note**: fMRI-Shape: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape.

# Environment Setup

```bash
git clone https://github.com/JianxGao/MinD-3D.git
cd MinD-3D
bash env_install.sh
```

# Dataset and checkpoints
You can download fMRI-Shape by this link: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape.

You can download fMRI-Objaverse by this link: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Objaverse.

You can download the weight of subject 1 through the link: 

https://drive.google.com/file/d/1ni4g1iCvdpoi2xYtmydr_w3XA5PpNrvm/view?usp=sharing



# Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \
 train_stage1.py --sub_id 0001 --ddp \
 --config ./configs/mind3d.yaml \
 --out_dir sub01_stage1 --batchsize 8
```

```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \
 train_stage2.py --sub_id 0001 --ddp \
 --config ./configs/mind3d.yaml \
 --out_dir sub01_stage2 --batchsize 2
```

You can access the quantized features for training through the link: https://drive.google.com/file/d/1R8IpG1bligLAfHkLQ2COrfTIkay14AEm/view?usp=drive_link.



# Inference

```bash
# Sub01 Plane
python generate_fmri2shape.py --config ./configs/mind3d.yaml  --check_point_path ./mind3d_sub01.pt \ 
 --uid b5d0ae4f723bce81f119374ee5d5f944 --topk 250

# Sub01 Car
python generate_fmri2shape.py --config ./configs/mind3d.yaml  --check_point_path ./mind3d_sub01.pt \ 
 --uid aebd98c5d7e8150b709ce7955adef61b --topk 250
```

# Citation
If you find our paper useful for your research and applications, please cite using this BibTeX:

```
@misc{gao2023mind3d,
      title={MinD-3D: Reconstruct High-quality 3D objects in Human Brain}, 
      author={Jianxiong Gao and Yuqian Fu and Yun Wang and Xuelin Qian and Jianfeng Feng and Yanwei Fu},
      year={2023},
      eprint={2312.07485},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{gao2025mind3dadvancingfmribased3d,
      title={MinD-3D++: Advancing fMRI-Based 3D Reconstruction with High-Quality Textured Mesh Generation and a Comprehensive Dataset}, 
      author={Jianxiong Gao and Yanwei Fu and Yuqian Fu and Yun Wang and Xuelin Qian and Jianfeng Feng},
      year={2025},
      eprint={2409.11315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.11315}, 
}
```
