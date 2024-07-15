## [ECCV 2024] MinD-3D: Reconstruct High-quality 3D objects in Human Brain

[Jianxiong Gao](https://jianxgao.github.io/), [Yuqian Fu](http://yuqianfu.com/), Yun Wang, [Xuelin Qian](https://naiq.github.io/), [Jianfeng Feng](https://www.dcs.warwick.ac.uk/~feng/) and [Yanwei Fu†](http://yanweifu.github.io/)

[![ArXiv](https://img.shields.io/badge/ArXiv-2312.07485-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2312.07485)
[![HomePage](https://img.shields.io/badge/HomePage-Visit-blue.svg?logo=homeadvisor&logoColor=f5f5f5)](https://jianxgao.github.io/MinD-3D/)
[![Dataset](https://img.shields.io/badge/Dataset-fMRI_Shape-F07B3F.svg)](https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape)




# Introduction
<img src='./imgs/teaser.png' width="100%">


# Dataset
You can download fMRI-Shape by this link: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape.

# Environment Setup

```bash
git clone https://github.com/JianxGao/MinD-3D.git
cd MinD-3D
bash env_install.sh
```

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
