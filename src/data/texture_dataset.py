import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
import importlib
import pandas as pd
import random
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

shapenet_name_dict = {
    "02691156" : "Airplane.",
    "02828884" : "Bench.",
    "02933112" : "Cabinet.",
    "02958343" : "Car.",
    "03001627" : "Chair.",
    "03211117" : "Display (Monitor).",
    "03636649" : "Lamp.",
    "03691459" : "Loudspeaker.",
    "04090263" : "Rifle.",
    "04256520" : "Sofa.",
    "04379243" : "Table.",
    "04401088" : "Telephone.",
    "04530566" : "Watercraft (Boat.)"
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def transparent_to_white(img_path):
    img = Image.open(img_path)
    new_img = Image.new("RGB", img.size, "white")
    new_img.paste(img, (0, 0), img)
    return new_img

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


class fMRI_Objaverse(Dataset):
    def __init__(self,
        fmri_dir='objaverse/',
        sub_id = "sub-0001",
        image_dir='images',
        validation=False,
    ):
        self.fmri_dir = os.path.join(fmri_dir, sub_id)
        self.image_dir = os.path.join(fmri_dir, image_dir)
        if not validation:
            self.file_list = np.genfromtxt(os.path.join(fmri_dir, 'train_list.txt'), dtype=np.str_, encoding='utf-8')
            self.mode="train"
        else:
            self.file_list = np.genfromtxt(os.path.join(fmri_dir, 'test_list.txt'), dtype=np.str_, encoding='utf-8')
            self.mode="test"
        print('============= length of dataset %d =============' % len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        class_name, obj_name = self.file_list[index]
        uid = obj_name.split("/")[-1].split(".")[0]
        npy_path = os.path.join(self.fmri_dir, "{}.npy".format(uid))
        fmri_frames = np.load(npy_path)
        if self.mode=="train":
            original_list = [0, 1, 2, 3, 4, 5, 6, 7]
            sampled_list = random.sample(original_list, 6)
            sampled_list.sort()
            sampled_list = np.array(sampled_list)
        else:
            sampled_list = np.array([1,2,3,4,5,6])

        fmri_frames = fmri_frames[sampled_list]
        fmri_frames = torch.from_numpy(fmri_frames)
        
        img_list = []
        for i in range(6):
            image_path = os.path.join(self.image_dir, uid+".glb", f"{i}.png")
            '''background color, default: white'''
            bkg_color = [1., 1., 1.]
            img, alpha = self.load_im(image_path, bkg_color)
            img_list.append(img)

        imgs = torch.stack(img_list, dim=0).float()
        data = {
            'target_imgs': imgs,      # (6, 3, H, W)
            'fmri': fmri_frames,      # (6, H, W)
        }
        return data


class fMRI_Shape(Dataset):
    def __init__(self,
        fmri_dir='/mnt/petrelfs/gaojianxiong/cinebrain/dataset/fmri_shape',
        sub_id='sub-0001',
        image_dir='images',
        validation=False,
    ):
        self.fmri_dir = os.path.join(fmri_dir, sub_id)
        self.image_dir = os.path.join(fmri_dir, image_dir)
        if not validation:
            self.file_list = np.genfromtxt(os.path.join(fmri_dir, 'core_train_list.txt'), dtype=np.str_, encoding='utf-8')
            self.mode="train"
        else:
            self.file_list = np.genfromtxt(os.path.join(fmri_dir, 'core_test_list.txt'), dtype=np.str_, encoding='utf-8')
            self.mode="test"
        print('============= length of dataset %d =============' % len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        class_name, obj_name = self.file_list[index].split("/")
        npy_path = os.path.join(self.fmri_dir,class_name,"{}.npy".format(obj_name))
        
        fmri_frames = np.load(npy_path)
        if self.mode=="train":
            original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            sampled_list = random.sample(original_list, 6)
            sampled_list.sort()
            sampled_list = np.array(sampled_list)
        else:
            sampled_list = np.array([2, 3, 4, 5, 6, 7])

        fmri_frames = fmri_frames[sampled_list]
        fmri_frames = torch.from_numpy(fmri_frames)
        
        img_list = []
        for i in range(6):
            image_path = os.path.join(self.image_dir, class_name, obj_name, f"{i}.png")
            '''background color, default: white'''
            bkg_color = [1., 1., 1.]
            img, alpha = self.load_im(image_path, bkg_color)
            img_list.append(img)

        imgs = torch.stack(img_list, dim=0).float()
        data = {
            'target_imgs': imgs,      # (6, 3, H, W)
            'fmri': fmri_frames,      # (6, H, W)
        }
        return data

