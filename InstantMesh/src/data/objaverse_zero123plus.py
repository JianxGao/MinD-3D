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

from src.utils.train_util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=8, num_workers=4, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir

        with open(os.path.join(root_dir, meta_fname)) as f:
            lvis_dict = json.load(f)
        paths = []
        for k in lvis_dict.keys():
            paths.extend(lvis_dict[k])
        self.paths = paths
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[-16:] # used last 16 as validation
        else:
            self.paths = self.paths[:-16]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.image_dir, self.paths[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    img_list.append(img)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
        }
        return data
    
import pandas as pd
import random

class fMRI_Objaverse(Dataset):
    def __init__(self,
        fmri_dir='objaverse/',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.fmri_dir = "/ssd/gaojianxiong/dataset/fmri_objaverse/npy_data"
        self.image_dir = "/ssd/gaojianxiong/dataset/fmri_objaverse/images"
        if not validation:
            self.file_list = np.genfromtxt("/ssd/gaojianxiong/dataset/fmri_objaverse/train_list.txt", dtype=np.str_, encoding='utf-8')
            self.mode="train"
        else:
            self.file_list = np.genfromtxt("/ssd/gaojianxiong/dataset/fmri_objaverse/test_list.txt", dtype=np.str_, encoding='utf-8')
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
        fmri_dir='objaverse/',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.fmri_dir = "/ssd/gaojianxiong/dataset/fmri_shape/npy_data"
        self.image_dir = "/ssd/gaojianxiong/dataset/fmri_shape/images"
        if not validation:
            self.file_list = np.genfromtxt("/ssd/gaojianxiong/dataset/fmri_shape/core_train_list.txt", dtype=np.str_, encoding='utf-8')
            self.mode="train"
        else:
            self.file_list = np.genfromtxt("/ssd/gaojianxiong/dataset/fmri_shape/core_test_list.txt", dtype=np.str_, encoding='utf-8')
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


class Combined_fMRI_Dataset(Dataset):
    def __init__(self,
        dir_path = "/ssd/gaojianxiong/dataset",
        fmri_objaverse_dir='fmri_objaverse/npy_data',
        image_objaverse_dir='fmri_objaverse/images',
        fmri_shape_dir='fmri_shape/npy_data',
        image_shape_dir='fmri_shape/images',
        validation=False,
    ):
        self.dir_path = dir_path
        self.fmri_objaverse_dir = os.path.join(dir_path, fmri_objaverse_dir)
        self.image_objaverse_dir = os.path.join(dir_path, image_objaverse_dir)
        self.fmri_shape_dir = os.path.join(dir_path, fmri_shape_dir)
        self.image_shape_dir = os.path.join(dir_path, image_shape_dir)

        self.objaverse_file_list = pd.read_csv(f"{dir_path}/fmri_objaverse/train_list.txt" if not validation else f"{dir_path}/fmri_objaverse/test_list.txt").values
        self.shape_file_list = np.genfromtxt(f"{dir_path}/fmri_shape/core_train_list.txt" if not validation else f"{dir_path}/fmri_shape/core_test_list.txt", dtype=np.str_, encoding='utf-8')
        
        self.file_list = [("objaverse", item) for item in self.objaverse_file_list] + [("shape", item) for item in self.shape_file_list]
        self.mode = "train" if not validation else "test"
        
        print('============= length of combined dataset %d =============' % len(self.file_list))

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
        dataset_type, file_entry = self.file_list[index]
        if dataset_type == "objaverse":
            class_name, obj_name = file_entry
            uid = obj_name.split("/")[-1].split(".")[0]
            npy_path = os.path.join(self.fmri_objaverse_dir, "{}.npy".format(uid))
            fmri_frames = np.load(npy_path)
            if self.mode == "train":
                original_list = [0, 1, 2, 3, 4, 5, 6, 7]
                sampled_list = random.sample(original_list, 6)
            else:
                sampled_list = [1, 2, 3, 4, 5, 6]
            sampled_list.sort()
            fmri_frames = fmri_frames[sampled_list]
            fmri_frames = torch.from_numpy(fmri_frames)

            img_list = []
            for i in range(6):
                image_path = os.path.join(self.image_objaverse_dir, uid + ".glb", f"{i}.png")
                bkg_color = [1., 1., 1.]
                img, alpha = self.load_im(image_path, bkg_color)
                img_list.append(img)

        elif dataset_type == "shape":
            class_name, obj_name = file_entry.split("/")
            npy_path = os.path.join(self.fmri_shape_dir, class_name, "{}.npy".format(obj_name))
            fmri_frames = np.load(npy_path)
            if self.mode == "train":
                original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                sampled_list = random.sample(original_list, 6)
            else:
                sampled_list = [2, 3, 4, 5, 6, 7]
            sampled_list.sort()
            fmri_frames = fmri_frames[sampled_list]
            fmri_frames = torch.from_numpy(fmri_frames)

            img_list = []
            for i in range(6):
                image_path = os.path.join(self.image_shape_dir, class_name, obj_name, f"{i}.png")
                bkg_color = [1., 1., 1.]
                img, alpha = self.load_im(image_path, bkg_color)
                img_list.append(img)

        imgs = torch.stack(img_list, dim=0).float()
        data = {
            'target_imgs': imgs,      # (6, 3, H, W)
            'fmri': fmri_frames,      # (6, H, W)
        }
        return data
