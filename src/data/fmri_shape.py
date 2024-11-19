import os
import cv2
import glob
import torch
import random
import json
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from src.data.augmentation import get_roi_mask, add_noise_to_roi


def _convert_image_to_rgb(image):
    return image.convert("RGB")
 
def transparent_to_white(img_path):
    img = Image.open(img_path)
    new_img = Image.new("RGB", img.size, "white")
    new_img.paste(img, (0, 0), img)
    return new_img


class fmri_shape_object(torch.utils.data.Dataset):
    def __init__(self, sub_id="0002", mode="train"):
        super(fmri_shape_object, self).__init__()
        self.file_list = np.genfromtxt("./dataset/{}_list.txt".format(mode), dtype=np.str_, encoding='utf-8')
        self.img_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.sub_id = sub_id
        self.mode = mode
        self.mask = np.load("./cmask.npy")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        class_name, obj_name = self.file_list[index].split("/")
        # Load fMRI signal and take augmentaion on fMRI frame-level
        npy_path = os.path.join("./dataset/sub-{}".format(self.sub_id), class_name, obj_name)
        fmri_frames = np.load(npy_path)
        if self.mode=="train":
            original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            sampled_list = random.sample(original_list, 6)
            sampled_list.sort()
            sampled_list = np.array(sampled_list)
        else:
            sampled_list = np.array([2, 3, 4, 5, 6, 7])
        fmri_frames = fmri_frames[sampled_list]

        # Data augmentation
        if self.mode=="train":
            for i in range(6):
                rnd = np.random.random()
                if rnd < 0.8:
                    augmentation_mask = get_roi_mask(self.mask)
                    fmri_frames[i] = add_noise_to_roi(fmri_frames[i], augmentation_mask)
        fmri_frames = torch.from_numpy(fmri_frames)
        
        # For the training of stage2
        # Load GT quantize_index
        uid = obj_name.split("/")[-1].replace(".npy","")
        quantize_index = np.load(os.path.join("./dataset/quantized", class_name, uid+".npz"))["index"]

        # Load image
        img_idx_list = [i for i in range(0,192)]
        img_sampled_list = random.sample(img_idx_list, 8)
        img_tensor_list = []
        for i in range(len(img_sampled_list)):
            img_path = os.path.join("./dataset/frame224", class_name, uid, "{}.png".format(str(img_sampled_list[i])))
            img_tensor_list.append(self.img_transform(transparent_to_white(img_path)))
        img_tensor = torch.stack(img_tensor_list)


        meta = {
            "fmri": fmri_frames,
            "class_name": class_name,
            "uid": uid,
            "quantize_index": quantize_index,
            "img_path": img_path,
            "img": img_tensor,
        }
        return meta

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res
    
    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn
            )
            for item in sample_loader:
                yield item



