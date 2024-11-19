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


def dfs(mask, start, visited, path, directions, target_length=20):
    if len(path) == target_length:
        return True
    
    random.shuffle(directions)  # Optimize by shuffling directions only once per recursion level
    for dx, dy in directions:
        x, y = start[0] + dx, start[1] + dy
        if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] == 1 and not visited[x, y]:  # Combined checks
            visited[x, y] = True
            path.append((x, y))
            if dfs(mask, (x, y), visited, path, directions, target_length):
                return True
            path.pop()  # Efficient backtracking

    return False

def find_connected_pixels(mask, target_length=20):
    visited = np.zeros_like(mask, dtype=bool)
    roi_indices = np.transpose(np.where(mask == 1))
    if roi_indices.size == 0:  # More efficient check
        return []
    
    start_index = random.choice(roi_indices)
    start = tuple(start_index)
    visited[start] = True
    path = [start]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Moved outside dfs for efficiency

    if dfs(mask, start, visited, path, directions, target_length):
        return path
    return []

def create_visualization_array(shape, path_list):
    visualization_array = np.zeros(shape)
    for path in path_list:
        for x, y in path:
            visualization_array[x, y] = 1
    return visualization_array

def get_roi_mask(mask):
    path_list = []
    path_number = np.random.randint(0, 6)  # Ensure at least one path is generated
    for _ in range(path_number):
        target_length = np.random.randint(100, 300)
        path = find_connected_pixels(mask, target_length=target_length)
        if path:  # Add only if the path is not empty
            path_list.append(path)
    visualization_array = create_visualization_array(mask.shape, path_list)
    return visualization_array

def add_noise_to_roi(data, mask):
    """
    Add random noise with a mean of `noise_mean` to `noise_percentage` of the pixels
    in the ROI defined by `mask` in the `data` array.

    Parameters:
    - data: 2D numpy array of shape (256, 256) with the image data.
    - mask: 2D numpy array of shape (256, 256) where the ROI is marked with 1s.
    - noise_mean: The mean of the noise to add.
    - noise_percentage: The percentage of the ROI pixels to add noise to.

    Returns:
    - noisy_data: The data array with added noise in the ROI.
    """
    noise_mean = np.mean(data)
    noise_std = np.std(data)

    # Create a copy of the data to not alter the original
    noisy_data = np.copy(data)

    # adjust the dimension of mask
    # new_mask = np.repeat(mask[np.newaxis,...],len(data),axis=0)
    new_mask = mask
    
    # Find the indices of the ROI pixels
    selected_indices = np.where(new_mask == 1)

    # Generate random noise with the specified mean
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(selected_indices[0]))
    
    # Add the noise to the selected pixels
    noisy_data[selected_indices] += noise
    
    return noisy_data