import h5py
import numpy as np
import os
from tqdm import tqdm
import cv2
import multiprocessing
import pickle


# Mask
image_size=(256, 256)
mask = np.load("./vc_roi.npz")['images'][0] # [H, W]
H, W = mask.shape
vc_mask = mask == 1 # [H, W]
fg_mask = (mask == 1) | (mask == -1) # [H, W]

x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xx, yy = np.meshgrid(x, y)
grid = np.stack([xx, yy], axis=0) # [2, H, W]

gird_ = grid * vc_mask[np.newaxis]
x1 = min(int(gird_[0].max()) + 1, W)
y1 = min(int(gird_[1].max()) + 10, H)
gird_[gird_ == 0] = 1e6
x0 = max(int(gird_[0].min() - 1), 0)
y0 = max(int(gird_[1].min() - 10), 0)

vc_mask = vc_mask
fg_mask = fg_mask
coord = [x0, x1, y0, y1]

crop_msk = vc_mask[coord[2]:coord[3] + 1, coord[0]:coord[1] + 1]
cmask = cv2.resize(crop_msk * 1., (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

# import pdb;pdb.set_trace()
def transform(image):
    image = np.array(image)
    image = image[coord[2]:coord[3] + 1, coord[0]:coord[1] + 1]
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image[cmask == 0] = 0
    return image


hdf5_dir = 'path/to/fMRI-Objaverse/obj_sub15/DNV/hdf5'
out_dir = 'path/to/fMRI-Objaverse/obj_sub15/npy_256_norm'

hdf5_files = sorted([f for f in os.listdir(hdf5_dir)])
index_list = []

def get_mean_std(index):
    h5_path = "/path/to/fmri_objaverse/sub15/hdf5/sub-0015_task-shape_run-{}_space-fsLR_den-91k_bold.dtseries.h5".format(index)
    samples = []
    images = h5py.File(os.path.join(hdf5_dir, h5_path), 'r')["images"]

    mean=np.mean(images, axis=0)
    std=np.std(images, axis=0, ddof=1)

    obj = {
        "mean":mean,
        "std":std,
        "number":len(images),
    }
    os.makedirs('/path/to/fmri_objaverse/sub15/mean_std_pkl',exist_ok=True)
    with open('/path/to/fmri_objaverse/sub15/mean_std_pkl/{}.pkl'.format(index), 'wb') as f:
        pickle.dump(obj, f)


def h5_to_npy_normlization(index):
    h5_path = "/path/to/fmri_objaverse/sub15/hdf5/sub-0015_task-shape_run-{}_space-fsLR_den-91k_bold.dtseries.h5".format(index)
    save_path = os.path.join("/path/to/fmri_objaverse/sub15/npy_256_norm","run-{}.npy".format(index))
    if not os.path.exists(save_path):
        os.makedirs("/path/to/fmri_objaverse/sub15/npy_256_norm",exist_ok=True)
        print(save_path)
        samples = []
        mean = np.load("/path/to/fmri_objaverse/sub15/mean.npy")
        std = np.load("/path/to/fmri_objaverse/sub15/std.npy")
        images = h5py.File(os.path.join(hdf5_dir, h5_path), 'r')["images"]
        images=(images-mean)/std
        images[np.isnan(images)]=0 
        for i in range(len(images)):
            gray_image = transform(images[i])
            samples.append(gray_image)
        samples = np.stack(samples)
        print(samples.shape)
        np.save(save_path, samples)


# step 1 get the mean and std
for i in range(53):
    index_list.append((i+1,))
with multiprocessing.Pool() as pool:
    results = pool.starmap(get_mean_std, index_list)

total_number = 0
with open("/path/to/fmri_objaverse/sub15/mean_std_pkl/1.pkl","rb") as f:
    example_data = pickle.load(f)
mean_array = np.zeros_like(example_data["mean"])
std_array = np.zeros_like(example_data["std"])
for file in os.listdir("/path/to/fmri_objaverse/sub15/mean_std_pkl"):
    with open("/path/to/fmri_objaverse/sub15/mean_std_pkl/{}".format(file),"rb") as f:
        data = pickle.load(f)
        mean_array+=example_data["mean"]*example_data["number"]
        std_array+=example_data["std"]*example_data["number"]
        total_number+=example_data["number"]


np.save("/path/to/fmri_objaverse/sub15/mean.npy",mean_array/total_number)
np.save("/path/to/fmri_objaverse/sub15/std.npy",std_array/total_number)


# step2 h52npy
for i in range(53):
    index_list.append((i+1,))
with multiprocessing.Pool() as pool:
    results = pool.starmap(h5_to_npy_normlization, index_list)

