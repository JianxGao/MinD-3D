import torch
import numpy as np
from pointnet import PointNetCls
import os
from tqdm import tqdm
import trimesh
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.linalg import sqrtm
import argparse
import glob
import pdb


def collect_gt_pc(gt_path_list):
    pc_paths = sorted(gt_path_list)
    gt_pc_list = []

    for shape_path in tqdm(pc_paths):
        # target_ply = trimesh.load(shape_path)
        pc_dict = np.load(shape_path)

        target_pc = pc_dict['points']
        # print(target_ply.shape)
        # target_pc = target_ply.vertices      # gt pc: 8192
        '''Normalize'''
        target_pc = pc_norm(target_pc)
        # assert target_pc.shape[0] == 8192
        # subsample gt point: 2048 
        target_pc = sample_pc(target_pc, 2048)
        gt_pc_list.append(target_pc.astype(np.float32))       
    gt_pcs = np.stack(gt_pc_list,axis=0)

    return gt_pcs

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (2 * m)
    return pc

def sample_pc(pc, sample_point):
    index = np.random.choice(pc.shape[0],sample_point,replace=True)
    pc = pc[index]
    return pc

def collect_gen_pc(gen_path_list):
    gen_path_list = sorted(gen_path_list)
    # Collect Generate pc
    gen_pcs = []
    name_list = gen_path_list
    for name in tqdm(name_list):
        # name_path = os.path.join(gen_path, name)
        name_path = name[:-7]
        # generate_list = os.listdir(name_path)
        generate_list = glob.glob(name_path+"*.obj")
        # if (len(generate_list)) != 5:
        #     print('\n\n\n',generate_list)
        for i in range(len(generate_list)):
            shape_path = os.path.join(name_path, generate_list[i])
            sample_pts = trimesh.load(shape_path)
            sample_pts,_ = trimesh.sample.sample_surface(sample_pts,2048)
            sample_pts = pc_norm(sample_pts)
            gen_pcs.append(sample_pts.astype(np.float32))

    gen_pcs = np.stack(gen_pcs, axis=0)
    return gen_pcs


def get_obj_files(directory):
    obj_files = []
    for file in os.listdir(directory):
        if file.endswith(".obj"):
            obj_files.append(file)
    return obj_files


def get_cate_dict(subset_path):
    category_list = os.listdir(subset_path)
    cate_dict = {}
    category_list = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649','03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
    for cate in category_list:
        data_list = os.listdir(os.path.join(subset_path, cate))
        cate_dict[cate] = sorted(data_list)[:250][::5]
    return cate_dict


def get_act_tensor(pc_tensor, bs=200, model=None, device=None):
    model.eval()
    act_list = []
    n_batch = pc_tensor.size(0) // bs + 1

    for i in tqdm(range(n_batch)):
        start = i * bs
        end = min((i+1) * bs, pc_tensor.shape[0])
        pc = pc_tensor[start:end]
        pc_num = pc.shape[0]
        _, _, act = model(pc)
        act = act.cpu().data.numpy().reshape(pc_num, -1)
        act_list.append(act)
    act_list = np.concatenate(act_list,axis=0)
    return act_list


def cal_final_fpd(mu1, sigma1, mu2, sigma2, eps=1e-4):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_activation_stat(pc_tensor, bs=200, model=None, device=None):
    act = get_act_tensor(pc_tensor, bs, model, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def cal_all_pvd(gen_path, gt_path, batch_size=50, device=None, model=None):
    gen_path_list = os.listdir(gen_path)
    gen_path_list = [os.path.join(gen_path,file) for file in gen_path_list]
    gt_path_list = os.listdir(gt_path)
    gt_path_list = [os.path.join(gt_path,file) for file in gt_path_list]

    generate_pcs = collect_gen_pc(gen_path_list)
    gt_pcs = collect_gt_pc(gt_path_list)
    print("Finished Loading")

    generate_pcs = torch.from_numpy(generate_pcs).to(device)
    gt_pcs = torch.from_numpy(gt_pcs).to(device)

    gen_tensor = generate_pcs.permute(0, 2, 1)
    gt_tensor = gt_pcs.permute(0, 2, 1)

    m_gen, s_gen = cal_activation_stat(gen_tensor, bs=batch_size, model=model, device=device)
    m_gt, s_gt = cal_activation_stat(gt_tensor, bs=batch_size, model=model, device=device)
    fpd_value = cal_final_fpd(m_gen, s_gen, m_gt, s_gt)

    return fpd_value


parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.p')
parser.add_argument('--src', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda")
bs = 60


subset_path= '~/MinD-3D/outputs/fmri_shapeshape_55k_r8/meshes'
gt_path = '~/MinD-3D/dataset/fMRI-Shape/test_pc'

print(" PointNet Init")
pointnet_model = PointNetCls(k=13)
pretrain = torch.load("~/MinD-3D/checkpoint_300_0.9205357142857142.pth")
pointnet_model.load_state_dict(pretrain["model_state_dict"])
pointnet_model = pointnet_model.to(device)
pointnet_model.eval()
print("Loading pretrain Pointnet success !")


gen_path = subset_path
pvd = cal_all_pvd(gen_path, gt_path, bs, device=device, model=pointnet_model)
print('Ours PVD:', pvd)



