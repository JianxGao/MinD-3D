import os
import torch
import numpy as np
import os
from tqdm import tqdm
import trimesh
from scipy.linalg import sqrtm
import argparse
import glob
import pdb
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import numpy as np
import os

import ot


def compute_emd(point_cloud1, point_cloud2):
    """
    Compute the Earth Mover's Distance (EMD) between two point clouds.

    Parameters:
    - point_cloud1: numpy array of shape (n, d), representing the first point cloud.
    - point_cloud2: numpy array of shape (m, d), representing the second point cloud.

    Returns:
    - EMD value as a float.
    """
    # Number of points in each point cloud
    n = point_cloud1.shape[0]
    m = point_cloud2.shape[0]

    # Uniform distribution on the point clouds
    a, b = np.ones((n,)) / n, np.ones((m,)) / m

    # Cost matrix: Euclidean distance between points
    M = ot.dist(point_cloud1, point_cloud2, metric='euclidean')

    # Compute EMD
    emd_value = ot.emd2(a, b, M)

    return emd_value

def scale_to_unit_sphere(points):
    """
    scale point clouds into a unit sphere
    :param points: (n, 3) numpy array
    :return:
    """
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    return points


def adjust_mesh(mesh):
    mesh.apply_translation(-mesh.center_mass)
    vertices = np.asarray(mesh.vertices)
    mean_centered = vertices - vertices.mean(axis=0)
    cov_matrix = np.cov(mean_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    rotation_matrix = eigvecs[:, ::-1]
    mesh.apply_transform(np.vstack([np.hstack([rotation_matrix, [[0], [0], [0]]]), [0, 0, 0, 1]]))
    
    return mesh


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
    gen_pcs = []
    name_list = gen_path_list
    for name in tqdm(name_list):
        sample_pts = trimesh.load(name)
        sample_pts = adjust_mesh(sample_pts)
        sample_pts, _ = trimesh.sample.sample_surface(sample_pts, 2048)
        sample_pts = pc_norm(sample_pts)
        gen_pcs.append(sample_pts.astype(np.float32))

    gen_pcs = np.stack(gen_pcs, axis=0)
    return gen_pcs


def collect_gt_pc(gt_path_list):
    pc_paths = gt_path_list
    gt_pc_list = []
    for shape_path in tqdm(pc_paths):
        target_pc = np.load(shape_path)
        gt_pc_list.append(target_pc.astype(np.float32))       
    gt_pcs = np.stack(gt_pc_list,axis=0)

    return gt_pcs

obj_id_list = os.listdir("gt_path")
gt_path_list = [os.path.join("gt_path", id) for id in obj_id_list]
gen_path_list = [os.path.join("gen_path", id) for id in obj_id_list]


generate_pcs = collect_gen_pc(gen_path_list)
gt_pcs = collect_gt_pc(gt_path_list)
total_emd = 0
for i in range(len(generate_pcs)):
    emd0 = compute_emd(gt_pcs[i], generate_pcs[i])
    total_emd += emd0
print(total_emd, total_emd/len(gt_pcs))
