import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import trimesh
from utils import libmcubes
from utils.libmise import MISE
from utils.libsimplify import simplify_mesh
from src.common import make_3d_grid, normalize_coord
from omegaconf import OmegaConf
# fmri encoder
from src.encoder.autoencoder import fMRI_Encoder
from src.encoder.feature_aggregation import Feature_Aggregation
# diffusion prior
from src.diffusion.fbdm import DiffusionNetwork, FBDM
# 3d decoder
from src.decoder.lad import Latent_Adapted_Decoder


class MinD3D(nn.Module):
    def __init__(self, config, device=None, dataset=None):
        super(MinD3D, self).__init__()
        self.device = device
        # fMRI encoder
        self.fmri_encoder = fMRI_Encoder(config).to(device)
        self.fa_module = Feature_Aggregation(2048).to(device)

        # Diffusion prior
        d_prior_path = config['diff_prior_config_path']
        self.d_prior = OmegaConf.load(d_prior_path)
        self.prior_network = DiffusionNetwork(**self.d_prior["diffusion_prior"])
        self.diffusion_prior = FBDM(
            net = self.prior_network,
            timesteps = self.d_prior["timesteps"],
            sample_timesteps = self.d_prior["sample_timesteps"],
            cond_drop_prob = self.d_prior["cond_drop_prob"],
        ).to(device)

        # 3D Generator
        self.shape_generator = Latent_Adapted_Decoder(config, device).to(device)
        self.config = config

        # Generation Parameter
        self.points_batch_size = 100000
        self.threshold=config['test']['threshold']
        self.resolution0=config['generation']['resolution_0']
        self.upsampling_steps=config['generation']['upsampling_steps']
        self.sample=config['generation']['use_sampling']
        self.refinement_step=config['generation']['refinement_step']
        self.simplify_nfaces=config['generation']['simplify_nfaces']
        self.input_type=config['data']['input_type']
        self.padding=config['data']['padding']
        self.vol_bound = None
        self.with_normals = False
        self.nll_loss_value = nn.CrossEntropyLoss()  


    def get_fmri_embedding_batch(self, fmri_data, diffusion_bs=16, decoder_bs=4, ddim_steps=50):
        x = fmri_data
        b,f,w,h = x.size()

        # Encode the fMRI frames
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        x = self.fmri_encoder.forward_encoder_w_pred(x)
        fmri_feature = self.fa_module(x,b,f)
        fmri_embedding = self.diffusion_prior.sample(fmri_feature, num_samples_per_batch = diffusion_bs, cond_scale = 1.)
        return fmri_embedding.squeeze(1).repeat(decoder_bs, 1)


    def generate_mesh_from_embedding(self, embedding, bs, top_k=250, return_stats=True, greed=False):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        device = self.device
        stats_dict = {}
        kwargs = {}
        t0 = time.time()
      
        '''Autoregressive generate Feature Grid c'''
        plane_xz, plane_xy, plane_yz = self.shape_generator.batch_predict_cond_from_fmri_embedding(embedding, bs, device,top_k=top_k, greed=greed)
        stats_dict['time (Autoregressive Generate c)'] = time.time() - t0
        mesh_list = []
        for i in range(plane_xz.shape[0]):
            c_quantized = {'xz':plane_xz[i].unsqueeze(0),'xy':plane_xy[i].unsqueeze(0),'yz':plane_yz[i].unsqueeze(0)}
            mesh = self.generate_from_latent(c_quantized, stats_dict=stats_dict, **kwargs)
            mesh_list.append(mesh)
        return mesh_list

    # implement for Argus3D model
    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh


    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh


    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))
    
        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(pi.clone(), vol_bound['input_vol'], plane=key).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n
        
        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.shape_generator.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)
        
        return occ_hat


    def eval_points(self, p, c_quantized=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            if self.input_type == 'pointcloud_crop':
                if self.vol_bound is not None: # sliding-window manner
                    occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
                    occ_hats.append(occ_hat)
                else: # entire scene
                    pi_in = pi.unsqueeze(0).to(self.device)
                    pi_in = {'p': pi_in}
                    p_n = {}
                    for key in c.keys():
                        # normalized to the range of [0, 1]
                        p_n[key] = normalize_coord(pi.clone(), self.input_vol, plane=key).unsqueeze(0).to(self.device)
                    pi_in['p_n'] = p_n
                    with torch.no_grad():
                        occ_hat = self.shape_generator.decode(pi_in, c, **kwargs).logits
                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())
            else:
                pi = pi.unsqueeze(0).to(self.device)
                occ_hat = self.shape_generator.Stage1_decode(pi, c_quantized, **kwargs).logits
                    # occ_hat = self.model.decode(pi, c, **kwargs).logits
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat


    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.shape_generator.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

