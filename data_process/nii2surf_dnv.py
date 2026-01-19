import argparse
import h5py
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os


def _make_flatmask(left,right, height=1024):
    from cortex import polyutils
    from PIL import Image, ImageDraw
    pts = np.vstack([left[0], right[0]])
    polys = np.vstack([left[1], right[1]+len(left[0])])

    left, right = polyutils.trace_poly(polyutils.boundary_edges(polys))

    aspect = (height / (pts.max(0) - pts.min(0))[1])
    lpts = (pts[left] - pts.min(0)) * aspect
    rpts = (pts[right] - pts.min(0)) * aspect

    im = Image.new('L', (int(aspect * (pts.max(0) - pts.min(0))[0]), height))
    draw = ImageDraw.Draw(im)
    draw.polygon(lpts[:,:2].ravel().tolist(), fill=255)
    draw.polygon(rpts[:,:2].ravel().tolist(), fill=255)
    extents = np.hstack([pts.min(0), pts.max(0)])[[0,3,1,4]]
    
    return np.array(im).T > 0, extents

def _make_vertex_cache(left,right, height=1024):
    from scipy import sparse
    from scipy.spatial import cKDTree
    flat = np.vstack([left[0], right[0]])
    polys = np.vstack([left[1], right[1]+len(left[0])])
    valid = np.unique(polys)
    fmax, fmin = flat.max(0), flat.min(0)
    size = fmax - fmin
    aspect = size[0] / size[1]
    width = int(aspect * height)
    grid = np.mgrid[fmin[0]:fmax[0]:width*1j, fmin[1]:fmax[1]:height*1j].reshape(2,-1)

    mask, extents = _make_flatmask(left,right, height=height)
    assert mask.shape[0] == width and mask.shape[1] == height

    kdt = cKDTree(flat[valid,:2])
    dist, vert = kdt.query(grid.T[mask.ravel()])
    dataij = (np.ones((len(vert),)), np.array([np.arange(len(vert)), valid[vert]]))
    return sparse.csr_matrix(dataij, shape=(mask.sum(), len(flat)))


def make_flatmap_image(left,right, data, height=1024, recache=False, nanmean=False, **kwargs):

    mask, extents = _make_flatmask(left, right, height=height)
    
    pixmap = _make_vertex_cache(left,right, height=height)

    if data.shape[0] > 1:
        raise ValueError("Input data was not the correct dimensionality - please provide 3D Volume or 2D Vertex data")

    if data.dtype != np.uint8:
        # Convert data to float to avoid image artifacts
        data = data.astype(np.float64)
    if data.dtype == np.uint8:
        img = np.zeros(mask.shape+(4,), dtype=np.uint8)
        img[mask] = pixmap * data.reshape(-1, 4)
        return img.transpose(1,0,2)[::-1], extents
    else:
        badmask = np.array(pixmap.sum(1) > 0).ravel()
        img = (np.nan*np.ones(mask.shape)).astype(data.dtype)
        mimg = (np.nan*np.ones(badmask.shape)).astype(data.dtype)

        # pixmap is a (pixels x voxels) sparse non-negative weight matrix
        # where each row sums to 1

        if not nanmean:
            # pixmap.dot(vec) gives mean of vec across cortical thickness
            mimg[badmask] = pixmap.dot(data.ravel())[badmask].astype(mimg.dtype)
        else:
            # to ignore nans in the weighted mean, nanmean =
            # sum(weights * non-nan values) / sum(weights on non-nan values)
            nonnan_sum = pixmap.dot(np.nan_to_num(data.ravel()))
            weights_on_nonnan = pixmap.dot((~np.isnan(data.ravel())).astype(data.dtype))
            nanmean_data = nonnan_sum / weights_on_nonnan
            mimg[badmask] = nanmean_data[badmask].astype(mimg.dtype)

        img[mask] = mimg

        return img.T[::-1], extents


class GLMtoSurfaces:
    def __init__(self,left,right,height = 1024, resolution='32k',lroi='',rroi='',output_images_dir='surface_images',data_dir=''):
        self.lroi = lroi
        self.rroi = rroi
        self.lroi_data = nib.load(lroi).agg_data()
        self.rroi_data = nib.load(rroi).agg_data()
        self.height = height
        self.output_images_dir = output_images_dir
        self.data_dir = data_dir
        self.left=nib.load(left).agg_data()
        self.right=nib.load(right).agg_data()

        self.left[0][:,0] -= self.left[0].max(0)[0]
        self.right[0][:,0] -= self.right[0].min(0)[0]

        self.lpts, self.lpolys = self.left
        self.rpts, self.rpolys = self.right
            
    def glm_to_surfaces(self,glm,fname):
        template_data = glm.reshape(1,-1).copy()
        newimg = nib.cifti2.cifti2.Cifti2Image(np.array(template_data).copy(),self.template.header,
                                               self.template.nifti_header,self.template.extra,self.template.file_map)
        newimg.to_filename(Path(self.output_surface_dir)/f'{fname}.dscalar.nii')
        rc,out=subprocess.getstatusoutput(f'wb_command -cifti-separate {self.output_surface_dir}/{fname}.dscalar.nii COLUMN -metric CORTEX_LEFT {self.output_surface_dir}/{fname}.l.func.gii -metric CORTEX_RIGHT {self.output_surface_dir}/{fname}.r.func.gii')
        
        if rc==0:
            left_data = nib.load(Path(self.output_surface_dir)/f'{fname}.l.func.gii').agg_data().copy()
            right_data = nib.load(Path(self.output_surface_dir)/f'{fname}.r.func.gii').agg_data().copy()
            if self.lroi!='' and self.rroi!='':
                left_data[self.lroi_data==0]=0
                right_data[self.rroi_data==0]=0
            lr = np.hstack([left_data,right_data]).reshape(1,-1)
            im,ex=make_flatmap_image(self.left,self.right,lr,nanmean=False)
            im[np.isnan(im)] = 0
            if self.lroi!='' and self.rroi!='':
                idx=np.where(im!=0)
                im = im[idx[0].min():idx[0].max(),idx[1].min():idx[1].max()]
            im = np.concatenate([np.expand_dims(im,axis=2),np.expand_dims(im,axis=2),np.expand_dims(im,axis=2)],axis=2)
            return im

    def glm_to_surfaces_pool(self,x,im_name):
        global args

        # im_name=Path(self.output_images_dir)/f'{args.sub}_{x.parts[-1]}.npz'
        if im_name.exists() and os.path.getsize(im_name)>3e6: # just to avoid repreprocessing
            print(f'{im_name} exists')
        else:
            left_data = nib.load(x/'left.32k_fs_LR.func.gii')
            right_data = nib.load(x/'right.32k_fs_LR.func.gii')
            len_data=len(left_data.darrays)
            assert len_data==len(right_data.darrays)
            images=[]
            for i in range(len_data):
                temp_left=left_data.darrays[i].data
                temp_right=right_data.darrays[i].data
                if self.lroi!='' and self.rroi!='':
                    temp_left[self.lroi_data==0]=0
                    temp_right[self.rroi_data==0]=0
                lr = np.hstack([temp_left,temp_right]).reshape(1,-1)
                im,ex=make_flatmap_image(self.left,self.right,lr,height=self.height,nanmean=False)
                im[np.isnan(im)] = 0
                if self.lroi!='' and self.rroi!='':
                    idx=np.where(im!=0)
                    im = im[idx[0].min():idx[0].max(),idx[1].min():idx[1].max()]
                im = np.expand_dims(im,axis=0)
                images.append(im)
            
            images=np.concatenate(images,axis=0).astype(np.float32)
            images[np.isnan(images)]=0
            with h5py.File(f'{im_name}', 'w') as f:
                ds=f.create_dataset('images',images.shape,compression="gzip", compression_opts=9, data=images)


import multiprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # cortex
    parser.add_argument('--surface_images_dir', type=str,required=False,default='../fMRI-Objaverse/obj_sub15/dtseries',help='surface_images_dir')
    parser.add_argument('--height', type=int, required=False,default=1024,help='surface image height')
    parser.add_argument('--lroi', type=str, required=False,default='./cortex.L.func.gii',help='lroi')
    parser.add_argument('--rroi', type=str, required=False,default='./cortex.R.func.gii',help='rroi')
    parser.add_argument('--n_jobs', type=int, required=False, default=1,help='')
    parser.add_argument('--data_dir', type=str,required=False,default='/mnt2/test/gaojianxiong/fMRI-Objaverse/obj_sub15/DNV',help='data dir')
    parser.add_argument('--left_surf', type=str, required=False,default='./S1200.L.flat.32k_fs_LR.surf.gii',help='left_surf')
    parser.add_argument('--right_surf', type=str, required=False,default='./S1200.R.flat.32k_fs_LR.surf.gii',help='right_surf')
    parser.add_argument('--sub', type=str, required=False,default='15',help='sub')


    args = parser.parse_args()
    print(args)

    surf2image = GLMtoSurfaces(output_images_dir=args.surface_images_dir,height=args.height,lroi=args.lroi,rroi=args.rroi,left=args.left_surf,right=args.right_surf)


    sub_dirs=list(Path(args.surface_images_dir).glob('sub*'))


    niis = []
    giis = []

    for n in list(Path("../fMRI-Objaverse/obj_sub15/dtseries").glob('*nii')):
        niis.append(n)
        giis.append(f'{n.stem}')

    gii=Path(args.data_dir)/'gii'
    gii.mkdir(exist_ok=True,parents=True)
    target_list = []
    for k,(n,g) in tqdm(enumerate(zip(niis,giis))):
        temp_gii=gii/f'{g}_gii'
        temp_h5=Path(args.data_dir)/'hdf5'/f'{g}.h5'
        temp_gii.mkdir(exist_ok=True,parents=True)
        # print(temp_gii)
        os.system(f'./workbench/bin_linux64/wb_command -cifti-separate {n} COLUMN -metric CORTEX_LEFT {temp_gii}/left.32k_fs_LR.func.gii -metric CORTEX_RIGHT {temp_gii}/right.32k_fs_LR.func.gii')
        # surf2image.glm_to_surfaces_pool(temp_gii, temp_h5)
        target_list.append((temp_gii,temp_h5))
    print(target_list)

    
    # Create a Pool of processes
    with multiprocessing.Pool() as pool:
        # Create a range of indices
        indices = range(32)
        results = pool.starmap(surf2image.glm_to_surfaces_pool, target_list)
    