import torch
import numpy as np
import argparse
from src.utils import load_config
from src.mind3d import MinD3D
from src.utils import torch_init_model, set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D mesh based on image input')
    parser.add_argument('--config', type=str, help='Path to config file.', default="./configs/mind3d.yaml")
    # 3D decoder
    parser.add_argument('--transformer_embed_dim', type=int, default=3072)
    parser.add_argument('--transformer_n_head', type=int, default=24)
    parser.add_argument('--transformer_layer', type=int, default=32)
    parser.add_argument('--topk', type=int, default=250)
    parser.add_argument('--sub_id', type=str, default="0006")
    # demo data
    parser.add_argument('--uid', type=str, default="b5d0ae4f723bce81f119374ee5d5f944")
    parser.add_argument('--out_dir', type=str, default="fmri_img2shape")
    parser.add_argument('--check_point_path', type=str, default="mind3d_30k.pt")

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')

    # initialize random seed
    seed=100
    set_random_seed(seed)

    device = torch.device("cuda")

    # load data
    fmri_data = np.load("./demo/{}.npy".format(args.uid))
    fmri_data = torch.from_numpy(fmri_data[2:8]).unsqueeze(0).cuda()

    # load model
    model_full = MinD3D(cfg, device=device)
    state_dict = torch.load(args.check_point_path, map_location='cpu')
    torch_init_model(model_full, state_dict)
    model_full.eval()
    
    with torch.no_grad(): 
        fmri_embedding = model_full.get_fmri_embedding_batch(fmri_data, diffusion_bs=128, decoder_bs=1)
        mesh_list = model_full.generate_mesh_from_embedding(fmri_embedding, bs=1, top_k=args.topk)

        for j in range(len(mesh_list)):
            mesh = mesh_list[j]
            if not mesh.vertices.shape[0]:
                continue
            mesh.export('./demo/{}.obj'.format(args.uid))
