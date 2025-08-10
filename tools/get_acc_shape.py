import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchmetrics.functional import accuracy
import lpips
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_images_from_dir(data_root, gt_root):
    test_dirs = os.listdir(data_root)
    pred_imgs, gt_imgs = [], []
    for dir_name in test_dirs:
        for i in range(6):
            pred_path = os.path.join(data_root, dir_name, f"{i}.png")
            gt_path = os.path.join(gt_root, dir_name.replace(".obj",""), f"{i}.png")
            pred_imgs.append(np.array(Image.open(pred_path).convert("RGB")))
            gt_imgs.append(np.array(Image.open(gt_path).convert("RGB")))
    return pred_imgs, gt_imgs

@torch.no_grad()
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range = [i for i in range(len(pred)) if i != class_id]
    accs = []
    for _ in range(num_trials):
        sampled = np.random.choice(pick_range, n_way - 1, replace=False)
        preds = torch.cat([pred[class_id].unsqueeze(0), pred[sampled]])
        acc = accuracy(
            preds.unsqueeze(0),
            torch.tensor([0], device=pred.device),
            top_k=top_k,
            task='multiclass',
            num_classes=len(preds)
        )
        accs.append(acc.item())
    return np.mean(accs), np.std(accs)

@torch.no_grad()
def get_n_way_top_k_acc(pred_imgs, gt_imgs, n_way, num_trials, top_k, device):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    accs, stds = [], []
    for pred, gt in tqdm(zip(pred_imgs, gt_imgs), total=len(gt_imgs), desc=f"{n_way}-way"):
        pred_tensor = preprocess(Image.fromarray(pred.astype(np.uint8))).unsqueeze(0).to(device)
        gt_tensor = preprocess(Image.fromarray(gt.astype(np.uint8))).unsqueeze(0).to(device)
        gt_class_id = model(gt_tensor).squeeze().softmax(0).argmax().item()
        pred_out = model(pred_tensor).squeeze().softmax(0)
        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        accs.append(acc)
        stds.append(std)
    return np.mean(accs)

def calculate_lpips(preds, gts, batch_size=64):
    loss_fn = lpips.LPIPS(net='alex').to('cuda')
    to_tensor = transforms.ToTensor()

    num_samples = len(preds)
    lpips_scores = []

    for i in range(0, num_samples, batch_size):
        batch_preds = [to_tensor(preds[j]) for j in range(i, min(i + batch_size, num_samples))]
        batch_gts = [to_tensor(gts[j]) for j in range(i, min(i + batch_size, num_samples))]

        pred_tensor = torch.stack(batch_preds).to('cuda')
        gt_tensor = torch.stack(batch_gts).to('cuda')

        with torch.no_grad():
            d = loss_fn(gt_tensor, pred_tensor).squeeze()
            lpips_scores.extend(d.cpu().numpy().tolist())

    return np.mean(lpips_scores)

def calculate_ssim(preds, gts):
    transform_resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    pred_tensor = torch.stack([transform_resize(transforms.ToTensor()(img)) for img in preds])
    gt_tensor = torch.stack([transform_resize(transforms.ToTensor()(img)) for img in gts])

    pred_gray = rgb2gray(pred_tensor.permute(0, 2, 3, 1).cpu().numpy())
    gt_gray = rgb2gray(gt_tensor.permute(0, 2, 3, 1).cpu().numpy())

    ssim_scores = [
        ssim(p, g, data_range=1.0, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        for p, g in zip(pred_gray, gt_gray)
    ]
    return np.mean(ssim_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', type=str, required=True,
                        help='Path to the directory containing predicted and ground truth images.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_imgs, gt_imgs = load_images_from_dir(args.pred_root,"~/dataset/shape_gt")

    # print("Calculating 2-way top-1 accuracy...")
    acc2 = get_n_way_top_k_acc(pred_imgs, gt_imgs, n_way=2, num_trials=100, top_k=1, device=device)
    print(f"2-way-top-1 accuracy: {acc2:.8f}")

    # print("Calculating 10-way top-1 accuracy...")
    acc10 = get_n_way_top_k_acc(pred_imgs, gt_imgs, n_way=10, num_trials=100, top_k=1, device=device)
    print(f"10-way-top-1 accuracy: {acc10:.8f}")

    # print("Calculating LPIPS...")
    lpips_score = calculate_lpips(pred_imgs, gt_imgs)
    print(f"LPIPS: {lpips_score:.8f}")

    # print("Calculating SSIM...")
    ssim_score = calculate_ssim(pred_imgs, gt_imgs)
    print(f"SSIM: {ssim_score:.8f}")

if __name__ == '__main__':
    main()
