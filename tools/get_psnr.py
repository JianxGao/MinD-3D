import os
from PIL import Image
import lpips
from torchvision import transforms
import numpy as np
from tqdm import tqdm

def set_white_background(image_path):
    img = Image.open(image_path).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white_bg.paste(img, (0, 0), img)
    return white_bg.convert("RGB")


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # No error means infinite PSNR
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


loss_fn = lpips.LPIPS(net='alex').cuda()  # You can use 'vgg' or 'alex'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sid_list= os.listdir("~/MinD-3D/fmri_shapeobj_sub15_90k")

psnr_value_total_list = []
ssim_value_total_list = []
lpips_value_total_list = []
for i in tqdm(range(len(sid_list))):
    sid = sid_list[i]
    PSNR_value_total = 0
    SSIM_value_total = 0
    lpips_value_total = 0
    try:
        for k in range(6):
            gt_img = os.path.join("~/MinD-3D/dataset/fmri_objaverse/images", sid.replace(".obj",""), f"{k}.png")
            pred_img = os.path.join("~/MinD-3D/fmri_shapeobj_sub15_90k", sid, f"{k}.png")
            img1 = set_white_background(gt_img)
            img2 = set_white_background(pred_img)
            psnr_value = calculate_psnr(np.array(img1).astype(np.float64), np.array(img2).astype(np.float64))
            PSNR_value_total+=psnr_value

        PSNR_value_total = PSNR_value_total/6
        psnr_value_total_list.append(PSNR_value_total.item())
    except:
        continue
    
psnr_value_total_list = np.array(psnr_value_total_list)
print(f"PSNR: {psnr_value_total_list.mean()}")
