# calculating image quality metrics (FID, KID, SSIM, PSNR, GMSD, LPIPS) for GPU

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import os
import numpy as np

# import h5py      ONLY for the .mat file case (i.e., SynDiff)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# Manual GMSD implementation (due to incompatibility with torchmetrics version)

# Following the formula and method from the following site:
# https://piqa.readthedocs.io/en/stable/api/piqa.gmsd.html#piqa.gmsd.ms_gmsd

# and from the following papers:

# Gradient Magnitude Similarity Deviation: An Highly Efficient Perceptual Image Quality Index (Xue et al., 2013)
# https://arxiv.org/abs/1308.3052
# Gradient Magnitude Similarity Deviation on multiple scales for color image quality assessment (Zhang et al., 2017)
# https://ieeexplore.ieee.org/document/7952357


# ==========================
def compute_gmsd(preds, target, c=0.0026):
    def gradient_magnitude(img):
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device
        )
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device
        )

        kx = kx.view(1, 1, 3, 3).expand(img.shape[1], 1, 3, 3).contiguous()
        ky = ky.view(1, 1, 3, 3).expand(img.shape[1], 1, 3, 3).contiguous()

        gx = F.conv2d(img, kx, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, ky, padding=1, groups=img.shape[1])

        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    gm_pred = gradient_magnitude(preds)
    gm_target = gradient_magnitude(target)

    # gms formula
    gms = (2 * gm_pred * gm_target + c) / (gm_pred**2 + gm_target**2 + c)
    return gms.std().item()


# ==========================
# Batch loading pngs (to ease computation) for KID/FID
# ==========================
def load_batch_png_uint8(file_list, folder_path):
    images = []
    for fname in file_list:
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        images.append(TF.to_tensor(img))
    batch = torch.stack(images)  # float [0,1]
    return (batch * 255).to(torch.uint8).to(device)


# ==========================
# Same as above, but for SSIM/PSNR and as float
# ==========================
def load_batch_png_float(file_list, folder_path):
    images = []
    for fname in file_list:
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        images.append(TF.to_tensor(img))
    return torch.stack(images).to(device)  # float [0,1]


# ==========================
# Loading images from mat files
# ==========================
# def load_real_images_from_mat(mat_path, dataset_key='data_fs'):
#     with h5py.File(mat_path, 'r') as f:
#         data = np.array(f[dataset_key])

#     # Reverse the normalization applied during training
#     data = (data * 0.5) + 0.5  # undo (data - 0.5) / 0.5

#     # Match the transpose from LoadDataSet
#     if data.ndim == 3:
#         data = np.transpose(data, (0, 2, 1))
#     else:
#         data = np.transpose(data, (1, 0, 3, 2))

#     return data  # shape: (N, 1, 256, 256)

# def mat_image_to_tensor(img_2d):
#     """Convert a single 2D grayscale image to float RGB tensor [0,1]."""
#     img = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
#     img = (img * 255).astype(np.uint8)
#     img_pil = Image.fromarray(img).convert('RGB')
#     return TF.to_tensor(img_pil)                  # float [0,1]


# ----------------------------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------------------------

gen_dir = "/your/generated/images/folder"
real_dir = "/your/ground_truth/images/folder"

gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(".png")])
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".png")])

# ONLY necessary for the .mat file case (i.e., SynDiff)
# real_mat_key = 'data_fs'

batch_size = 64

# Move metrics to GPU
fid = FrechetInceptionDistance(feature=2048).to(device)
kid = KernelInceptionDistance(subset_size=min(1000, len(gen_files))).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
    device
)


# ==========================
# FID/KID update - generated images
# ==========================
def update_fid_kid_in_batches(file_list, folder, is_real):
    for i in range(0, len(file_list), batch_size):
        batch = load_batch_png_uint8(file_list[i : i + batch_size], folder)
        fid.update(batch, real=is_real)
        kid.update(batch, real=is_real)
        print(
            f"  {'Real' if is_real else 'Generated'}: {min(i + batch_size, len(file_list))}/{len(file_list)}",
            end="\r",
        )
    print()


# ==========================
# FID/KID (for the .mat case)
# ==========================
# def update_real_from_mat(mat_path, mat_key):
#     images = load_real_images_from_mat(mat_path, mat_key)
#     n_images = images.shape[0]
#     for i in range(0, n_images, batch_size):
#         batch_tensors = []
#         for j in range(i, min(i + batch_size, n_images)):
#             img_2d = images[j, :, :]
#             img_2d = np.rot90(img_2d)
#             batch_tensors.append(mat_image_to_tensor(img_2d))
#         batch = torch.stack(batch_tensors)
#         batch = (batch * 255).to(torch.uint8).to(device)
#         fid.update(batch, real=True)
#         kid.update(batch, real=True)
#         print(f"  Real: {min(i+batch_size, n_images)}/{n_images}", end='\r')
#     print()


# ==========================
# Paired metrics: SSIM, PSNR, GMSD, LPIPS
# ==========================
def compute_paired_metrics(real_files, gen_files, real_folder, gen_folder):
    n = min(len(real_files), len(gen_files))

    if len(real_files) != len(gen_files):
        print(
            f"  Warning: {len(real_files)} real images but {len(gen_files)} generated. Using first {n}."
        )

    ssim_scores = []
    psnr_scores = []
    gmsd_scores = []
    lpips_scores = []

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)

        real_batch = load_batch_png_float(real_files[i:end], real_folder)
        fake_batch = load_batch_png_float(gen_files[i:end], gen_folder)

        ssim_scores.append(ssim(fake_batch, real_batch).item())
        psnr_scores.append(psnr(fake_batch, real_batch).item())
        gmsd_scores.append(compute_gmsd(fake_batch, real_batch))
        lpips_scores.append(lpips(fake_batch, real_batch).item())

        print(f"  Paired metrics: {end}/{n}", end="\r")
    print()

    return (
        np.mean(ssim_scores),
        np.mean(psnr_scores),
        np.mean(gmsd_scores),
        np.mean(lpips_scores),
    )


# For the .mat condition

# update_real_from_mat(real_mat, real_mat_key)
# update_gen_in_batches(gen_files, gen_dir, is_real=False)
# ssim_score, psnr_score, gmsd_score, lpips_score = compute_paired_metrics(real_mat, real_mat_key, gen_files, gen_dir)


# FID/KID --> using both gen and real images
update_fid_kid_in_batches(real_files, real_dir, is_real=True)
update_fid_kid_in_batches(gen_files, gen_dir, is_real=False)

# Compute FID/KID
fid_score = fid.compute()
kid_mean, kid_std = kid.compute()

#
# Paired metrics: SSIM, PSNR, GMSD, LPIPS
ssim_score, psnr_score, gmsd_score, lpips_score = compute_paired_metrics(
    real_files, gen_files, real_dir, gen_dir
)


# -------------------------
# Save results
# -------------------------
output_path = "/your/output/file"
with open(output_path, "w") as f:
    f.write("=== Image Quality Metrics ===\n\n")
    f.write(f"FID Score:        {fid_score.item():.4f}\n")
    f.write(f"KID Mean:         {kid_mean.item():.6f}\n")
    f.write(f"KID Std:          {kid_std.item():.6f}\n")
    f.write(f"SSIM:             {ssim_score:.4f}\n")
    f.write(f"PSNR:             {psnr_score:.4f} dB\n")
    f.write(f"GMSD:             {gmsd_score:.4f}\n")
    f.write(f"LPIPS:            {lpips_score:.4f}\n")
    f.write(f"\nGenerated images: {len(gen_files)}\n")
    f.write(f"Device used:      {device}\n")


print(f"FID:   {fid_score.item():.4f}")
print(f"KID:   {kid_mean.item():.6f} ± {kid_std.item():.6f}")
print(f"SSIM:  {ssim_score:.4f}")
print(f"PSNR:  {psnr_score:.4f} dB")
print(f"GMSD:  {gmsd_score:.4f}")
print(f"LPIPS: {lpips_score:.4f}")
