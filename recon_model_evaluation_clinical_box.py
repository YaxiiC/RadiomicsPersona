

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import math
from dataset import MRIDataset
from diffusion_model import DDPM3D
import pandas as pd

# Set device and paths
device_ids = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = '/home/yaxi/HealKnee_ddpm_central/best_diffusion_model.pth'  # Path to the best model weights
root_dir = "/home/yaxi/MRNet-v1.0_nii"  # Dataset root directory
visualization_save_path = '/home/yaxi/HealKnee_ddpm_central/diffusion_inference_visualizations'  # Path to save visualizations
labels_files = {
    'abnormal': os.path.join(root_dir, 'valid-abnormal.csv'),
    'acl': os.path.join(root_dir, 'valid-acl.csv'),
    'meniscus': os.path.join(root_dir, 'valid-meniscus.csv')
}

num_visualizations = 10
# Load the dataset and DataLoader for inference
target_size = (32, 128, 128)
test_dataset_recon = MRIDataset(root_dir, labels_files, phase='valid', target_size=target_size)
test_loader = DataLoader(test_dataset_recon, batch_size=4, shuffle=False)

# Load the diffusion model
diffusion_model = DDPM3D()

# Load model state_dict and remove the 'module.' prefix if needed
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
diffusion_model.load_state_dict(new_state_dict)

# Move the model to the device and wrap in DataParallel if multiple GPUs are available
diffusion_model = diffusion_model.to(device)
if torch.cuda.device_count() > 1:
    diffusion_model = torch.nn.DataParallel(diffusion_model, device_ids=device_ids)

diffusion_model.eval()

# Define PSNR calculation
def calculate_psnr(true_img, recon_img):
    mse = torch.mean((true_img - recon_img) ** 2).item()
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

# Define SSIM calculation
def calculate_ssim(true_img, recon_img):
    true_img_np = true_img.cpu().detach().numpy()
    recon_img_np = recon_img.cpu().detach().numpy()
    return ssim(true_img_np, recon_img_np, data_range=true_img_np.max() - true_img_np.min())

# Define MSE calculation
def calculate_mse(true_img, recon_img):
    return torch.mean((true_img - recon_img) ** 2).item()

# Define visualization function
def plot_reconstruction_results(original_images, reconstructed_images, corrupted_boxes, batch_idx, save_path):
    num_images = min(4, original_images.shape[0])  # Plot at most 4 images
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2 * i + 1)
        middle_slice_original = original_images[i, 0, original_images.shape[2] // 2, :, :].cpu().detach().numpy()
        plt.imshow(middle_slice_original, cmap='gray')
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')

        # Plot corrupted box on the original image
        box = corrupted_boxes[i]
        D, H, W = original_images.shape[2], original_images.shape[3], original_images.shape[4]
        x = box['y']
        y = box['z']
        width = box['h']
        height = box['w']
        rect = plt.Rectangle(
            (y, z := box['z']),  # (x, y) coordinates
            width,
            height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)

        # Reconstructed image
        plt.subplot(num_images, 2, 2 * i + 2)
        middle_slice_reconstructed = reconstructed_images[i, 0, reconstructed_images.shape[2] // 2, :, :].cpu().detach().numpy()
        plt.imshow(middle_slice_reconstructed, cmap='gray')
        plt.title(f'Reconstructed Image {i + 1}')
        plt.axis('off')

    plt.suptitle(f'Reconstruction Results for Batch {batch_idx}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()

# Inference and evaluation function
def perform_diffusion_inference(loader, model, save_dir, num_visualizations):
    psnr_list = []
    ssim_list = []
    mse_list = []
    visualization_count = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            # Create corrupted images
            corrupted_images = images.clone()
            corrupted_boxes = []
            for img_idx, img in enumerate(corrupted_images):
                _, D, H, W = img.shape
                # Calculate cube size as 60% of each dimension
                d, h, w = int(D * 0.6), int(H * 0.3), int(W * 0.6)

                # Calculate central starting indices
                x = (D - d) // 2
                y = (H - h) // 2
                z = (W - w) // 2
                img[:, x:x + d, y:y + h, z:z + w] = 0
                corrupted_boxes.append({'x': x, 'y': y, 'z': z, 'd': d, 'h': h, 'w': w})

            # Add random noise to corrupted images
            timesteps = model.module.timesteps if isinstance(model, torch.nn.DataParallel) else model.timesteps
            t = torch.randint(low=0, high=timesteps, size=(corrupted_images.size(0),), device=device)
            x_t, _ = model.module.add_noise(corrupted_images, t) if isinstance(model, torch.nn.DataParallel) else model.add_noise(corrupted_images, t)

            # Run model inference to reconstruct images
            reconstructed_images = model.module(x_t, t) if isinstance(model, torch.nn.DataParallel) else model(x_t, t)

            # Calculate metrics
            for i in range(images.size(0)):
                psnr = calculate_psnr(images[i], reconstructed_images[i])
                ssim_value = calculate_ssim(images[i, 0].cpu(), reconstructed_images[i, 0].cpu())
                mse_value = calculate_mse(images[i], reconstructed_images[i])

                psnr_list.append(psnr)
                ssim_list.append(ssim_value)
                mse_list.append(mse_value)

            # Save visualizations
            if visualization_count < num_visualizations:
                save_path = os.path.join(save_dir, f'diffusion_visualization_batch_{batch_idx}.png')
                plot_reconstruction_results(images, reconstructed_images, corrupted_boxes, batch_idx, save_path)
                visualization_count += 1

    # Compute average metrics
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)

    avg_ssim = np.mean(ssim_list)
    std_ssim = np.std(ssim_list)

    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    print(f"Avg PSNR: {avg_psnr:.4f}, Std PSNR: {std_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}, Std SSIM: {std_ssim:.4f}")
    print(f"Avg MSE: {avg_mse:.4f}, Std MSE: {std_mse:.4f}")

# Ensure visualization path exists
if not os.path.exists(visualization_save_path):
    os.makedirs(visualization_save_path)

# Perform inference, evaluation, and save visualizations
perform_diffusion_inference(test_loader, diffusion_model, save_dir=visualization_save_path, num_visualizations=num_visualizations)
print(f"Inference and visualizations saved at: {visualization_save_path}")
