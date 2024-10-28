import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import math
from dataset import MRIDataset
from reconstruction_model import UNet3D

# Set device and paths
device_ids = [0, 1, 2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = '/home/yaxi/healknee_model/unet_epoch_19650.pth'  # Path to the best model weights
root_dir = "/home/yaxi/MRNet-v1.0_gpu"  # Dataset root directory
visualization_save_path = '/home/yaxi/healknee_model/inference_visualizations'  # Path to save visualizations
labels_files = {
    'abnormal': os.path.join(root_dir, 'valid-abnormal.csv'),
    'acl': os.path.join(root_dir, 'valid-acl.csv'),
    'meniscus': os.path.join(root_dir, 'valid-meniscus.csv')
}

# Load the dataset and DataLoader for inference
target_size = (32, 128, 128)
test_dataset = MRIDataset(root_dir, labels_files, phase='valid', view='coronal', target_size=target_size)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # Use batch size of 4 for visualization

unet = UNet3D()

# Load model state_dict and remove the 'module.' prefix if needed
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
unet.load_state_dict(new_state_dict)

# Move the model to the device and wrap in DataParallel if multiple GPUs are available
unet = unet.to(device)
if torch.cuda.device_count() > 1:
    unet = torch.nn.DataParallel(unet, device_ids=device_ids)

unet.eval()

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
def plot_reconstruction_results(original_images, reconstructed_images, corrupted_boxes, epoch, save_path):
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
        rect = plt.Rectangle(
            (box['y'], box['z']),
            box['h'],
            box['w'],
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

    plt.suptitle(f'Reconstruction Results at Epoch {epoch}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()

# Inference and evaluation function
def perform_inference_and_plot(loader, model, epoch, save_path):
    total_psnr, total_ssim, total_mse = 0, 0, 0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            
            # Create corrupted images
            corrupted_images = images.clone()
            corrupted_boxes = []
            for img_idx, img in enumerate(corrupted_images):
                _, D, H, W = img.shape
                corrupted = False
                max_attempts = 10
                attempts = 0

                # Randomly zero out a region in each image
                while not corrupted and attempts < max_attempts:
                    d, h, w = np.random.randint(D // 4, D // 2), np.random.randint(H // 4, H // 2), np.random.randint(W // 4, W // 2)
                    x = np.random.randint(0, D - d)
                    y = np.random.randint(0, H - h)
                    z = np.random.randint(0, W - w)
                    img[:, x:x + d, y:y + h, z:z + w] = 0
                    corrupted_boxes.append({'x': x, 'y': y, 'z': z, 'd': d, 'h': h, 'w': w})
                    corrupted = True

            # Run model inference
            outputs = model(corrupted_images)

            # Calculate metrics for each image in the batch
            for i in range(images.size(0)):
                psnr = calculate_psnr(images[i], outputs[i])
                ssim_value = calculate_ssim(images[i, 0].cpu(), outputs[i, 0].cpu())
                mse_value = calculate_mse(images[i], outputs[i])

                total_psnr += psnr
                total_ssim += ssim_value
                total_mse += mse_value
                count += 1
            
            # Visualize the reconstruction results
            plot_reconstruction_results(images, outputs, corrupted_boxes, epoch, save_path)
            break  # Stop after one batch

    # Compute average metrics
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mse = total_mse / count

    print(f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}, Avg MSE: {avg_mse:.4f}")

# Ensure visualization path exists
if not os.path.exists(visualization_save_path):
    os.makedirs(visualization_save_path)

# Perform inference, evaluation, and save visualization
inference_save_path = f"{visualization_save_path}/inference_visualization.png"
perform_inference_and_plot(test_loader, unet, epoch=1, save_path=inference_save_path)
print(f"Inference and visualization saved at: {inference_save_path}")
