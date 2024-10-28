
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MRIDataset
from reconstruction_model import UNet3D
from lesion_detection_model import LesionDetectionModel
from classification_model import LesionClassifier
import numpy as np
import torch.nn.functional as F

from torchvision import transforms
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random

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

        corrupted_index = i * 3 + random.choice([0, 1, 2])
        selected_box = corrupted_boxes[corrupted_index]

        # Plot the corrupted image
        plt.subplot(num_images, 2, 2 * i + 2)
        middle_slice_reconstructed = reconstructed_images[corrupted_index, 0, reconstructed_images.shape[2] // 2, :, :].cpu().detach().numpy()
        plt.imshow(middle_slice_reconstructed, cmap='gray')
        plt.title(f'Reconstructed Image {i + 1}')
        plt.axis('off')

        # Add rectangle for the selected corrupted box
        rect = plt.Rectangle(
            (selected_box['y'], selected_box['z']),
            selected_box['h'],
            selected_box['w'],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        
    plt.suptitle(f'Reconstruction Results at Epoch {epoch}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()
   

device_ids = [0, 1, 2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100000
batch_size = 10
learning_rate = 1e-4
patience = 5000
model_save_frequency = 50  # Save models every 50 epochs
max_models_to_keep = 5  # Maximum number of models to keep
target_size = (32, 128, 128)

root_dir = "/home/yaxi/MRNet-v1.0_gpu"  
labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }
model_save_path = '/home/yaxi/healknee_model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

visualization_save_path = '/home/yaxi/healknee_model/visualizations'
if not os.path.exists(visualization_save_path):
    os.makedirs(visualization_save_path)


# Dataset and DataLoader for Reconstruction Model
train_dataset_recon = MRIDataset(root_dir, labels_files, phase='train', view='coronal', target_size=target_size)
train_dataset_recon.labels_abnormal = train_dataset_recon.labels_abnormal[train_dataset_recon.labels_abnormal['Label'] == 0]
train_loader_recon = DataLoader(train_dataset_recon, batch_size=batch_size, shuffle=True)
total_images_recon = len(train_dataset_recon)
print(f"Total number of images in train_loader_recon: {total_images_recon}")

# Dataset and DataLoader for Detection and Classification Model
train_dataset = MRIDataset(root_dir, labels_files, phase='train', view='coronal', target_size=target_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
total_images = len(train_dataset)
print(f"Total number of images in train_loader_recon: {total_images}")

unet = UNet3D().to(device)
unet = nn.DataParallel(unet, device_ids=device_ids)
#lesion_detector = LesionDetectionModel().to(device)
#classifier = LesionClassifier(input_size=6).to(device) 

criterion_recon = nn.MSELoss()
optimizer_recon = optim.Adam(unet.module.parameters(), lr=learning_rate)

#criterion_bbox = nn.MSELoss()
#optimizer_bbox = optim.Adam(lesion_detector.parameters(), lr=learning_rate)

#criterion_cls = nn.BCELoss()
#optimizer_cls = optim.Adam(list(lesion_detector.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Metrics
best_loss_unet = float('inf')
early_stopping_counter_unet = 0
saved_unet_models = []
saved_visualization_groups = []


def calculate_psnr(true_img, recon_img):
    mse = torch.mean((true_img - recon_img) ** 2).item()
    if mse == 0:
        return 100
    pixel_max = 1.0  # assuming normalized image
    return 20 * math.log10(pixel_max / math.sqrt(mse))

# Training U-Net for reconstruction (Task 1)
print("Starting U-Net training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    unet.train()
    running_loss = 0.0
    psnr_total, ssim_total = 0, 0
    corrupted_boxes = []
    
    for batch_idx, (images, _) in enumerate(train_loader_recon):
        #print(f"Processing batch {batch_idx + 1}/{len(train_loader_recon)}")
        images = images.to(device)
        #print(f"Loaded images with shape: {images.shape}")

        corrupted_images = []
        new_corrupted_boxes = []

        for img_idx, img in enumerate(images):
            _, D, H, W = img.shape
        
            for corruption_num in range(3):
                corrupted_img = img.clone()
                d, h, w = np.random.randint(D // 4, D // 2), np.random.randint(H // 4, H // 2), np.random.randint(W // 4, W // 2)
                x, y, z = np.random.randint(0, D - d), np.random.randint(0, H - h), np.random.randint(0, W - w)
                
                corrupted_img[:, x:x + d, y:y + h, z:z + w] = 0
                #print(f"Corruption {corruption_num + 1} for image {img_idx + 1}: (x, y, z, d, h, w) = ({x}, {y}, {z}, {d}, {h}, {w})")
                
                corrupted_images.append(corrupted_img)
                new_corrupted_boxes.append({'x': x, 'y': y, 'z': z, 'd': d, 'h': h, 'w': w})

        if corrupted_images:
            #print(f"Total corrupted images in batch {batch_idx + 1}: {len(corrupted_images)}")
            corrupted_images = torch.stack(corrupted_images).to(device)
        else:
            print("Error: No corrupted images were created.")
            continue

        corrupted_boxes.extend(new_corrupted_boxes)
        outputs = unet(corrupted_images)
        loss = criterion_recon(outputs, images.repeat(3, 1, 1, 1, 1))

        optimizer_recon.zero_grad()
        loss.backward()
        optimizer_recon.step()
        running_loss += loss.item()

    
        for i in range(images.size(0)):
            psnr = calculate_psnr(images[i // 3], outputs[i])
            psnr_total += psnr

    # Save visualizations at the epoch
        if (epoch + 1) % model_save_frequency == 0 and batch_idx == 0:
            save_path = f"{visualization_save_path}/reconstruction_epoch_{epoch + 1}.png"
            plot_reconstruction_results(images, outputs, corrupted_boxes, epoch + 1, save_path)
            saved_visualization_groups.append(save_path)
        
            # Keep only the latest `max_models_to_keep` visualizations
            if len(saved_visualization_groups) > max_models_to_keep:
                os.remove(saved_visualization_groups.pop(0))
        if batch_idx == 0:
            break

    avg_loss = running_loss / len(train_loader)
    avg_psnr = psnr_total / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}")

    if (epoch + 1) % model_save_frequency == 0:
        model_path = f'{model_save_path}/unet_epoch_{epoch + 1}.pth'
        torch.save(unet.module.state_dict(), model_path)
        saved_unet_models.append(model_path)

        # Keep only the latest `max_models_to_keep` models
        if len(saved_unet_models) > max_models_to_keep:
            os.remove(saved_unet_models.pop(0))

    # Early Stopping Logic
    if avg_loss < best_loss_unet:
        best_loss_unet = avg_loss
        torch.save(unet.module.state_dict(), 'best_unet_model.pth')
        early_stopping_counter_unet = 0
    else:
        early_stopping_counter_unet += 1
    if early_stopping_counter_unet >= patience:
        print("Early stopping triggered.")
        break
print("Finished training U-Net.")




