
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

def plot_reconstruction_results(original_images, reconstructed_images, epoch, save_path):
    num_images = min(4, original_images.shape[0])  # Plot at most 4 images
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(original_images[i, 0, :, :, original_images.shape[4] // 2].cpu().detach().numpy(), cmap='gray')
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')

        # Reconstructed image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(reconstructed_images[i, 0, :, :, reconstructed_images.shape[4] // 2].cpu().detach().numpy(), cmap='gray')
        plt.title(f'Reconstructed Image {i + 1}')
        plt.axis('off')

    plt.suptitle(f'Reconstruction Results at Epoch {epoch}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close()


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

num_epochs = 100000
batch_size = 16
learning_rate = 1e-4
patience = 10
model_save_frequency = 50  # Save models every 50 epochs
max_models_to_keep = 5  # Maximum number of models to keep
target_size = (32, 128, 128)

root_dir = "/home/yaxi/MRNet-v1.0"  
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

# Dataset and DataLoader for Detection and Classification Model
train_dataset = MRIDataset(root_dir, labels_files, phase='train', view='coronal', target_size=target_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

unet = UNet3D().to(device)
lesion_detector = LesionDetectionModel().to(device)
classifier = LesionClassifier(input_size=6).to(device) 

criterion_recon = nn.MSELoss()
optimizer_recon = optim.Adam(unet.parameters(), lr=learning_rate)

criterion_bbox = nn.MSELoss()
optimizer_bbox = optim.Adam(lesion_detector.parameters(), lr=learning_rate)

criterion_cls = nn.BCELoss()
optimizer_cls = optim.Adam(list(lesion_detector.parameters()) + list(classifier.parameters()), lr=learning_rate)

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
    for batch_idx, (images, _) in enumerate(train_loader_recon):
        #print(f"Batch {batch_idx + 1}/{len(train_loader)}")
        
        images = images.to(device)
        
        corrupted_images = images.clone()
        for img_idx, img in enumerate(corrupted_images):
            # Randomly zero out a cubic region
            _, D, H, W = img.shape
            min_volume = 0.5 * D * H * W
            corrupted = False
            while not corrupted:
                d, h, w = np.random.randint(D // 4, D // 2), np.random.randint(H // 4, H // 2), np.random.randint(W // 4, W // 2)
                if d * h * w >= min_volume:
                    x = np.random.randint(0, D - d)
                    y = np.random.randint(0, H - h)
                    z = np.random.randint(0, W - w)
                    img[:, x:x + d, y:y + h, z:z + w] = 0
                    corrupted = True
        
        outputs = unet(corrupted_images)
        loss = criterion_recon(outputs, images)
        print("Loss computed:", loss.item())

        optimizer_recon.zero_grad()
        loss.backward()

        optimizer_recon.step()
        running_loss += loss.item()
        for i in range(images.size(0)):
            psnr = calculate_psnr(images[i], outputs[i])
            psnr_total += psnr
        print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, PSNR: {psnr:.4f}")
    
    

        if (epoch + 1) % model_save_frequency == 0 and batch_idx == 0:
            save_path = f"{visualization_save_path}/reconstruction_epoch_{epoch + 1}.png"
            plot_reconstruction_results(images, outputs, epoch + 1, save_path)
            saved_visualization_groups.append(save_path)

            # Keep only the latest `max_visualization_groups` visualizations
            if len(saved_visualization_groups) > max_models_to_keep:
                os.remove(saved_visualization_groups.pop(0))
        if batch_idx == 0:
            break
    avg_loss = running_loss / len(train_loader)
    avg_psnr = psnr_total / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}")

    if (epoch + 1) % model_save_frequency == 0:
        model_path = f'{model_save_path}/unet_epoch_{epoch + 1}.pth'
        torch.save(unet.state_dict(), model_path)
        saved_unet_models.append(model_path)

        # Keep only the latest `max_models_to_keep` models
        if len(saved_unet_models) > max_models_to_keep:
            os.remove(saved_unet_models.pop(0))

    # Early Stopping Logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(unet.state_dict(), 'best_unet_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break
print("Finished training U-Net.")

best_loss_cls = float('inf')
early_stopping_counter_cls = 0
saved_detection_models = []
saved_classifier_models = []
# Lesion detection and classification (Task 2)
print("Starting lesion detection and classification training...")
for epoch in range(num_epochs):
    unet.eval()
    lesion_detector.train()
    classifier.train()
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size, _, D, H, W = images.shape
        print("Batch size:", batch_size, "Image dimensions (D, H, W):", D, H, W) 

        predicted_bboxes = lesion_detector(images.detach())
        print("Predicted bounding boxes shape:", predicted_bboxes.shape)

        hollowed_images = images.clone()

        # Hollow out the images using the predicted bounding boxes
        for i in range(batch_size):
            x_min, y_min, z_min, x_max, y_max, z_max = predicted_bboxes[i].int()
            print("Predicted bounding boxes shape:", predicted_bboxes.shape)
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, D - 1))
            y_min = max(0, min(y_min, H - 1))
            z_min = max(0, min(z_min, W - 1))
            x_max = max(x_min + 1, min(x_max, D))
            y_max = max(y_min + 1, min(y_max, H))
            z_max = max(z_min + 1, min(z_max, W))

            if x_max > x_min and y_max > y_min and z_max > z_min:
                hollowed_images[i, :, x_min:x_max, y_min:y_max, z_min:z_max] = 0  # Hollow out the region inside bbox
            else:
                pass

        with torch.no_grad():
            reconstructed_images = unet(images)
            #print("Reconstructed images shape:", reconstructed_images.shape)
            

        # Step 3: Compare the lesion box-cropped original image and the lesion box-cropped reconstructed image and extract features
        features = []
        valid_indices = []
        for i in range(batch_size):
            x_min, y_min, z_min, x_max, y_max, z_max = predicted_bboxes[i].int()
            #print(f"Processing bbox for image {i}: x_min={x_min}, y_min={y_min}, z_min={z_min}, x_max={x_max}, y_max={y_max}, z_max={z_max}")
            
            x_min = max(0, min(x_min, D - 1))
            y_min = max(0, min(y_min, H - 1))
            z_min = max(0, min(z_min, W - 1))
            x_max = max(x_min + 1, min(x_max, D))
            y_max = max(y_min + 1, min(y_max, H))
            z_max = max(z_min + 1, min(z_max, W))

            if x_max <= x_min or y_max <= y_min or z_max <= z_min:
                
                continue

            # Extract the region inside the bounding box from both original and reconstructed images
            original_region = images[i, :, x_min:x_max, y_min:y_max, z_min:z_max]
            reconstructed_region = reconstructed_images[i, :, x_min:x_max, y_min:y_max, z_min:z_max]
            print(f"Original region shape: {original_region.shape}, Reconstructed region shape: {reconstructed_region.shape}")

            # Mean Squared Error
            mse = F.mse_loss(original_region, reconstructed_region).item()

            # Mean Absolute Error
            mae = F.l1_loss(original_region, reconstructed_region).item()

            # Similarity
            orig_flat = original_region.view(-1)
            recon_flat = reconstructed_region.view(-1)
            cosine_sim = F.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)).item()

            # Intensity Difference Statistics
            diff = original_region - reconstructed_region
            mean_diff = torch.mean(diff).item()
            std_diff = torch.std(diff).item()
            print(f"Image {i} - Mean Diff: {mean_diff}, Std Diff: {std_diff}")

            features.append([mse, mae, cosine_sim, mean_diff, std_diff])
            valid_indices.append(i)
        
        if not features:
            continue 

        features = torch.tensor(features, dtype=torch.float32).to(device)
        valid_labels = labels[valid_indices]

        # Step 4: Classification
        outputs = torch.sigmoid(classifier(features))
        cls_loss = criterion_cls(outputs, labels)

        optimizer_cls.zero_grad()
        cls_loss.backward()
        optimizer_cls.step()

        running_cls_loss += cls_loss.item()

        gt_bboxes = torch.tensor([
            [D // 4, H // 4, W // 4, 3 * D // 4, 3 * H // 4, 3 * W // 4]
            for _ in range(len(valid_indices))
        ], dtype=torch.float32).to(device)

        # Compute loss for lesion detection model
        bbox_loss = criterion_bbox(predicted_bboxes[valid_indices], gt_bboxes)
        optimizer_bbox.zero_grad()
        bbox_loss.backward()
        optimizer_bbox.step()

        running_bbox_loss += bbox_loss.item()
    
    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, pred_labels)
    precision = precision_score(all_labels, pred_labels, average='binary')
    recall = recall_score(all_labels, pred_labels, average='binary')
    f1 = f1_score(all_labels, pred_labels, average='binary')

    avg_cls_loss = running_cls_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Classification Loss: {avg_cls_loss:.4f}, "
          f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    if (epoch + 1) % model_save_frequency == 0:
        detection_model_path = f'{model_save_path}/lesion_detector_epoch_{epoch + 1}.pth'
        classifier_model_path = f'{model_save_path}/classifier_epoch_{epoch + 1}.pth'
        torch.save(lesion_detector.state_dict(), detection_model_path)
        torch.save(classifier.state_dict(), classifier_model_path)
        saved_detection_models.append(detection_model_path)
        saved_classifier_models.append(classifier_model_path)

        # Keep only the latest `max_models_to_keep` models
        if len(saved_detection_models) > max_models_to_keep:
            os.remove(saved_detection_models.pop(0))
        if len(saved_classifier_models) > max_models_to_keep:
            os.remove(saved_classifier_models.pop(0))
    
    if avg_cls_loss < best_loss_cls:
        best_loss_cls = avg_cls_loss
        torch.save(lesion_detector.state_dict(), 'best_lesion_detector_model.pth')
        torch.save(classifier.state_dict(), 'best_classifier_model.pth')
        early_stopping_counter_cls = 0
    else:
        early_stopping_counter_cls += 1

    if early_stopping_counter_cls >= patience:
        print("Early stopping triggered for lesion detection and classification training.")
        break

    torch.save(lesion_detector.state_dict(), f'lesion_detector_epoch_{epoch + 1}.pth')
    torch.save(classifier.state_dict(), f'classifier_epoch_{epoch + 1}.pth')   


