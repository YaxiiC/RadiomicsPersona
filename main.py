
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 3
batch_size = 2
learning_rate = 1e-4
target_size = (32, 256, 256)

root_dir = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0"  
labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }

# Dataset and DataLoader
train_dataset = MRIDataset(root_dir, labels_files, phase='train', view ='coronal', target_size=target_size)
img, labels = train_dataset[0]
print("Image shape:", img.shape)
print("Labels:", labels)
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
    for batch_idx, (images, _) in enumerate(train_loader):
        #print(f"Batch {batch_idx + 1}/{len(train_loader)}")
        
        images = images.to(device)
        
        corrupted_images = images.clone()
        for img_idx, img in enumerate(corrupted_images):
            # Randomly zero out a cubic region
            _, D, H, W = img.shape
            d, h, w = np.random.randint(D//4, D//2), np.random.randint(H//4, H//2), np.random.randint(W//4, W//2)
            x, y, z = np.random.randint(0, D - d), np.random.randint(0, H - h), np.random.randint(0, W - w)
            img[:, x:x+d, y:y+h, z:z+w] = 0
            print(f"Corrupted region for image {img_idx}: x={x}, d={d}, y={y}, h={h}, z={z}, w={w}")

        
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Avg PSNR: {psnr_total/len(train_loader):.4f}")

print("Finished training U-Net.")
torch.save(unet.state_dict(), 'unet_model.pth')

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
        print("Images and labels moved to device.")

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
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            z_min = max(0, z_min)
            x_max = min(D, x_max)
            y_max = min(H, y_max)
            z_max = min(W, z_max)
            if x_max > x_min and y_max > y_min and z_max > z_min:
                hollowed_images[i, :, x_min:x_max, y_min:y_max, z_min:z_max] = 0  # Hollow out the region inside bbox
            else:
                pass

        with torch.no_grad():
            reconstructed_images = unet(images)
            print("Reconstructed images shape:", reconstructed_images.shape)
            

        # Step 3: Compare the lesion box-cropped original image and the lesion box-cropped reconstructed image and extract features
        features = []
        valid_indices = []
        for i in range(batch_size):
            x_min, y_min, z_min, x_max, y_max, z_max = predicted_bboxes[i].int()
            print(f"Processing bbox for image {i}: x_min={x_min}, y_min={y_min}, z_min={z_min}, x_max={x_max}, y_max={y_max}, z_max={z_max}")
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            z_min = max(0, z_min)
            x_max = min(D, x_max)
            y_max = min(H, y_max)
            z_max = min(W, z_max)
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Classification Loss: {running_cls_loss/len(train_loader):.4f}, "
          f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

torch.save(lesion_detector.state_dict(), 'lesion_detector.pth')
torch.save(classifier.state_dict(), 'classifier.pth')


