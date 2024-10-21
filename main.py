
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
batch_size = 1
learning_rate = 1e-4
target_size = (32, 256, 256)

root_dir = "/Users/chrissychen/Documents/PhD_2nd_year/miccai2025/MRNet-v1.0"  
labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
    }

# Dataset and DataLoader
train_dataset = MRIDataset(root_dir, labels_files, phase='train', view = 'coronal', target_size=target_size)
img, labels = train_dataset[0]
print("Image shape:", img.shape)
print("Labels:", labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

unet = UNet3D().to(device)
lesion_detector = LesionDetectionModel().to(device)
#######
classifier = LesionClassifier(input_size=6).to(device) 

criterion_recon = nn.MSELoss()
optimizer_recon = optim.Adam(unet.parameters(), lr=learning_rate)

criterion_cls = nn.BCELoss()
optimizer_cls = optim.Adam(list(lesion_detector.parameters()) + list(classifier.parameters()), lr=learning_rate)

# Training U-Net for reconstruction (Task 1)
print("Starting U-Net training...")
for epoch in range(num_epochs):
    unet.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)

      
        corrupted_images = images.clone()
        for img in corrupted_images:
            # Randomly zero out a cubic region
            _, D, H, W = img.shape
            d, h, w = np.random.randint(D//4, D//2), np.random.randint(H//4, H//2), np.random.randint(W//4, W//2)
            x, y, z = np.random.randint(0, D - d), np.random.randint(0, H - h), np.random.randint(0, W - w)
            img[:, x:x+d, y:y+h, z:z+w] = 0

        
        outputs = unet(corrupted_images)
        loss = criterion_recon(outputs, images)

       
        optimizer_recon.zero_grad()
        loss.backward()
        optimizer_recon.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save(unet.state_dict(), 'unet_model.pth')

# Lesion detection and classification (Task 2)
print("Starting lesion detection and classification training...")
for epoch in range(num_epochs):
    unet.eval()
    lesion_detector.train()
    classifier.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Reconstruct images using U-Net
        with torch.no_grad():
            reconstructed_images = unet(images)

        # Compute difference map
        difference_map = torch.abs(images - reconstructed_images)

        # Initial bounding box (center of the volume)
        # For simplicity, we'll assume the bounding box is fixed
        bbox = lesion_detector(difference_map)

        # Extract features (dummy features for illustration)
        features = torch.rand(images.size(0), 6).to(device)  # Replace with actual feature extraction

        # Classification
        outputs = classifier(features)
        loss = criterion_cls(outputs.squeeze(), labels)

        # Backward and optimize
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Classification Loss: {running_loss/len(train_loader):.4f}")

# Save the trained models
torch.save(lesion_detector.state_dict(), 'lesion_detector.pth')
torch.save(classifier.state_dict(), 'classifier.pth')


