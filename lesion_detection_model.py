# lesion_detection_model.py

import torch
import torch.nn as nn

class LesionDetectionModel(nn.Module):
    def __init__(self):
        super(LesionDetectionModel, self).__init__()
        # Simple CNN for bounding box regression
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 6)  # Output bounding box coordinates (x1, y1, z1, x2, y2, z2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 32 * 32)
        x = F.relu(self.fc1(x))
        bbox = self.fc2(x)
        return bbox
