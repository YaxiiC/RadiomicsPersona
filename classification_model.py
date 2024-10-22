# classification_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LesionClassifier(nn.Module):
    def __init__(self, input_size):
        super(LesionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 3) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x  
