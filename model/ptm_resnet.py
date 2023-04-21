import torch
import torch.nn as nn
from torchvision import models
from utils.load_data import read_data, Dataset_ptm


class Ptm_ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load pretrained ResNet50 Model
        self.resnet50 = models.resnet50(pretrained=True)

        # Freeze model parameters
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Change the final layer of ResNet50 Model for Transfer Learning
        self.fc_inputs = self.resnet50.fc.in_features  # ==2048
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(args.ptm_dropout),
            nn.Linear(256, args.class_num)
        )

    def forward(self, x):
        return self.resnet50(x)

