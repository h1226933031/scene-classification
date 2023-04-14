import torch.nn as nn
import torch.nn.functional as F


class CNN_3_layers(nn.Module):
    def __init__(self):
        super(CNN_3_layers, self).__init__()
        # Conv1: 1 input channel, 4 output channels, kernel size 7x7; shape to 128*128
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((128, 128)),
        )

        # Conv2: 4 input channel, 8 output channels, kernel size 5x5; shape to 64*64
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((64, 64)),
        )

        # Conv3: 8 input channel, 16 output channels, kernel size 3x3;shape to 8*8
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
        )

        # Fully connected layer
        self.fc = nn.Linear(16 * 8 * 8, 15)

    def forward(self, x):  # input: [batch_size, n_channels=1, height=224, weight=224]
        x = self.conv1(x)  # [batch_size, n_channels=4, height-7+1, weight-7+1]
        # x = F.relu(x)
        # x = self.pool1(x)  # [batch_size, n_channels=4, 128, 128]

        x = self.conv2(x)  # [batch_size, n_channels=8, 128-5+1, 128-5+1]
        # x = F.relu(x)
        # x = self.pool2(x)  # [batch_size, n_channels=8, 64, 64]

        x = self.conv3(x)  # [batch_size, n_channels=16, 64-3+1, 64-3+1]
        # x = F.relu(x)
        # x = self.pool3(x)  # [batch_size, n_channels=4, 8, 8]

        x = x.view(-1, 16 * 8 * 8)

        return self.fc(x)  # [batch_size, num_class=15]
