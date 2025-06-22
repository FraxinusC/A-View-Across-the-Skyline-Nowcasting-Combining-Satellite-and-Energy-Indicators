import torch
import torch.nn as nn

class AFF(nn.Module):
    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = channels // r
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        wei = self.sigmoid(self.local_att(xa) + self.global_att(xa))
        return 2 * x * wei + 2 * residual * (1 - wei)
