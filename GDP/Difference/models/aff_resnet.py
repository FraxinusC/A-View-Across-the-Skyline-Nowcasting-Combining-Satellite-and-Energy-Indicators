import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from models.aff_module import AFF

class AFFResNet50Economic(nn.Module):
    def __init__(self, econ_features_dim=11, num_classes=1, return_concat=False):
        super().__init__()
        self.return_concat = return_concat
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)

        new_conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data[:, :3] = self.resnet.conv1.weight.data
        nn.init.kaiming_normal_(new_conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        self.resnet.conv1 = new_conv1
        self.resnet.fc = nn.Identity()

        self.econ_fc = nn.Linear(econ_features_dim, 2048)
        self.aff = AFF(2048)
        self.final_fc = nn.Linear(2048, num_classes)

    def forward(self, images, econ_features):
        img_features = self.resnet(images).unsqueeze(-1).unsqueeze(-1)
        econ_features = self.econ_fc(econ_features).unsqueeze(-1).unsqueeze(-1)
        fused = self.aff(img_features, econ_features)
        pooled = F.adaptive_avg_pool2d(fused, (1, 1)).view(fused.size(0), -1)
        return pooled if self.return_concat else self.final_fc(pooled)
