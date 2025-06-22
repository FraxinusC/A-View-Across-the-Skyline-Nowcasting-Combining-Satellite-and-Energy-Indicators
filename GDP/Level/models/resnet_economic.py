import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .aff_module import AFF

class AFFResNet50Economic(nn.Module):
    def __init__(self, econ_features_dim=6, num_classes=1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        new_conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_weights = self.resnet.conv1.weight.data
        new_conv1.weight.data[:, :3] = original_weights
        nn.init.kaiming_normal_(new_conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        self.resnet.conv1 = new_conv1
        self.resnet.fc = nn.Identity()
        self.econ_fc = nn.Linear(econ_features_dim, 2048)
        self.aff = AFF(2048)
        self.final_fc = nn.Linear(2048, num_classes)
        

    def forward(self, images, econ_features):
        img_features = self.resnet(images).unsqueeze(-1).unsqueeze(-1)
        econ_features = self.econ_fc(econ_features).unsqueeze(-1).unsqueeze(-1)
        combined = self.aff(img_features, econ_features)
        #combined = torch.cat((img_features, econ_features), dim=1)
        combined = F.adaptive_avg_pool2d(combined, (1, 1))
        return self.final_fc(combined.view(combined.size(0), -1))

class ResNet50Economic(nn.Module):
    def __init__(self, econ_features_dim=6, num_classes=1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        new_conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_weights = self.resnet.conv1.weight.data
        new_conv1.weight.data[:, :3] = original_weights
        nn.init.kaiming_normal_(new_conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        self.resnet.conv1 = new_conv1
        self.resnet.fc = nn.Identity()
        #self.econ_fc = nn.Linear(econ_features_dim, 2048)
        self.aff = AFF(2048)
        self.final_fc = nn.Linear(2059, num_classes)
        

    def forward(self, images, econ_features):
        img_features = self.resnet(images).unsqueeze(-1).unsqueeze(-1)
        econ_features = econ_features.unsqueeze(-1).unsqueeze(-1)
        combined = torch.cat((img_features, econ_features), dim=1)
        combined = F.adaptive_avg_pool2d(combined, (1, 1))
        return self.final_fc(combined.view(combined.size(0), -1))
