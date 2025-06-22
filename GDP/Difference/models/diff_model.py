import torch
import torch.nn as nn

class DiffFromLevelModel(nn.Module):
    def __init__(self, pretrained_level_model, hidden_size=1024):
        super().__init__()
        self.level_model = pretrained_level_model
        #self.level_model.return_concat = False
        self.mlp = nn.Sequential(
            nn.Linear(13, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(13, 1)
    def forward(self, img1, img2, econ_features):
        out1 = self.level_model(img1, econ_features)
        out2 = self.level_model(img2, econ_features)
        return self.mlp(torch.cat([out1, out2, econ_features], dim=1))

class DiffFromLevelModel2(nn.Module):
    def __init__(self, pretrained_level_model, hidden_size=1024):
        super().__init__()
        self.level_model = pretrained_level_model 
        self.img_feat_dim = self.level_model.output_dim if hasattr(self.level_model, 'output_dim') else 512
        econ_dim = 11

        self.mlp = nn.Sequential(
            nn.Linear(24, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, img1, img2, cur_econ, prev_econ):
        feat1 = self.level_model(img1, cur_econ)   # shape: [B, D]
        feat2 = self.level_model(img2, prev_econ)  # shape: [B, D]
        concat_input = torch.cat([feat1, feat2, cur_econ, prev_econ], dim=1)
        return self.mlp(concat_input)
