import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.aff_module import AFF
import pickle

class AFFResNet50Economic(nn.Module):
    def __init__(self, econ_features_dim=11, num_classes=1, return_concat=False):
        super().__init__()
        self.return_concat = return_concat

        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)

        # 修改7通道输入
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

        if self.return_concat:
            return pooled  
        return self.final_fc(pooled)
class DiffFromLevelModel(nn.Module):
    def __init__(self, pretrained_level_model, hidden_size=128):
        super().__init__()
        self.level_model = pretrained_level_model
        self.level_model.return_concat = False  
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, img1, img2, econ_features):
        out1 = self.level_model(img1, econ_features)  # shape: [B, 1]
        out2 = self.level_model(img2, econ_features)  # shape: [B, 1]
        combined = torch.cat([out1, out2], dim=1)     # shape: [B, 2]
        return self.mlp(combined)

def train_diff_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, patience=10):
    model.to(device)
    best_val_r2 = -float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ssr, total_tss, total_samples = 0, 0, 0, 0
        for img1, img2, econ_features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img1, img2 = img1.to(device), img2.to(device)
            econ_features = econ_features.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2, econ_features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            residuals = outputs - targets
            total_loss += loss.item() * targets.size(0)
            total_ssr += torch.sum(residuals ** 2).item()
            total_tss += torch.sum((targets - torch.mean(targets)) ** 2).item()
            total_samples += targets.size(0)

        train_loss = total_loss / total_samples
        train_r2 = 1 - (total_ssr / total_tss)

        val_loss, val_r2, *_ = evaluate_diff_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, R2: {train_r2:.4f} | Val Loss: {val_loss:.4f}, R2: {val_r2:.4f}")

        if val_r2 > best_val_r2:
            print(f"Validation R² improved: {best_val_r2:.4f} → {val_r2:.4f}. Saving model.")
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), "best_diff_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
import numpy as np

def evaluate_diff_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_ssr, total_tss, total_samples = 0, 0, 0, 0
    actuals, preds = [], []

    with torch.no_grad():
        for img1, img2, econ_features, targets in loader:
            img1, img2 = img1.to(device), img2.to(device)
            econ_features = econ_features.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            outputs = model(img1, img2, econ_features)
            loss = criterion(outputs, targets)

            outputs_exp = torch.exp(outputs)
            targets_exp = torch.exp(targets)

            residuals = outputs_exp - targets_exp
            total_loss += loss.item() * targets.size(0)
            total_ssr += torch.sum(residuals ** 2).item()
            total_tss += torch.sum((targets_exp - torch.mean(targets_exp)) ** 2).item()
            total_samples += targets.size(0)

            actuals.extend(targets_exp.cpu().numpy())
            preds.extend(outputs_exp.cpu().numpy())

    test_loss = total_loss / total_samples
    r2 = 1 - (total_ssr / total_tss)
    rmsfe = np.sqrt(np.mean(((np.array(preds) - np.array(actuals)) / (np.array(actuals) + 1e-8)) ** 2))
    
    print(f"Test Loss: {test_loss:.4f}, R²: {r2:.4f}, RMSFE: {rmsfe:.4f}")
    return test_loss, r2, np.array(actuals), np.array(preds), rmsfe





# 1. 加载 level 模型
level_model = AFFResNet50Economic(econ_features_dim=11, num_classes=1)
level_model.load_state_dict(torch.load("AFF_best_model.pth")['model_state_dict'])



# 2. 构建 diff 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diff_model = DiffFromLevelModel(level_model).to(device)

# 3. 定义优化器与损失
optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
criterion = nn.MSELoss()

# 4. 训练模型

# 加载处理后的数据集
def load_processed_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

train_dataset = load_processed_dataset("fprocessed_train_dataset.pkl")
val_dataset = load_processed_dataset("fprocessed_val_dataset.pkl")
test_dataset = load_processed_dataset("fprocessed_test_dataset.pkl")

# 创建数据加载器
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

train_diff_model(diff_model, train_loader, val_loader, optimizer, scheduler, criterion, device)

# 5. 评估模型
evaluate_diff_model(diff_model, test_loader, criterion, device)
