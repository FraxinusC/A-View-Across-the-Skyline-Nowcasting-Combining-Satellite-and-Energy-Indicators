import torch
import pickle
from torch.utils.data import DataLoader
from models.aff_resnet import AFFResNet50Economic
from models.diff_model import DiffFromLevelModel2
from utils.training2 import train_diff_model, evaluate_diff_model
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_processed_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load base level model
level_model = AFFResNet50Economic(econ_features_dim=11, num_classes=1)
checkpoint = torch.load("AFF_best_model.pth")
level_model.load_state_dict(checkpoint['model_state_dict'])

# Diff model
diff_model = DiffFromLevelModel2(level_model).to(device)

for param in diff_model.level_model.parameters():
    param.requires_grad = False

# Optimizer and Loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, diff_model.parameters()),
                             lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
criterion = nn.MSELoss()

# Load datasets
train_dataset = load_processed_dataset("data/5f_val_dataset.pkl")
val_dataset = load_processed_dataset("data/5f_val_dataset.pkl")
test_dataset = load_processed_dataset("data/5f_test_dataset.pkl")

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Train
train_diff_model(diff_model, train_loader, val_loader, optimizer, scheduler, criterion, device)

# Evaluate
evaluate_diff_model(diff_model, test_loader, criterion, device)
