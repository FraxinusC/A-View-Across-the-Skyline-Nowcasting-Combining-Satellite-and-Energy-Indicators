import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.resnet_economic import ResNet50Economic, AFFResNet50Economic
from utils.dataset_loader import load_processed_dataset
from utils.training import train_model, AFF_train_model
from utils.evaluation import evaluate_model
import torch.nn as nn
from utils.seed import set_seed

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_ds = load_processed_dataset("data/employment01_20_train_dataset.pkl")
val_ds = load_processed_dataset("data/employment01_20_val_dataset.pkl")
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = ResNet50Economic(econ_features_dim=11)
#AFFmodel = AFFResNet50Economic(econ_features_dim=11)  # Uncomment to use the AFF module
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
criterion = nn.MSELoss()

model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, patience=5)


#AFFmodel = AFF_train_model(AFFmodel,train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, patience=5)