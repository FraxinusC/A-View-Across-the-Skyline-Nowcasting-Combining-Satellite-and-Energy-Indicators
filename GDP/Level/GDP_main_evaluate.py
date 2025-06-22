import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.resnet_economic import ResNet50Economic, AFFResNet50Economic
from utils.dataset_loader import load_processed_dataset
from utils.evaluation import evaluate_model
import torch.nn as nn

use_aff = False  # Set to False to use the standard model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ds = load_processed_dataset("data/GDP01_20_train_dataset.pkl")
test_loader = DataLoader(test_ds, batch_size=8)

criterion = nn.MSELoss()

if use_aff:
    model = AFFResNet50Economic(econ_features_dim=11)
    checkpoint = torch.load("AFF_best_model.pth")
else:
    model = ResNet50Economic(econ_features_dim=11)
    checkpoint = torch.load("best_model.pth")

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

test_loss, test_r2, test_rmsfe, actuals, preds, years, fips = evaluate_model(
    model, test_loader, criterion, device
)

df = pd.DataFrame({
    'GeoFIPS': fips.flatten(),
    'Year': years.flatten(),
    'Actuals': actuals.flatten(),
    'Predictions': preds.flatten()
})

model_name = "AFF" if use_aff else "standard"
df.to_csv(f"test_predictions_{model_name}.csv", index=False)
