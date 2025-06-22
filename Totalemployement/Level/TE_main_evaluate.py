import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.resnet_economic import ResNet50Economic, AFFResNet50Economic
from utils.dataset_loader import load_processed_dataset
from utils.evaluation import evaluate_model
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds = load_processed_dataset("data/employment01_20_train_dataset.pkl")
test_loader = DataLoader(test_ds, batch_size=8)

model = ResNet50Economic(econ_features_dim=11)
#AFFmodel = AFFResNet50Economic(econ_features_dim=11)  # Uncomment to use the AFF module

checkpoint = torch.load("best_model.pth")
#checkpoint_aff = torch.load("AFF_best_model.pth")  # Uncomment if using AFF model
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
#AFFmodel.load_state_dict(checkpoint_aff['model_state_dict'])
#AFFmodel.to(device)
criterion = nn.MSELoss()
test_loss, test_r2, test_rmsfe, actuals, preds, years, fips = evaluate_model(model, test_loader, criterion, device)

df = pd.DataFrame({'GeoFIPS': fips.flatten(), 'Year': years.flatten(), 'Actuals': actuals.flatten(), 'Predictions': preds.flatten()})
df.to_csv("test_predictions.csv", index=False)
