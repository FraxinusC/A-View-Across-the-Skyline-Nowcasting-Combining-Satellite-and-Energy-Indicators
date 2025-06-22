import torch
import numpy as np

def rmsfe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]) ** 2))

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, total_ssr, total_tss, total_samples = 0, 0, 0, 0
    actuals, preds, years, fips = [], [], [], []
    with torch.no_grad():
        for images, econ_features, targets, year, fip in loader:
            images, econ_features, targets = images.to(device), econ_features.to(device), targets.float().unsqueeze(1).to(device)
            outputs = model(images, econ_features)
            loss = criterion(outputs, targets)
            residuals = outputs - targets

            actuals.extend(targets.cpu().numpy())
            preds.extend(outputs.cpu().numpy())
            years.extend(year.cpu().numpy())
            fips.extend(fip.cpu().numpy())

            total_loss += loss.item() * targets.size(0)
            total_ssr += torch.sum(residuals ** 2).item()
            total_tss += torch.sum((targets - torch.mean(targets)) ** 2).item()
            total_samples += targets.size(0)

    r2 = 1 - (total_ssr / total_tss)
    mse = total_loss / total_samples
    rmsfe_score = rmsfe(actuals, preds)
    print(f"Loss: {mse:.4f}, R2: {r2:.4f}, RMSFE: {rmsfe_score:.4f}")
    return mse, r2, rmsfe_score, np.array(actuals), np.array(preds), np.array(years), np.array(fips)
