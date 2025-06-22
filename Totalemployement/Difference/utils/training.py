import torch
import numpy as np
from tqdm import tqdm

def train_diff_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=500, patience=10):
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
            torch.save(model.state_dict(), "best_diff_model.pth")
            best_val_r2 = val_r2
            patience_counter = 0
            print("Validation improved. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
            
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

