from tqdm import tqdm
import torch
from utils.evaluation import evaluate_model

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, patience=5):
    model.to(device)
    best_r2 = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ssr, total_tss, total_samples = 0, 0, 0, 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, econ_features, targets, _, _ in tqdm_loader:
            images = images.to(device)
            econ_features = econ_features.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images, econ_features)
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

        # === RMSFE on training set ===
        _, _, train_rmsfe, *_ = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_r2, val_rmsfe, *_ = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # === Print metrics ===
        print(f"[Epoch {epoch+1}] "
              f"Train Loss={train_loss:.4f}, R²={train_r2:.4f}, RMSFE={train_rmsfe:.4f} | "
              f"Val Loss={val_loss:.4f}, R²={val_r2:.4f}, RMSFE={val_rmsfe:.4f}")

        # === Early stopping check ===
        if val_r2 > best_r2:
            print(f"Validation R² improved from {best_r2:.4f} to {val_r2:.4f}. Saving model.")
            best_r2 = val_r2
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation R². Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # === Final Training Set Evaluation ===
    final_train_loss, final_train_r2, final_train_rmsfe, *_ = evaluate_model(model, train_loader, criterion, device)
    print("\n=== Final Training Set Evaluation ===")
    print(f"Loss: {final_train_loss:.4f}, R²: {final_train_r2:.4f}, RMSFE: {final_train_rmsfe:.4f}")

    return model

def AFF_train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=50, patience=5):
    model.to(device)
    best_r2 = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ssr, total_tss, total_samples = 0, 0, 0, 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, econ_features, targets, _, _ in tqdm_loader:
            images = images.to(device)
            econ_features = econ_features.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images, econ_features)
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

        # === RMSFE on training set ===
        _, _, train_rmsfe, *_ = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_r2, val_rmsfe, *_ = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # === Print metrics ===
        print(f"[Epoch {epoch+1}] "
              f"Train Loss={train_loss:.4f}, R²={train_r2:.4f}, RMSFE={train_rmsfe:.4f} | "
              f"Val Loss={val_loss:.4f}, R²={val_r2:.4f}, RMSFE={val_rmsfe:.4f}")

        # === Early stopping check ===
        if val_r2 > best_r2:
            print(f"Validation R² improved from {best_r2:.4f} to {val_r2:.4f}. Saving model.")
            best_r2 = val_r2
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'AFF_best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation R². Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # === Final Training Set Evaluation ===
    final_train_loss, final_train_r2, final_train_rmsfe, *_ = evaluate_model(model, train_loader, criterion, device)
    fianl_val_loss, final_val_r2, final_val_rmsfe, *_ = evaluate_model(model, val_loader, criterion, device)
    print("\n=== Final Training Set Evaluation ===")
    print(f"Loss: {final_train_loss:.4f}, R²: {final_train_r2:.4f}, RMSFE: {final_train_rmsfe:.4f},val_loss: {fianl_val_loss:.4f}, val_r2: {final_val_r2:.4f}, val_rmsfe: {final_val_rmsfe:.4f}")

    return model
