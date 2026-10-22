import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time
from pathlib import Path
from src.config import Config


def compute_metrics(preds, targets):
    preds_binary = (preds > 0.5).int().view(-1)
    targets = targets.int().view(-1)
    acc = (preds_binary == targets).float().mean().item()
    f1 = f1_score(targets.cpu().numpy(), preds_binary.cpu().numpy())
    return acc, f1


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.detach())
            all_targets.append(labels.detach())

    avg_loss = total_loss / len(loader.dataset)

    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)
    acc, f1 = compute_metrics(preds_tensor, targets_tensor)

    return avg_loss, acc, f1


def run_training(model, train_loader, val_loader, model_name, writer=None):
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    best_val_f1 = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device
        )

        if writer:
            writer.add_scalar(f"{model_name}/train_loss", train_loss, epoch)
            writer.add_scalar(f"{model_name}/train_acc", train_acc, epoch)
            writer.add_scalar(f"{model_name}/train_f1", train_f1, epoch)
            writer.add_scalar(f"{model_name}/val_loss", val_loss, epoch)
            writer.add_scalar(f"{model_name}/val_acc", val_acc, epoch)
            writer.add_scalar(f"{model_name}/val_f1", val_f1, epoch)

        print(f"epoch {epoch + 1}/{Config.NUM_EPOCHS} - "
              f"train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f} | "
              f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            artifacts_dir = Path(Config.ARTIFACTS_DIR)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), artifacts_dir / f"{model_name}_best.pth")

    elapsed = time.time() - start_time

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "train_time": elapsed
    }


def test_model(model, test_loader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = Path(Config.ARTIFACTS_DIR)
    model_path = artifacts_dir / f"{model_name}_best.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, device)
    print(f"\ntest results for {model_name}:")
    print(f"  loss: {test_loss:.4f}")
    print(f"  accuracy: {test_acc:.4f}")
    print(f"  f1: {test_f1:.4f}")

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
