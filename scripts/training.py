"""Training loop with early stopping and W&B logging (EfficientNet / similar classifiers)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def train_model(
    model: nn.Module,
    train_dl,
    val_dl,
    epochs: int,
    lr: float,
    run_name: str,
    device: torch.device,
    wandb_project: str,
    patience: int = 8,
):
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config={"epochs": epochs, "lr": lr},
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1, best_state, no_improve = 0, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_dl, desc=f"Ep {epoch + 1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                labels.extend(yb.numpy())

        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_dl),
                "val_f1_macro": f1,
                "val_accuracy": acc,
            }
        )
        print(
            f"Ep {epoch + 1:02d} | Loss: {total_loss / len(train_dl):.4f} | "
            f"F1: {f1:.4f} | Acc: {acc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    wandb.log({"best_val_f1": best_f1})
    run.finish()
    return model, best_f1
