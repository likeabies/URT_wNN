from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import ensure_dir, save_json


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_n += int(xb.size(0))

    return total_loss / total_n, total_correct / total_n


def train_model(model, train_ds, val_ds, config: Dict, run_dir: Path, device: str = "cpu"):
    run_dir = ensure_dir(run_dir)
    ckpt_path = run_dir / "best_model.pt"

    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    lr = float(config["training"].get("lr", 1e-3))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_val_loss": None,
        "best_epoch": None,
    }

    best_val = float("inf")
    wait = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < best_val:
            best_val = va_loss
            wait = 0
            history["best_val_loss"] = va_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1

        if wait >= patience:
            break

    save_json(history, run_dir / "history.json")
    return history, ckpt_path
