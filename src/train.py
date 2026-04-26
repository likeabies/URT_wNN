from pathlib import Path
from time import perf_counter
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import ensure_dir, save_json


def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_n += int(xb.size(0))

    return total_loss / total_n, total_correct / total_n


def validate_one_epoch(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_n += int(xb.size(0))

    return total_loss / total_n, total_correct / total_n


def train_model(model, train_ds, val_ds, config: Dict, run_dir: Path, device: str = "cpu"):
    total_start = perf_counter()

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
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_time_sec": [],
        "val_time_sec": [],
        "epoch_time_sec": [],
        "best_val_loss": None,
        "best_epoch": None,
        "total_time_sec": None,
    }

    best_val = float("inf")
    wait = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = perf_counter()

        train_start = perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        train_time = perf_counter() - train_start

        val_start = perf_counter()
        va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device=device)
        val_time = perf_counter() - val_start

        epoch_time = perf_counter() - epoch_start

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_time_sec"].append(train_time)
        history["val_time_sec"].append(val_time)
        history["epoch_time_sec"].append(epoch_time)
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "train_time_sec": train_time,
                "val_time_sec": val_time,
                "epoch_time_sec": epoch_time,
            }
        )

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

    history["total_time_sec"] = perf_counter() - total_start

    save_json(history, run_dir / "history.json")
    return history, ckpt_path
