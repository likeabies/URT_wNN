from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adf_utils import evaluate_adf_binary, evaluate_adf_multiclass
from .utils import save_json


def predict_model(model, ds, batch_size=1000, device="cpu"):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def binary_metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    ur_mask = y_true == 0
    st_mask = y_true == 1
    empirical_size = float((y_pred[ur_mask] == 1).mean()) if ur_mask.any() else 0.0
    empirical_power = float((y_pred[st_mask] == 1).mean()) if st_mask.any() else 0.0
    return {"accuracy": acc, "empirical_size": empirical_size, "empirical_power": empirical_power}


def multiclass_metrics(y_true, y_pred, n_classes=3):
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    class_acc = {}
    for c in range(n_classes):
        m = y_true == c
        class_acc[f"class_{c}"] = float((y_pred[m] == y_true[m]).mean()) if m.any() else 0.0
    return {"accuracy": float((y_true == y_pred).mean()), "class_accuracy": class_acc, "confusion_matrix": conf.tolist()}


def evaluate_binary_pipeline(model, test_ds, test_x, test_y, alphas, run_dir: Path, device="cpu"):
    y_pred, y_true = predict_model(model, test_ds, device=device)
    lstm = binary_metrics(y_true, y_pred)
    adf = evaluate_adf_binary(test_x, test_y, alphas=alphas, regression="n")
    out = {"lstm": lstm, "adf": adf}
    save_json(out, Path(run_dir) / "evaluation_binary.json")
    return out


def evaluate_multiclass_pipeline(model, test_ds, test_x, test_y, alphas, run_dir: Path, device="cpu"):
    y_pred, y_true = predict_model(model, test_ds, device=device)
    lstm = multiclass_metrics(y_true, y_pred)
    adf = evaluate_adf_multiclass(test_x, test_y, alphas=alphas, regression="n")
    out = {"lstm": lstm, "adf": adf}
    save_json(out, Path(run_dir) / "evaluation_multiclass.json")
    return out


def evaluate_non_theoretical_table(model, datasets: Dict[str, Dict], alphas, device="cpu"):
    rows = []
    for p_key, pack in datasets.items():
        ds = pack["ds"]
        x, y = pack["x"], pack["y"]
        pred, y_true = predict_model(model, ds, device=device)
        lstm_acc = float((pred == y_true).mean())
        adf = evaluate_adf_binary(x, y, alphas=alphas, regression="n")
        rows.append({
            "p": p_key,
            "composition": pack["composition"],
            "lstm_accuracy": lstm_acc,
            "adf_accuracy": {k: v["accuracy"] for k, v in adf.items()},
        })
    return rows
