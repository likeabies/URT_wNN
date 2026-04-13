from typing import Dict, List

import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_is_stationary(series: np.ndarray, alpha: float, regression: str = "n") -> bool:
    stat, _, _, _, crit_vals, _ = adfuller(series, regression=regression, autolag="AIC")
    threshold = crit_vals[f"{int(alpha*100)}%"]
    return stat < threshold


def adf_binary_predict(series: np.ndarray, alpha: float, regression: str = "n") -> int:
    # binary label mapping: unit_root=0, stationary=1
    return 1 if adf_is_stationary(series, alpha, regression) else 0


def adf_multiclass_predict(series: np.ndarray, alpha: float, regression: str = "n") -> int:
    # ur2=0, ur1=1, ur0=2
    if adf_is_stationary(series, alpha, regression):
        return 2
    diff1 = np.diff(series)
    if adf_is_stationary(diff1, alpha, regression):
        return 1
    return 0


def evaluate_adf_binary(x: np.ndarray, y: np.ndarray, alphas: List[float], regression: str = "n") -> Dict[str, Dict[str, float]]:
    out = {}
    for alpha in alphas:
        preds = np.array([adf_binary_predict(s, alpha, regression) for s in x], dtype=np.int64)
        acc = float((preds == y).mean())
        ur_mask = y == 0
        st_mask = y == 1
        empirical_size = float((preds[ur_mask] == 1).mean()) if ur_mask.any() else 0.0
        empirical_power = float((preds[st_mask] == 1).mean()) if st_mask.any() else 0.0
        out[f"alpha_{alpha}"] = {
            "accuracy": acc,
            "empirical_size": empirical_size,
            "empirical_power": empirical_power,
        }
    return out


def evaluate_adf_multiclass(x: np.ndarray, y: np.ndarray, alphas: List[float], regression: str = "n") -> Dict[str, Dict[str, float]]:
    out = {}
    for alpha in alphas:
        preds = np.array([adf_multiclass_predict(s, alpha, regression) for s in x], dtype=np.int64)
        class_acc = {}
        for cls in [0, 1, 2]:
            m = y == cls
            class_acc[f"class_{cls}"] = float((preds[m] == y[m]).mean()) if m.any() else 0.0
        out[f"alpha_{alpha}"] = class_acc
    return out
