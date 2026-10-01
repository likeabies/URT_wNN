from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

BINARY_LABELS = {"unit_root": 0, "stationary": 1}  # one-hot: ur=(1,0), st=(0,1)
MULTICLASS_LABELS = {"ur2": 0, "ur1": 1, "ur0": 2}  # one-hot order requested: ur2, ur1, ur0


@dataclass
class SeriesSample:
    values: np.ndarray
    phi1: float
    phi2: float


def is_stationary_ar2(phi1: float, phi2: float) -> bool:
    return (-1 < phi2 < 1) and (phi1 + phi2 < 1) and (phi2 - phi1 < 1)


def sample_stationary_ar2_coeffs(rng: np.random.Generator) -> Tuple[float, float]:
    while True:
        phi1, phi2 = rng.uniform(-0.9, 0.9, size=2)
        if is_stationary_ar2(phi1, phi2):
            return float(phi1), float(phi2)


def generate_stationary_series(length: int, rng: np.random.Generator) -> SeriesSample:
    phi1, phi2 = sample_stationary_ar2_coeffs(rng)
    eps = rng.normal(0.0, 1.0, size=length)
    z = np.zeros(length, dtype=np.float32)
    for t in range(length):
        z_tm1 = z[t - 1] if t - 1 >= 0 else 0.0
        z_tm2 = z[t - 2] if t - 2 >= 0 else 0.0
        z[t] = phi1 * z_tm1 + phi2 * z_tm2 + eps[t]
    return SeriesSample(values=z, phi1=phi1, phi2=phi2)


def integrate_series(series: np.ndarray, order: int) -> np.ndarray:
    out = np.array(series, dtype=np.float32)
    for _ in range(order):
        out = np.cumsum(out, dtype=np.float32)
    return out


def _to_tensors(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32)  # (N, L, 1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def _sanity_counts(y: np.ndarray, expected: Dict[int, int]) -> None:
    for cls, n in expected.items():
        actual = int((y == cls).sum())
        if actual != n:
            raise ValueError(f"Class {cls} count mismatch: expected {n}, got {actual}")


def make_binary_dataset(n_stationary: int, n_ur1: int, n_ur2: int, length: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = n_stationary + n_ur1 + n_ur2
    x = np.zeros((n_total, length), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)

    idx = 0
    for _ in range(n_stationary):
        s = generate_stationary_series(length, rng)
        if not is_stationary_ar2(s.phi1, s.phi2):
            raise ValueError("AR(2) stationarity check failed")
        x[idx] = s.values
        y[idx] = BINARY_LABELS["stationary"]
        idx += 1

    for _ in range(n_ur1):
        s = generate_stationary_series(length, rng)
        x[idx] = integrate_series(s.values, order=1)
        y[idx] = BINARY_LABELS["unit_root"]
        idx += 1

    for _ in range(n_ur2):
        s = generate_stationary_series(length, rng)
        x[idx] = integrate_series(s.values, order=2)
        y[idx] = BINARY_LABELS["unit_root"]
        idx += 1

    perm = rng.permutation(n_total)
    x, y = x[perm], y[perm]
    _sanity_counts(y, {
        BINARY_LABELS["stationary"]: n_stationary,
        BINARY_LABELS["unit_root"]: n_ur1 + n_ur2,
    })
    return {"x": x, "y": y}


def make_multiclass_dataset(n_ur2: int, n_ur1: int, n_ur0: int, length: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = n_ur2 + n_ur1 + n_ur0
    x = np.zeros((n_total, length), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)

    idx = 0
    for _ in range(n_ur2):
        s = generate_stationary_series(length, rng)
        x[idx] = integrate_series(s.values, order=2)
        y[idx] = MULTICLASS_LABELS["ur2"]
        idx += 1
    for _ in range(n_ur1):
        s = generate_stationary_series(length, rng)
        x[idx] = integrate_series(s.values, order=1)
        y[idx] = MULTICLASS_LABELS["ur1"]
        idx += 1
    for _ in range(n_ur0):
        s = generate_stationary_series(length, rng)
        x[idx] = s.values
        y[idx] = MULTICLASS_LABELS["ur0"]
        idx += 1

    perm = rng.permutation(n_total)
    x, y = x[perm], y[perm]
    _sanity_counts(y, {
        MULTICLASS_LABELS["ur2"]: n_ur2,
        MULTICLASS_LABELS["ur1"]: n_ur1,
        MULTICLASS_LABELS["ur0"]: n_ur0,
    })
    return {"x": x, "y": y}


def _balanced_split_count(total: int, k: int) -> List[int]:
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def make_non_theoretical_binary_testset(total: int, p_unit_root: float, length: int, seed: int):
    rng = np.random.default_rng(seed)
    n_unit_root = int(round(total * p_unit_root))
    n_stationary = total - n_unit_root

    ur1_count, ur2_count = _balanced_split_count(n_unit_root, 2)
    ur1_m = _balanced_split_count(ur1_count, 3)  # methods 13/14/15
    ur2_m = _balanced_split_count(ur2_count, 3)

    x = []
    y = []

    for _ in range(n_stationary):
        s = generate_stationary_series(length, rng)
        x.append(s.values)
        y.append(BINARY_LABELS["stationary"])

    def make_method_series(order: int, method_id: int):
        s1 = generate_stationary_series(length, rng).values
        ur = integrate_series(generate_stationary_series(length, rng).values, order=order)
        if method_id == 13:
            return ur
        if method_id == 14:
            out = np.concatenate([s1[:50], ur[50:]], axis=0)
            return out.astype(np.float32)
        out = np.concatenate([ur[:50], s1[50:]], axis=0)
        return out.astype(np.float32)

    for order, counts in [(1, ur1_m), (2, ur2_m)]:
        for method_id, c in zip((13, 14, 15), counts):
            for _ in range(c):
                x.append(make_method_series(order, method_id))
                y.append(BINARY_LABELS["unit_root"])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]

    composition = {
        "total": int(total),
        "stationary": int(n_stationary),
        "unit_root": int(n_unit_root),
        "ur1_total": int(ur1_count),
        "ur2_total": int(ur2_count),
        "ur1_method_13": int(ur1_m[0]),
        "ur1_method_14": int(ur1_m[1]),
        "ur1_method_15": int(ur1_m[2]),
        "ur2_method_13": int(ur2_m[0]),
        "ur2_method_14": int(ur2_m[1]),
        "ur2_method_15": int(ur2_m[2]),
    }

    if composition["stationary"] + composition["unit_root"] != total:
        raise ValueError("Non-theoretical test composition mismatch")

    return {"x": x, "y": y, "composition": composition}


def make_tensordataset(data: Dict[str, np.ndarray]) -> TensorDataset:
    return _to_tensors(data["x"], data["y"])


def normalize_ann_series(values: np.ndarray) -> np.ndarray:
    """Center and scale each series independently; a constant series stays zero."""
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.max(np.abs(centered), axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale != 0)


def make_ann_dataset(length: int, rhos, betas, n_unit_root: int, n_stationary: int, seed: int):
    """Counts are per (rho, beta). Return normalized inputs and aligned labels/parameters.

    General model: Y_t = rho*Y_(t-1) + Z_t,
    Z_t = alpha*Z_(t-1) + beta*e_(t-1) + e_t.
    Simulation uses alpha=0, Y_0=0, and iid N(0,1) errors e_0,...,e_T.
    Vectorize across samples, retaining the exact recurrence over time.
    """
    rng = np.random.default_rng(seed)
    total = sum(n_unit_root if rho == 1 else n_stationary for rho in rhos) * len(betas)
    x = np.empty((total, length), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    rho_values = np.empty(total, dtype=np.float64)
    beta_values = np.empty(total, dtype=np.float64)
    offset = 0
    for rho in rhos:
        for beta in betas:
            n = n_unit_root if rho == 1 else n_stationary
            errors = rng.normal(size=(n, length + 1))
            innovations = beta * errors[:, :-1] + errors[:, 1:]
            values = np.empty((n, length), dtype=np.float64)
            previous = np.zeros(n)
            for t in range(length):
                previous = rho * previous + innovations[:, t]
                values[:, t] = previous
            region = slice(offset, offset + n)
            x[region] = normalize_ann_series(values)
            y[region] = BINARY_LABELS["unit_root" if rho == 1 else "stationary"]
            rho_values[region] = rho
            beta_values[region] = beta
            offset += n
    order = rng.permutation(total)
    return {"x": x[order], "y": y[order], "rho": rho_values[order], "beta": beta_values[order]}
