from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


ROOT = Path("results/integration_order_finetuning_20261113")

EXPERIMENTS = {
    "hard_only": ROOT / "finetune_hard_only",
    "hard_replay": ROOT / "finetune_hard_replay",
    "hard_margin": ROOT / "finetune_hard_margin",
}

TS = [50, 100, 250]

OUTDIR = ROOT / "analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. JSON 안에서 Stage1 성능 찾기
# =========================================================

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def find_metric_dict(obj):
    """
    JSON 구조가 조금 달라도 stage1_size가 들어 있는 dict를 재귀적으로 탐색.
    """
    if isinstance(obj, dict):
        if "stage1_size" in obj:
            return obj

        for v in obj.values():
            found = find_metric_dict(v)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for v in obj:
            found = find_metric_dict(v)
            if found is not None:
                return found

    return None


def find_metrics(exp_path, T):
    candidates = []

    for f in exp_path.rglob("*.json"):
        text = str(f).lower()

        if f"t{T}" in text or f"/{T}/" in text or f"_{T}_" in text:
            obj = load_json(f)

            if obj is None:
                continue

            m = find_metric_dict(obj)

            if m is not None:
                candidates.append((f, m))

    if not candidates:
        return None, None

    # metrics / evaluation / summary 이름이 들어간 파일 우선
    candidates.sort(
        key=lambda x: (
            0 if any(k in x[0].name.lower()
                     for k in ["metric", "eval", "summary", "result"])
            else 1,
            len(str(x[0]))
        )
    )

    return candidates[0]


rows = []

print("\n" + "=" * 90)
print("FINE-TUNING PERFORMANCE")
print("=" * 90)

for exp_name, exp_path in EXPERIMENTS.items():

    for T in TS:

        f, m = find_metrics(exp_path, T)

        if m is None:
            print(f"[WARN] metrics 못 찾음: {exp_name}, T={T}")
            continue

        rho_power = m.get("rho_power", {})

        def rp(r):
            # JSON key가 "0.99" 또는 float처럼 저장된 경우 모두 대응
            for key in [str(r), str(float(r))]:
                if key in rho_power:
                    return 100 * float(rho_power[key])
            return np.nan

        rows.append({
            "experiment": exp_name,
            "T": T,
            "size_%": 100 * float(m.get("stage1_size", np.nan)),
            "overall_power_%": 100 * float(
                m.get("stage1_I0_power", np.nan)
            ),
            "rho_.2": rp(.2),
            "rho_.5": rp(.5),
            "rho_.9": rp(.9),
            "rho_.95": rp(.95),
            "rho_.99": rp(.99),
            "metric_file": str(f),
        })


df = pd.DataFrame(rows)

if len(df) > 0:

    display_cols = [
        "experiment",
        "T",
        "size_%",
        "overall_power_%",
        "rho_.2",
        "rho_.5",
        "rho_.9",
        "rho_.95",
        "rho_.99",
    ]

    print(
        df[display_cols]
        .sort_values(["T", "experiment"])
        .round(3)
        .to_string(index=False)
    )

    outfile = OUTDIR / "performance_summary.csv"
    df.to_csv(outfile, index=False)

    print(f"\n저장: {outfile}")

else:
    print("성능 결과를 찾지 못했습니다.")


# =========================================================
# 2. Encoder latent 찾기
# =========================================================

LATENT_KEYS = [
    "latent",
    "latents",
    "encoder_latent",
    "encoder_latents",
    "hidden",
    "hidden_state",
    "hidden_states",
    "embedding",
    "embeddings",
    "z",
    "h",
]

D_KEYS = [
    "d",
    "order",
    "orders",
    "integration_order",
    "true_order",
    "label",
    "labels",
]

RHO_KEYS = [
    "rho",
    "rhos",
]


def pick_key(data, keys):
    for k in keys:
        if k in data:
            return data[k]
    return None


def read_npz(path):
    try:
        a = np.load(path, allow_pickle=True)

        z = pick_key(a, LATENT_KEYS)
        d = pick_key(a, D_KEYS)
        rho = pick_key(a, RHO_KEYS)

        if z is None or d is None or rho is None:
            return None

        z = np.asarray(z)
        d = np.asarray(d).reshape(-1)
        rho = np.asarray(rho).reshape(-1)

        if (
            z.ndim == 2
            and z.shape[0] == len(d)
            and z.shape[0] == len(rho)
        ):
            return z, d, rho

    except Exception:
        pass

    return None


def read_csv_latent(path):
    try:
        x = pd.read_csv(path)

        dcol = next(
            (k for k in D_KEYS if k in x.columns),
            None
        )
        rcol = next(
            (k for k in RHO_KEYS if k in x.columns),
            None
        )

        latent_cols = [
            c for c in x.columns
            if (
                c.startswith("h_")
                or c.startswith("latent_")
                or c.startswith("z_")
                or c.startswith("hidden_")
            )
        ]

        if dcol and rcol and len(latent_cols) >= 2:
            return (
                x[latent_cols].to_numpy(),
                x[dcol].to_numpy(),
                x[rcol].to_numpy(),
            )

    except Exception:
        pass

    return None


def find_latent(exp_path, T):

    candidates = []

    for f in exp_path.rglob("*"):

        if not f.is_file():
            continue

        text = str(f).lower()

        if not (
            f"t{T}" in text
            or f"/{T}/" in text
            or f"_{T}_" in text
        ):
            continue

        if "latent" in text or "embedding" in text or "hidden" in text:
            candidates.insert(0, f)
        elif f.suffix.lower() in [".npz", ".csv"]:
            candidates.append(f)

    for f in candidates:

        result = None

        if f.suffix.lower() == ".npz":
            result = read_npz(f)

        elif f.suffix.lower() == ".csv":
            result = read_csv_latent(f)

        if result is not None:
            return f, result

    return None, None


# =========================================================
# 3. PCA
# =========================================================

print("\n" + "=" * 90)
print("STAGE1 ENCODER LATENT PCA")
print("=" * 90)

for exp_name, exp_path in EXPERIMENTS.items():

    for T in TS:

        f, result = find_latent(exp_path, T)

        if result is None:
            print(f"[WARN] latent 못 찾음: {exp_name}, T={T}")
            continue

        z, d, rho = result

        print(
            f"{exp_name:12s} T={T:3d} "
            f"shape={z.shape} | {f}"
        )

        # Stage1 비교에는 I(0), I(1)만 사용
        keep = np.isin(d, [0, 1])

        z_plot = z[keep]
        d_plot = d[keep]
        rho_plot = rho[keep]

        pca = PCA(n_components=2)
        pc = pca.fit_transform(z_plot)

        explained = pca.explained_variance_ratio_.sum()

        fig, ax = plt.subplots(figsize=(9, 7))

        # I(0): rho별 표시
        stationary_rhos = sorted(
            np.unique(rho_plot[d_plot == 0])
        )

        for r in stationary_rhos:

            mask = (
                (d_plot == 0)
                & np.isclose(rho_plot, r)
            )

            ax.scatter(
                pc[mask, 0],
                pc[mask, 1],
                s=8,
                alpha=0.25,
                label=f"I(0), rho={r:g}",
            )

        # I(1)
        mask = d_plot == 1

        ax.scatter(
            pc[mask, 0],
            pc[mask, 1],
            s=9,
            alpha=0.30,
            marker="x",
            label="I(1), rho=1",
        )

        ax.set_title(
            f"{exp_name} | T={T}\n"
            f"Stage1 encoder latent PCA "
            f"(PC1+PC2={explained:.3f})"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.legend(
            fontsize=8,
            markerscale=2,
        )

        fig.tight_layout()

        outfile = (
            OUTDIR
            / f"{exp_name}_T{T}_stage1_encoder_PCA.png"
        )

        fig.savefig(
            outfile,
            dpi=180,
        )

        plt.close(fig)


print("\nPCA 저장 위치:")
print(OUTDIR)


# =========================================================
# 4. centroid distance: 각 rho ↔ I(1)
# =========================================================

centroid_rows = []

for exp_name, exp_path in EXPERIMENTS.items():

    for T in TS:

        f, result = find_latent(exp_path, T)

        if result is None:
            continue

        z, d, rho = result

        # 원 latent 기준 거리
        unit_mask = d == 1

        if not unit_mask.any():
            continue

        unit_centroid = z[unit_mask].mean(axis=0)

        for r in sorted(np.unique(rho[d == 0])):

            mask = (
                (d == 0)
                & np.isclose(rho, r)
            )

            centroid = z[mask].mean(axis=0)

            distance = np.linalg.norm(
                centroid - unit_centroid
            )

            centroid_rows.append({
                "experiment": exp_name,
                "T": T,
                "rho": float(r),
                "distance_to_I1_centroid": float(distance),
            })


cdf = pd.DataFrame(centroid_rows)

if len(cdf) > 0:

    cfile = OUTDIR / "centroid_distance_to_I1.csv"

    cdf.to_csv(
        cfile,
        index=False,
    )

    print("\nCentroid distance:")
    print(
        cdf
        .sort_values(["T", "experiment", "rho"])
        .round(4)
        .to_string(index=False)
    )

    print(f"\n저장: {cfile}")