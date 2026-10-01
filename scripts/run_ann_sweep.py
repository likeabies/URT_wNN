"""Sequential ANN grid with shared data; --dry-run inspects without starting runs."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np

from scripts.run_ann import run_ann
from src.data import make_ann_dataset
from src.utils import load_config, save_json


def data_fingerprint(data):
    """Hash every input, label and DGP parameter in sample order."""
    digest = hashlib.sha256()
    for key in ("x", "y", "rho", "beta"):
        values = np.ascontiguousarray(data[key])
        digest.update(f"{key}:{values.dtype}:{values.shape}".encode())
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def prepare_data(base, data_seed):
    """Generate once per (T, data_seed), independent of model_seed and w2."""
    dcfg = deepcopy(base["data"])
    length = dcfg["seq_len"]
    streams = np.random.SeedSequence([data_seed, length]).spawn(4)
    seeds = [int(stream.generate_state(1)[0]) for stream in streams]
    if len(set(seeds)) != 4:
        raise ValueError("Derived split seeds collided; choose another data_seed")
    shared = {}
    for split, seed in zip(("train", "val", "test"), seeds):
        dcfg[split]["seed"] = seed
        shared[split] = make_ann_dataset(length, dcfg["rhos"], dcfg["betas"], **dcfg[split])

    # Preserve the entire base validation set, including its sample ordering.
    # Only append new, independently generated unit-root samples to balance it.
    extra_count = (len(dcfg["rhos"]) - 1) * dcfg["val"]["n_stationary"] - dcfg["val"]["n_unit_root"]
    extension_cfg = {"n_unit_root": extra_count, "seed": seeds[3]}
    extension = make_ann_dataset(length, [1], dcfg["betas"], n_stationary=0, **extension_cfg)
    balanced = {key: np.concatenate([shared["val"][key], extension[key]]) for key in shared["val"]}
    stationary = {key: value[shared["val"]["y"] == 1] for key, value in shared["val"].items()}
    stationary_balanced = {key: value[balanced["y"] == 1] for key, value in balanced.items()}
    for key in stationary:
        np.testing.assert_array_equal(stationary[key], stationary_balanced[key])
    assert (balanced["y"] == 0).sum() == (balanced["y"] == 1).sum()
    datasets = {
        "val_72k": shared,
        "val_120k": {"train": shared["train"], "val": balanced, "test": shared["test"]},
    }
    fingerprints = {
        design: {**{split: data_fingerprint(raw) for split, raw in pack.items()},
                 "stationary_val": data_fingerprint(stationary)}
        for design, pack in datasets.items()
    }
    return datasets, dcfg, extension_cfg, fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sweep = load_config(args.config)
    for key in ("T_values", "w2_values", "data_seeds", "model_seeds"):
        values = sweep[key]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be nonempty and contain no duplicates")
    if any(not math.isfinite(w) or w <= 0 for w in sweep["w2_values"]):
        raise ValueError("w2 values must be finite and positive")
    if any(not isinstance(s, int) or s < 0 for s in sweep["data_seeds"] + sweep["model_seeds"]):
        raise ValueError("Seeds must be nonnegative integers")

    designs = ("val_72k", "val_120k")
    bases = {}
    for length in sweep["T_values"]:
        base = load_config(sweep["base_configs"][length])
        if length not in (50, 100, 250) or base["data"]["seq_len"] != length:
            raise ValueError("Each T must match its base ANN config")
        if base["data"]["rhos"] != [1, 0.99, 0.95, 0.9, 0.5, 0.2]:
            raise ValueError("Use the existing six-rho ANN grid")
        val = base["data"]["val"]
        if val["n_unit_root"] != val["n_stationary"] or val["n_stationary"] <= 0:
            raise ValueError("Base validation must have equal positive counts per cell")
        if "training_overrides" in sweep:
            base["training"].update(sweep["training_overrides"])
        base["output"]["root_dir"] = sweep["output"]["root_dir"]
        bases[length] = base

    combinations = list(product(sweep["T_values"], sweep["data_seeds"], sweep["model_seeds"], designs, sweep["w2_values"]))
    prefix = args.run_prefix or f"ann_sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("run-prefix must be a directory-safe name")
    print(f"Sweep {prefix}: {len(combinations)} runs; w1=1; W&B project=URT_wNN", flush=True)
    for length, base in bases.items():
        dcfg = base["data"]
        count = lambda split: len(dcfg["betas"]) * (dcfg[split]["n_unit_root"] + 5 * dcfg[split]["n_stationary"])
        print(f"T={length}: train={count('train')}, val_72k={count('val')}, "
              f"val_120k={10 * len(dcfg['betas']) * dcfg['val']['n_stationary']}, "
              f"test={count('test')}; training={base['training']}", flush=True)
    jobs = []
    for length, data_seed, model_seed, design, w2 in combinations:
        name = f"{prefix}_T{length}_{design}_w2-{w2}_ds-{data_seed}_ms-{model_seed}"
        jobs.append((length, data_seed, model_seed, design, w2, name))
        print(name, flush=True)
    if args.dry_run:
        return

    root = Path(sweep["output"]["root_dir"])
    manifest_dir = root / prefix
    if manifest_dir.exists() or any((root / job[-1]).exists() for job in jobs):
        raise FileExistsError("Sweep prefix already exists; choose a new prefix to preserve previous results")
    manifest_dir.mkdir(parents=True)
    manifest = {"sweep_config": sweep, "group": prefix, "expected_runs": len(jobs), "runs": []}
    save_json(manifest, manifest_dir / "sweep_manifest.json")
    current_key = None
    for length, data_seed, model_seed, design, w2, name in jobs:
        key = (length, data_seed)
        if key != current_key:
            datasets, dcfg, extension, fingerprints = prepare_data(bases[length], data_seed)
            current_key = key
        cfg = deepcopy(bases[length])
        cfg.update({"T": length, "data_seed": data_seed, "model_seed": model_seed,
                    "seed": model_seed, "validation_design": design, "w2": float(w2),
                    "sweep_group": prefix, "data_fingerprints": fingerprints[design]})
        cfg["data"] = deepcopy(dcfg)
        cfg["loss"] = {"w1": 1.0, "w2": float(w2)}
        cfg["validation_extension"] = extension if design == "val_120k" else None
        cfg["sample_counts"] = {split: len(raw["y"]) for split, raw in datasets[design].items()}
        info = run_ann(cfg, name, prepared_data=datasets[design])
        manifest["runs"].append({**info, "T": length, "data_seed": data_seed,
                                 "model_seed": model_seed, "validation_design": design,
                                 "w2": float(w2), "data_fingerprints": fingerprints[design],
                                 "status": "finished"})
        save_json(manifest, manifest_dir / "sweep_manifest.json")
    print(f"Finished {len(jobs)} runs. Manifest: {manifest_dir / 'sweep_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
