import argparse
import hashlib
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset

from src.data import BINARY_LABELS, make_ann_dataset
from src.evaluate import evaluate_ann
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy, train_model
from src.utils import ensure_dir, load_config, save_json, select_device, set_seed


def run_ann(cfg, run_name=None, prepared_data=None):
    """Run one experiment; a sweep may supply shared, already generated arrays."""
    dcfg = cfg["data"]
    length = int(dcfg["seq_len"])
    if length not in (50, 100, 250):
        raise ValueError("ANN sequence length must be 50, 100, or 250")
    if cfg["model"]["hidden_size"] != (20 if length == 50 else 50):
        raise ValueError("ANN hidden size must be 20 for T=50, otherwise 50")
    if cfg["model"]["dropout"] != 0 or cfg["training"]["optimizer"] != "Adam":
        raise ValueError("This reproduction uses Adam and no dropout")
    if len({dcfg[split]["seed"] for split in ("train", "val", "test")}) != 3:
        raise ValueError("Train, validation, and test seeds must be distinct")
    w1, w2 = float(cfg["loss"]["w1"]), float(cfg["loss"]["w2"])
    if not (0 < w1 < float("inf") and 0 < w2 < float("inf")):
        raise ValueError("Loss weights must be finite and positive")

    model_seed = int(cfg.get("model_seed", cfg["seed"]))
    set_seed(model_seed)
    device = select_device()
    run_name = run_name or f"ann_T{length}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg["output"]["root_dir"]) / run_name)
    save_json(cfg, run_dir / "config_snapshot.json")
    print(f"ANN T={length}; device={device}", flush=True)

    with wandb.init(project="URT_wNN", name=run_name, config=cfg, group=cfg.get("sweep_group")) as run:
        run.config.update({"actual_device": device})
        run.summary["device"] = device
        datasets = {}
        raw = dict(prepared_data) if prepared_data is not None else {}
        for split in ("train", "val"):
            if split not in raw:
                raw[split] = make_ann_dataset(
                    length, dcfg["rhos"], dcfg["betas"], **dcfg[split]
                )
                if split == "val" and cfg.get("validation_extension"):
                    extension = make_ann_dataset(
                        length, [1], dcfg["betas"], n_stationary=0,
                        **cfg["validation_extension"],
                    )
                    raw[split] = {key: np.concatenate([raw[split][key], extension[key]]) for key in raw[split]}
            datasets[split] = TensorDataset(
                torch.from_numpy(raw[split]["x"]), torch.from_numpy(raw[split]["y"])
            )
        # Initialization and training randomness depend only on model_seed, not
        # W&B setup, data generation, or the preceding run in a sweep.
        set_seed(model_seed)
        model = ANNClassifier(length, cfg["model"]["hidden_size"])
        identifiers = {}
        if "sweep_group" in cfg:
            identifiers = {key: cfg[key] for key in (
                "T", "validation_design", "w2", "data_seed", "model_seed", "data_fingerprints"
            )}
            identifiers["initial_model_sha256"] = hashlib.sha256(
                b"".join(p.detach().numpy().tobytes() for p in model.parameters())
            ).hexdigest()
            identifiers["training_rng_sha256"] = hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest()
            run.summary.update(identifiers)
        criterion = partial(ann_weighted_cross_entropy, w1=w1, w2=w2)
        history, checkpoint = train_model(
            model, datasets["train"], datasets["val"], cfg, run_dir,
            device=device, wandb_run=run, criterion=criterion,
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

        evaluations = {}
        for split in ("val", "test"):
            # Shared sweep test arrays are only used after model selection.
            if split == "test":
                if split not in raw:
                    raw[split] = make_ann_dataset(
                        length, dcfg["rhos"], dcfg["betas"], **dcfg[split]
                    )
                datasets[split] = TensorDataset(
                    torch.from_numpy(raw[split]["x"]), torch.from_numpy(raw[split]["y"])
                )
            result = evaluate_ann(
                model, datasets[split], raw[split], criterion,
                batch_size=cfg["training"]["batch_size"], device=device,
            )
            evaluations[split] = result
            save_json(result, run_dir / f"evaluation_{split}.json")
            run.log({split: {
                "type_i_error": result["overall"]["empirical_size"],
                "power": result["overall"]["empirical_power"],
                "overall": result["overall"],
                "rejection_rate": {
                    f"rho={row['rho']:g},beta={row['beta']:g}": row["rejection_rate"]
                    for row in result["by_parameter"]
                },
            }})
        save_json({
            **identifiers,
            "device": device,
            "label_mapping": BINARY_LABELS,
            "history_summary": {
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
            },
            "evaluation": evaluations,
        }, run_dir / "summary.json")
        print(f"Saved ANN run artifacts to: {run_dir}")
        run_info = {"name": run_name, "id": run.id, "url": run.url, "device": device}
    return run_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_ann(load_config(args.config), args.run_name)


if __name__ == "__main__":
    main()
