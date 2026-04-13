import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.data import MULTICLASS_LABELS, make_multiclass_dataset, make_tensordataset
from src.evaluate import evaluate_multiclass_pipeline
from src.model import LSTMClassifier
from src.train import train_model
from src.utils import config_to_jsonable, ensure_dir, load_config, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    run_name = args.run_name or f"multiclass_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg["output"]["root_dir"]) / run_name)
    save_json(config_to_jsonable(cfg), run_dir / "config_snapshot.json")

    dcfg = cfg["multiclass_data"]
    train_raw = make_multiclass_dataset(dcfg["train"]["ur2"], dcfg["train"]["ur1"], dcfg["train"]["ur0"], cfg["data"]["seq_len"], seed=cfg["seed"] + 21)
    val_raw = make_multiclass_dataset(dcfg["val"]["ur2"], dcfg["val"]["ur1"], dcfg["val"]["ur0"], cfg["data"]["seq_len"], seed=cfg["seed"] + 22)
    test_raw = make_multiclass_dataset(dcfg["test"]["ur2"], dcfg["test"]["ur1"], dcfg["test"]["ur0"], cfg["data"]["seq_len"], seed=cfg["seed"] + 23)

    train_ds = make_tensordataset(train_raw)
    val_ds = make_tensordataset(val_raw)
    test_ds = make_tensordataset(test_raw)

    model = LSTMClassifier(input_size=1, hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], num_classes=3)
    device = "cuda" if torch.cuda.is_available() and cfg["training"].get("use_cuda", False) else "cpu"

    history, ckpt_path = train_model(model, train_ds, val_ds, cfg, run_dir, device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    evaluation = evaluate_multiclass_pipeline(
        model,
        test_ds,
        test_raw["x"],
        test_raw["y"],
        alphas=cfg["adf"]["alphas"],
        run_dir=run_dir,
        device=device,
    )

    summary = {
        "label_mapping": MULTICLASS_LABELS,
        "history_summary": {"best_val_loss": history["best_val_loss"], "best_epoch": history["best_epoch"]},
        "evaluation": evaluation,
    }
    save_json(summary, run_dir / "summary.json")
    print(f"Saved multiclass run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
