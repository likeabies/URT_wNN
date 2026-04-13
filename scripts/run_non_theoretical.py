import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.data import make_non_theoretical_binary_testset, make_tensordataset
from src.evaluate import evaluate_non_theoretical_table
from src.model import LSTMClassifier
from src.utils import ensure_dir, load_config, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or f"non_theoretical_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(cfg["output"]["root_dir"]) / run_name)

    model = LSTMClassifier(input_size=1, hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], num_classes=2)
    device = "cuda" if torch.cuda.is_available() and cfg["training"].get("use_cuda", False) else "cpu"
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    datasets = {}
    for i, p in enumerate(cfg["non_theoretical"]["p_values"]):
        raw = make_non_theoretical_binary_testset(
            total=cfg["non_theoretical"]["total"],
            p_unit_root=float(p),
            length=cfg["data"]["seq_len"],
            seed=cfg["seed"] + 100 + i,
        )
        datasets[str(p)] = {
            "ds": make_tensordataset(raw),
            "x": raw["x"],
            "y": raw["y"],
            "composition": raw["composition"],
        }

    rows = evaluate_non_theoretical_table(model, datasets, cfg["adf"]["alphas"], device=device)
    save_json(rows, run_dir / "non_theoretical_results.json")
    print(f"Saved non-theoretical results to: {run_dir}")


if __name__ == "__main__":
    main()
