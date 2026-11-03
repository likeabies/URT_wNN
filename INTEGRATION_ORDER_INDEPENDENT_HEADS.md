# Integration-order LSTM independent-head ablation

This experiment removes only the direct Stage1-to-Stage2 connection from the
fixed-cycle experiment merged into main at `9cb02bd` (PR #12). The original
model, runner, and config remain unchanged.

`IndependentHeadIntegrationOrderLSTM` uses one shared `nn.LSTM(1, 30, 1)`:
`Head1(h_level)` and `Head2(h_diff)` are separate `nn.Linear(30, 1)` heads.
Stage2 consumes neither p1 nor logit1. The loss is still balanced `L1 + L2`:
L2 reaches the shared encoder and Head2, but cannot reach Head1 parameters.
Shared-encoder gradient interference is therefore still possible.

The separate runner mirrors the original runner. Only the model import,
model construction, config path, and W&B run identity differ. Its config copies
all scientific settings from main; only W&B group and architecture metadata
are added/changed. It uses standard `torch.optim.Adam`, no clipping, no Safe
Adam, no early stopping, and no checkpoint resume. Each T starts from seed 7
and trains 200 epochs, unless a numerical failure terminates that T. The
original failure handling preserves logs and continues to the next T.

From the repository root, using the pinned requirements:

```sh
PYTHONPATH=. python -u scripts/run_integration_order_independent_cycle.py \
  --output results/integration_order_lstm_independent_fixed_cycle_20261103
```

The destination must not already exist. MPS is required, matching the original
fixed-cycle runner. Each run saves the original output schema: all epoch
checkpoints/history, cohorts and fingerprints, calibration thresholds, test
probabilities/predictions, neural/ADF results, W&B URL, and failure logs.

After all three runs finish, audit/export the comparison:

```sh
PYTHONPATH=. python scripts/compare_integration_order_heads.py \
  --baseline /path/to/integration_order_lstm_fixed_cycle_20261029 \
  --ablation results/integration_order_lstm_independent_fixed_cycle_20261103 \
  --output experiment_reports/integration_order_independent_heads_20261103
```

The comparison checks configuration equality, matching cohort fingerprints,
200 completed epochs, thresholds recomputed from calibration only, neural
metrics recomputed from saved test probabilities, and identical ADF predictions.
Failed runs retain their failure evidence and receive no invented test metrics.
It exports a joint table (CSV/JSON/Markdown), confusion matrices, recall and
precision, I(1)-to-I(0) changes, ADF metrics, W&B links, and compact run evidence.
Large raw datasets/checkpoints stay in the ignored local results directory.

Validation:

```sh
python -m unittest discover -s tests -p test_integration_order_independent_lstm.py -v
```
