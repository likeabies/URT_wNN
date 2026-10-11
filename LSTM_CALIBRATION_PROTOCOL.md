# LSTM extension: fixed-200 two-stage calibration

Based on ANN fixed-200 main commit `8c23ecf0faa51ce13db86b527199a57d10d6cdf9`
(PR #5). The ANN workflows and their configuration files are unchanged.

`configs/lstm_fixed200_w2_calibration.json` inherits the existing ANN fixed-200
configuration and its primary DGP. The only classifier change is a distinct
`Paper2019LSTMClassifier`: input `(batch,T)` is viewed as `(batch,T,1)`, one
LSTM layer with hidden size 30, last hidden state, and a two-logit linear head.
There is no dropout. The existing ANN sample-mean WCE is reused with class 0
unit root / w1=1 and class 1 stationary / w2. The rejection rule is unchanged.

Data seed **20191007**, its T-specific SeedSequence derivation, the complete
rho/beta grids, ARMA-innovation DGP, normalization, and sampling counts come
directly from the ANN `dataset` function. There are 120,000 training samples
per T: 5,000 unit-root samples per beta and 1,000 samples per stationary cell.
Each of the three datasets is created once and reused across every weight and
model seed in both stages. There is no validation set or independent test.

Every production fit uses MPS, Adam, lr=.001, batch=256, zero weight decay,
training shuffle, and **exactly 200 epochs**. Model seeds are [7,17,27].
There is no early stopping, convergence/size/power gate, optimizer sensitivity,
architecture search, polynomial fitting, ADF, or ANN retraining. Nonfinite
loss/parameters/metrics, corrupt checkpoint/data, and runtime failures raise an
error and mark the manifest failed. Unsupported MPS operations are not hidden
by a CPU fallback at launch.

Coarse: T=[50,100,250], w2=[.20,.25,.30,.35,.40,.45,.50,.55], three seeds:
**72 fits**. The unchanged ANN `local_grid` selects an observed adjacent
interval near mean training Type I error .05, with its original crossing and
tie-break rules. Five equally spaced fine points include both endpoints.
Eighteen endpoint results are reused, giving **27 new fine fits** and 99 fits
total. No user interaction separates the two stages. If no crossing exists,
fine still runs within the selected local interval, and the manifest/candidate
flags `target_may_be_outside_coarse_range`; no range expansion is performed.

Output: `results/lstm_paper2019_extension/w2_calibration/`. Refuse an existing
directory instead of overwriting results. Save per-epoch loss, accuracy, timing,
final checkpoint and training evaluation; aggregate into `coarse_summary.csv`,
`fine_summary.csv`, `fine_plan.json`, `fine_records.json`,
`calibration_candidates.json`, and `run_manifest.json`. Candidate statistics
reuse ANN mean/sample-SD and per-seed/beta summaries. Candidates remain
provisional. No final independent test is run.

W&B project: `URT_wNN`; online group:
`lstm_paper2019_extension_w2_calibration`. Config includes model, dimensions,
T, weights, seeds, dataset fingerprint, epochs, and training settings. Log
epoch metrics, final training Type I error, aggregate power, beta-specific size,
all rho-beta rejection rates, and runtime. Generated outputs are Git-ignored.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.run_lstm_fixed200_calibration --dry-run
.venv/bin/python -u -m scripts.run_lstm_fixed200_calibration --smoke
```

Smoke performs only three tiny three-epoch LSTM fits (one batch of 256 per T),
checks checkpoint reload and MPS forward/backward/evaluation, and logs online
under a separate timestamped smoke group. The two post-warmup batch timings
estimate full-data 200-epoch × 99-fit runtime; results are saved in
`runtime_estimate.json`. This short extrapolation is not a time guarantee and
does not reduce the production protocol to fit a runtime budget.

After merge, launch from latest main using the measured estimate:

```bash
mkdir -p logs
WANDB_MODE=online PYTORCH_ENABLE_MPS_FALLBACK=0 nohup caffeinate -is \
  .venv/bin/python -u -m scripts.run_lstm_fixed200_calibration \
  --runtime-estimate results/lstm_paper2019_extension/smoke_TIMESTAMP/runtime_estimate.json \
  > logs/lstm_fixed200_calibration.log 2>&1 < /dev/null &
```

`nohup` plus `caffeinate` keeps the process independent of the interactive
session and prevents idle sleep. A shutdown/reboot still stops the process;
the runner does not implement resume. The manifest records the launched Git SHA.
