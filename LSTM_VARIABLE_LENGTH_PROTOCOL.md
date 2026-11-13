# Exact-length LSTM experiment families

The preserved pre-transition sources are under
analysis_outputs/exact_length_transition_20261020/source_before.
Historical results, diagnostics and checkpoints remain unchanged.

Random-T: scripts/run_lstm_variable_length.py uses
configs/lstm_variable_length_randomT_exact_batch.json. Lengths remain independent
DiscreteUniform{50,...,500}. This family is code/test only; no production launch
is authorized in this transition.

Grid50: scripts/run_lstm_multilength_grid50.py uses
configs/lstm_multilength_grid50_equalN.json. One shared LSTM handles all ten
lengths 50,100,...,500. Each T has 60000 training series: 30000 H0
(2500/beta) and 30000 stationary (500/rho-beta cell).
Calibration has 6000 H0/T (500/beta), 60000 total.
Independent test has 12000/T: 6000 H0 (500/beta) and 6000 stationary
(100/cell), 120000 total. The beta/rho grids, DGP and series-wise normalization
are unchanged. Seeds are train20191007, calibration20261019, test20261020.
Grid data uses a reproducible SeedSequence([split_seed,T]) child seed per T;
derived seeds and dataset fingerprints are recorded. Random-T preserves its
independent length/noise child streams and generation chunk size.

Both families store only valid observations in flat arrays with offsets.
Every model input is (B,T,1), all lengths within a batch identical. There is
no padding, packing, mask, or timestep-output selection. Classification uses
only h_n[-1]. Shuffle within each T, then shuffle all batches, deterministically
from model seed and epoch. Include all incomplete batches and audit each index
exactly once. Architecture: input1, hidden30, one layer, Linear30->2, dropout0.
Adam lr.001, batch256, fixed200 epochs, seed7, MPS, original sample-mean WCE
(class0=stationary,w2; class1=H0,w1=1). No optimizer workaround is applied.

Pooled held-out calibration uses coarse [.2,.4,.6,.8,1], necessary .2 range
extension and .05 fine points under the existing search rule. Selection
minimizes absolute pooled size error from .05 (lower weight breaks ties).
Per-T calibration size/counts are recorded but never used for selection.
Calibration has H0 only, so its power is null. Write selection.json before
generating independent test. Final test reports pooled size/power/accuracy,
rho power and each T's size/power/counts. Test never affects calibration.

Every epoch publishes an immutable checkpoint using temp-file, fsync, atomic
rename and checksum receipt. Model/optimizer, epoch/history, Torch/MPS,
Python/NumPy RNG, and deterministic next shuffle state are saved. Resume
validates source/protocol/data/checkpoint integrity, reuses completed weights,
and resumes incomplete attempts from the latest valid epoch in a fresh attempt.
Prior attempts remain intact. Finite loss/gradient checks precede updates;
finite parameters and Adam state are checked after every update. Coverage and
metrics are checked each epoch. W&B failures are recorded separately.

Run grid50 with:
    .venv/bin/python -u -m scripts.run_lstm_multilength_grid50
Resume with the same command plus --resume.
Output/W&B group: lstm_multilength_grid50_equalN.
--smoke uses distinct seeds (+100000000), timestamped output/group, three
epochs at w2=.6, and tiny counts (grid:1200 train/240 calibration/840 test).
No production test observations are used by smoke checks.

This tests pooled feasibility under one model seed, not nominal size at every
length or robustness across model seeds. Per-T results remain visible.
