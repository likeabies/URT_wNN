# Variable-length LSTM feasibility experiment

Run `python -m scripts.run_lstm_variable_length` with
`configs/lstm_variable_length_feasibility.json`. This separate experiment does
not alter fixed-T code, checkpoints, calibration, or official test results.

One `Paper2019LSTMClassifier` parameter structure (LSTM input 1, hidden 30,
one unidirectional layer, Linear 30→2) handles lengths 50–500. The subclass
changes only forward indexing: select the LSTM output at each actual length−1.
Right-padding cannot affect earlier states of the unidirectional LSTM. No
length embedding, extra module, bidirectionality, or threshold adjustment is
introduced. CPU tests check logits and zero padding gradients; MPS checks
compare padded batches with individually unpadded original-model predictions.

Each series gets an independent uniform integer length in [50,500]. The DGP
is the original Gaussian AR(1) with MA(1) innovations, Y0=0, alpha=0, and iid e0.
For efficient chunk generation, Gaussian paths extend to the chunk maximum
length; discarded suffixes have no effect on the retained prefix. The original
normalization function centers and max-abs scales each retained prefix before
padding. Fingerprints cover normalized float32 x, labels, rho, beta, and lengths.
Dataset cache files are immutable NumPy arrays and read through memory maps.

Seeds are train=20191007, held-out H0 calibration=20261019, independent
test=20261020, model=7. Each data seed spawns two independent NumPy streams:
lengths and innovations. Generation chunk size 2048 is frozen in config because
it affects exact RNG consumption. Train has 600,000 samples (300,000/class),
calibration has 60,000 H0 samples, test has 120,000 samples (60,000/class).
Cell counts and rho/beta grids follow the config and the requested design.

Length buckets have width 25. Shuffle observations within each bucket and
shuffle the batch order every epoch; include each observation exactly once.
Batch size remains 256 with smaller final bucket batches. Bucket padding rounds
lengths up to the next multiple of 25, capped at 500. This saves compute while
retaining the same parameter structure and all observations. Each weight uses
the same initial model seed, training batch order, and train/calibration samples.
Adam lr=.001, weight decay=0, no dropout, fixed 200 epochs, original sample-mean
WCE with class 0=unit root/w1=1, class 1=stationary/w2. Device selection is
CUDA→MPS→CPU; actual device is recorded and cannot change during resume.

Evaluate only pooled held-out calibration size after each completed fit.
Run all coarse weights [.2,.4,.6,.8,1.0]. If unbracketed, extend only the needed
side by .2: all sizes below target extends upwards; all above extends downwards.
Weights cannot be negative. If w2=0 is still unbracketed, record that boundary
and select the closest observed result rather than use a negative WCE weight.
For an observed crossing, the existing local_grid selector chooses one interval
(nearest coarse point/crossing, endpoint error and lower interval tie-breaks).
Evaluate missing .05-spaced points in that interval, reusing endpoints. An
exact coarse hit at .05 needs no refinement. Select the minimum absolute pooled
size error over ALL completed weights; lower w2 breaks ties. Length-specific
size is never calculated for calibration or used by selection.

Write selection.json before even generating the independent test. Then evaluate
only that frozen checkpoint: pooled size, pooled power, accuracy and mean power
by stationary rho. Final results, calibration table, all run histories,
100/150/175/200 convergence snapshots and interval changes are saved locally.
W&B receives epoch metrics, calibration results, and a separate final test run
under group `lstm_variable_length_feasibility_20261019`.

Resume with `python -m scripts.run_lstm_variable_length --resume`. Exclusive
process locking prevents concurrent runners. Completed records, config, data
fingerprints, checkpoint hashes/epoch, and history coverage are validated before
reuse. Each valid epoch now publishes an immutable checkpoint with an atomic
temporary-file rename and checksum receipt. It contains model, Adam state,
completed epoch, verified history, Python/NumPy/Torch/MPS (or CUDA) RNG state,
and the deterministic next-epoch shuffle generator specification/state.
An interrupted attempt is retained and resumed in a unique directory/W&B run
from its most recent valid checksum-verified checkpoint. Corrupt checkpoints
are reported and preserved; the next valid earlier epoch is used. No checkpoint
is overwritten. Every epoch verifies processed and unique counts, missing and
duplicate indices, finite loss/accuracy/gradients/parameters/optimizer state,
and accuracy range. Nonfinite gradients stop before an optimizer update.
W&B exceptions are recorded locally without invalidating saved training state.
Selection, once
written, is frozen; a saved independent test is reused. A fully complete
experiment is a no-op. Protocol/source changes fail resume instead of silently
mixing experiments. The resume code assumes the same machine/device and source.

The original failed 20261019 attempt predates epoch checkpointing and has only
history through epoch 28, not an epoch-28 checkpoint. `--recover-from OLD
--output NEW` starts a separate experiment and reuses only fingerprint-verified
training/calibration datasets, preserving the failed experiment. Scientific
config fields must match. The w2=.2 run records a verification gate after 35
fully audited epochs before proceeding through the remaining fixed-200 run.
This recovery mode is only for an experiment with zero completed fits.

`--smoke` uses its own timestamped folder/group, seeds increased by 100,000,000,
12,000 training samples, 600 calibration samples, 1,200 test samples, one fixed
diagnostic w2=.6, and three epochs. It checks the pipeline, not feasibility or
scientific performance. It never examines production test samples. The measured
post-warmup batch time provides an approximate production runtime estimate.

Production launch (no early stopping):

```bash
WANDB_MODE=online PYTORCH_ENABLE_MPS_FALLBACK=0 nohup caffeinate -is \
  .venv/bin/python -u -m scripts.run_lstm_variable_length \
  > logs/lstm_variable_length_feasibility_20261019.log 2>&1 < /dev/null &
```

The goal is pooled feasibility over the specified length mixture, not a
guarantee of nominal size or good power at every individual length. A single
model seed cannot establish seed robustness. Test results never trigger further
weight selection, training, or calibration.
