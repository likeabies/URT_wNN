# Binary LSTM labels

Public LSTM class order is **[stationary, unit_root]**: stationary/non-unit-root
is 0; unit root is 1. H0 remains unit root. Size is P(predict 0 | actual 1),
power is P(predict 0 | actual 0). `w1` always weights unit-root samples and `w2`
stationary samples; the LSTM sample-mean loss uses `[w2, w1]`, without dividing
by the sum of weights.

## Boundaries

`src/lstm_binary.py` provides LSTM data wrappers, weighted loss, prediction,
evaluation, and class metrics. Generic ANN data/loss/metrics retain their old
convention; callers must not mix them with the LSTM API. `run_binary`,
`run_non_theoretical`, fixed-200 calibration/extension, independent evaluation,
and variable/shared LSTM runners use the LSTM-specific boundary. Neither RNN
nor integration-order I0/I1/I2 nor three-class LSTM semantics are changed.

The DGP and sample arrays are unchanged. New variable-length datasets store
canonical labels and `label_order` metadata. Historical unmarked variable-length
caches are read as legacy labels and converted only in memory after validating
against rho. Their immutable stored y is retained for original fingerprint
verification. Fixed-T fingerprints use the original ANN label encoding for
sample identity; returned LSTM labels are canonical. No archived data is rewritten.

## Checkpoint compatibility

Both old and new LSTM checkpoints keep **physical fc rows [unit_root,
stationary]**, including optimizer moments. Binary `forward` flips logits into
the public order [stationary, unit_root]. Thus direct `load_state_dict` of the
historical checkpoint is supported; no head/state mutation, extra RNG draw, or
optimizer conversion occurs. New initialization and restored optimizer states
retain the original statistical meaning. Do not additionally swap checkpoint
rows: that would convert twice. Three-output models are not permuted.

The LSTM prediction helper preserves the old tie decision (unit root), now
class 1. Clipping and nonfinite checks retain their existing positions.

## Archived metrics

Rows are actual and columns predicted. Reverse **both** axes of a legacy CM:
`[[a,b],[c,d]] -> [[d,c],[b,a]]`. Precision, recall, specificity and F1 are
recomputed in [stationary, unit_root] order. Recall_stationary=power and
Recall_unit_root=1-size (when each class has observations). Accuracy, size,
and power must be invariant. Undefined class ratios are reported as zero,
matching the prior aggregate metric policy.

`python -m scripts.migrate_lstm_label_results --root /path/to/original/project
--output /new/report/directory --verify-checkpoints` reads archived cell counts
or saved predictions. The optional inference check uses only saved official
arrays/checkpoints; it never trains, calibrates, or generates data. Output must
be a new directory. The report contains source hashes and all reconstructed CMs.
The fixed-T evaluator continues checking untouched DGP/generator source hashes
and official fingerprints; migrated model/loss/metric sources have new hashes
recorded in each new evaluation manifest rather than claiming old source identity.

See `reports/lstm_label_migration_20261103/REPORT.md` for the local verification.
