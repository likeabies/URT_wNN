# Grid50 safe Adam restart — prepared, NOT executed

No optimizer tests, LSTM sanity checks, training, production, calibration, or
independent test were executed for this preparation. Prior diagnostic A is
evidence for the scalar expression only, not validation of the new optimizer.

## Scope and unchanged settings

New config: configs/lstm_multilength_grid50_equalN_safeadam_20261020.json.
The old grid50/random-T configs, fixed-T sources and all prior result files
remain unchanged. New optimizer: src/grid50_safe_adam.py; used only when the
new config explicitly selects grid50_scaled_second_moment_v1.
The ordinary shared-runner branch still uses torch.optim.Adam.

Single shared LSTM(1,30,1) -> h_n[-1] -> Linear(30,2), no padding/packing/output
selection; T=50,100,...,500; exact-length shuffled batches. Train600000
(60000/T, balanced); H0 calibration60000 (6000/T); independent test120000
planned only after eventual calibration. Training seed20191007/model seed7,
lr.001/batch256/fixed200 epochs/WCE w1=1/MPS unchanged. No clipping or fallback.

Only the second moment uses scaled_grad=g*sqrt(1-beta2), followed by
v.mul_(beta2).addcmul_(scaled_grad,scaled_grad). The first moment lerp, CPU
step counter, bias correction, eps=1e-8, betas=(.9,.999), and parameter addcdiv
match installed torch2.7.1 noncapturable real-valued Adam with zero decay.
The implementation intentionally supports only the modes needed here.

## Future execution order (requires user's next execution authorization)

1. Run prepared CPU optimizer/fail-fast/epoch-gate unit tests:
   .venv/bin/python -m unittest tests.test_grid50_safeadam_gate -v
2. Run short CPU/MPS preflight, saving into a new directory:
   PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m scripts.check_grid50_safeadam --output analysis_outputs/grid50_safeadam_preflight_20261020
   If the path exists, use a new suffix; never overwrite it.
   This compares torch CPU Adam vs safe CPU/MPS parameters, first and second
   moments over normal and large finite gradients, reports absolute/relative
   errors and finiteness, and checks nine h_n-only batches at T50/250/500.
   Any mismatch fails preflight; do not launch production.
3. Only after all tests pass and execution is authorized, launch the epoch gate:
   WANDB_MODE=online PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u -m scripts.run_lstm_grid50_safeadam_gate --preflight analysis_outputs/grid50_safeadam_preflight_20261020/preflight.json
   For detached execution, wrap with nohup+caffeinate and use a NEW log file
   opened exclusively, e.g. logs/lstm_multilength_grid50_equalN_safeadam_20261020.log.
   No launch has been performed by this preparation.

## Hard stop / output plan

New output and W&B group: lstm_multilength_grid50_equalN_safeadam_20261020.
The entry point requires successful preflight with matching source/config
hashes, torch version and platform, explicit fallback=0 and available MPS.
An existing output directory is rejected. No old checkpoint or partial state
is imported. Fixed200 remains in config: the one-epoch gate is an operational
pause, not a changed training schedule.

Only w2=.2 epoch1 runs. Its immutable epoch_001.pt and checksum include model,
optimizer, RNG, history and shuffle specification. Full coverage/finite checks
precede checkpoint publication. Manifest becomes awaiting_user_approval;
the process exits. It does not evaluate calibration, generate test, finish
200 epochs or advance to another w2. Further execution needs new authorization
and a separately reviewed continuation path, intentionally not provided here.

## Fail-fast and visibility

src/grid50_step_audit.py snapshots the preceding valid model/optimizer and
RNG state in memory before each batch. Any failure publishes pre_failure_state.pt
atomically and numeric_failure.json once, preserving both good and failed
tensor states for forensic inspection. Those files are NOT resume checkpoints.
The outer runner records failed and propagates the exception; no next weight,
automatic restart or retry follows. Source/protocol and old attempts stay intact.

step_diagnostics.jsonl records every successful step, including around1312:
global max absolute gradient/parameter name, parameter/first-/second-moment
finiteness, second-moment maximum and NaN/Inf counts. Failures retain exact T,
batch/indices,w2,seed, per-tensor min/max/absolute-max and NaN/Inf counts.
Epoch history includes peak gradient over the epoch, final-batch gradient max,
peak second moment and end-of-epoch second moment. Audit snapshots/statistics
add overhead during this gate; they do not alter the update equations.

The existing interruption/atomic checkpoint machinery remains; this new entry
point deliberately offers no --resume and the safe configuration refuses a
failed or approval-waiting manifest. No automatic continuation is authorized.
