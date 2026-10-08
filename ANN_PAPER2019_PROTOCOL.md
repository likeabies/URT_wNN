# Primary ANN reproduction protocol

Primary family: `ann_paper2019_reproduction`.

Supplementary category: `validation_based_calibration_sensitivity`, containing the untouched `ann_valratio_1to5_vs_1to1_5x5` and `ann_valratio_fine_5x5` results. Neither determines the primary weights or fixed epoch.

## Current status and next step

The three T=100, w2=1 convergence trajectories (model seeds 7,17,27) completed
checkpoints 50/100/150/200/300. Aggregate loss and accuracy are nearly plateaued
by 200, but the documented per-seed rate-change gate rejected both 150 and 200.
The workflow stopped normally before hyperparameter sensitivity, calibration,
polynomial fitting or final testing. Generated results are not part of this PR.

The proposed next step after merge is fixed 200-epoch w2 calibration for all
three T values, following explicit resolution of the diagnostic gate and the
pending sensitivity screen. This PR does not relax thresholds, select a fixed
epoch, implement automatic continuation, or claim complete paper reproduction.
The existing primary output directory is preserved: rerunning the entry point
against it intentionally fails instead of overwriting the diagnostic run.

Source: Ekanayake & Samaranayake (2019), https://ww2.amstat.org/meetings/proceedings/2019/data/assets/pdf/1199554.pdf, sections 2.2–3. The new entry point reuses the existing ANN DGP, normalization, architecture, weighted loss, and minibatch training function; it never calls validation-based `train_model` or generates validation samples.

## Specified methodology and explicit assumptions

- T=50/100/250; hidden sizes 20/50/50, ReLU and two softmax probabilities, argmax classification.
- Y0=0, alpha=0, normal innovations; existing standard-normal and initial innovation conventions retained. Per-series centering and division by maximum absolute centered value.
- Training rho=[1,.99,.95,.9,.5,.2], 12 beta values including zero. 5,000 null samples per beta and 1,000 per stationary cell: 60,000+60,000 series. Beta=0 is included as directed, consistent with the paper's total counts and subsequent tables despite its omission in printed Table 1.
- Literal printed loss convention: class 0 is unit root with weight w1=1, class 1 is stationary with weight w2. Increasing w2 favors stationary classification. The paper's reported weights above one are not targets and appear inconsistent with the direction expected from its printed equation. We do not swap labels, invert weights, or silently redesign the grid.
- Paper-unspecified choices: Adam, lr=.001, batch=256, default PyTorch initialization, zero dropout/weight decay, shuffled minibatches, deterministic seeds (7,17,27), primary data seed 20191007 with T-specific SeedSequence streams. Seeded MPS does not promise cross-version bitwise reproducibility.
- Fixed-epoch training; no validation, early stopping, or best-epoch checkpoint selection. Calibration error uses the training null cohort. Epoch diagnostics use loss and stability, never proximity to .05.

## Diagnostic decisions fixed before observing results

Three T=100/w2=1 trajectories each run once to 300 epochs. Save model, optimizer, RNG, and full training-cohort evaluations at 50/100/150/200/300. Checkpoint evaluation preserves the RNG so it cannot alter subsequent shuffle order.

Select the first of 150 then 200 for which every seed's later checkpoints differ by at most 2% relative training loss and .01 absolute accuracy, Type I error, and power. These pragmatic thresholds are implementation choices, not formal statistical tests or paper requirements. If neither qualifies, stop and report; never select 300 solely because it is longest.

One-factor sensitivity alternatives: (.0005,256), (.002,256), (.001,128), (.001,512), initially three paired model seeds. Reuse convergence checkpoints as (.001,256) baseline. Flag a major issue if any paired alternative changes loss by more than 5% or training size/power by more than .03. If flagged, stop before calibration; do not optimize hyperparameters.

## Calibration and polynomial gates

For each T, train w2=[.1,.2,.35,.5,.75,1,1.5,2,3], three model seeds and identical underlying training data across weights. Save full cell rejection rates including beta-specific training size. There are 81 logical calibration results; three T100/w2=1 checkpoints can be reused exactly from convergence at the selected epoch.

Fit ordinary least-squares degree 2 and degree 3 polynomials to all nine mean training Type I errors. No test-based fitting or polynomial-degree averaging. A root is admissible only if real and inside an adjacent observed interval bracketing .05. Each degree must have exactly one admissible root. Their difference must not exceed max(.01, 5% of the quadratic root). These tolerances are explicit implementation judgments. If roots are missing, ambiguous, or materially different, stop and report the coefficients, roots and RMSE before final fitting. Do not silently change fitting windows to obtain agreement. If acceptable, select degree 2 and freeze all T-specific weights before creating test samples.

## Final fits and independent test

Nine fresh final trajectories: each T and three model seeds, full training sample, chosen fixed epoch and quadratic weight. Fresh test stream seed 20261008 is separate from all prior sweep seeds. Test grid adds rho=.8 to the full training parameter grid, retaining 12 betas and 1,000 samples per cell (84,000 per T). Report full cell means/SD/min/max and the paper subset rho=[1,.95,.9,.8], beta=[.9,.6,.3,0]. Different aggregate power grids must not be conflated. Test data never feed calibration or diagnostic decisions.

ADF comparison is deferred. Existing AIC-autolag is not relabeled as paper GtS. This is an ANN component reproduction; complete ANN-versus-paper-ADF reproduction remains a separate substep.

## Compute, records, and preservation

Expected new fits: 3 convergence + 12 sensitivity + 78 calibration + 9 final = 102 (plus 3 tiny smoke fits); 81 logical calibration results. No independent retraining for each checkpoint. Initial estimates from measured training-only epoch times plus 10% overhead: 11.83 hours at 150 epochs; 15.55 hours at 200. Re-estimate using convergence timing before calibration. If over 16 hours, reduce alternative diagnostic seeds from 3 to 2 then 1, preserving all calibration/final seeds, all T, the grid, and paper sample structure. If still over budget, stop and report. Wall-clock estimates are not guarantees.

Run via `WANDB_MODE=online nohup caffeinate -is .venv/bin/python -u -m scripts.run_ann_paper2019`. A smoke run uses `--smoke`, tiny samples, all three T values, checkpoint reload and polynomial guard tests. Results and status are under a new family directory; existing family directories cause refusal rather than overwrite. The workflow advances automatically only when its gates pass, and records `needs_attention` or `failed` otherwise. No commit or push is performed.
