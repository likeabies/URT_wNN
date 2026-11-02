# Ordered shared-length RNN queue

Run one scheduler tick from the repository with `.venv/bin/python -m scripts.control_rnn_shared_length`.
The live protocol and append-only attempt directories reside in `results/rnn_shared_length_20261029`.
The versioned configuration is `configs/rnn_shared_length_ordered16.json`; the live protocol additionally records SHA256 hashes of the runner, controller, model, data, loss and uploaded optimizer source. Initialization must preserve existing attempts and must not overwrite a live protocol.

The sixteen conditions use eight parameter settings in the configured order, each ascending then descending. Every run visits T=50..500 in steps of25, with 31580 samples at each of the first nine lengths and31578 at each remaining length. Every length is class-balanced. Samples shuffle within each length; length blocks stay ordered, including short final batches. Each epoch uses exactly600000 samples without padding or packing.

The model is the prepared vanilla tanh RNN with hidden30, one layer, dropout0 and final hidden state followed by Linear(30,2). MPS float32, seeds20191007/7, fixed200 epochs, lr.001, weighted loss and clipping9553.912684754807 are preserved. The optimizer imports the unchanged merged `src.grid50_safe_adam.Grid50SafeAdam` implementation. Backward finite checks, existing clipping, post-clip checks, step and parameter/optimizer-state checks retain their order. CSV output contains only the nine batch fields specified in the runner.

The controller launches at most one process per tick, protects itself with a file lock, checks all existing attempts and live training processes, and advances after completed200 or numerical failure only. An interrupted or incomplete attempt blocks progression. Existing attempt directories are never retried. No clipping-frequency threshold is used. Once all sixteen attempts are terminal and processes have exited, it saves `summary.json` and returns `all_done` with `pause_automation=true`. The desktop heartbeat must pause itself on that result. There are no random-T or two-module runs in this queue.

Validation: `python -m unittest discover -s tests -p 'test_safe_adam.py' -v` and `python -m unittest discover -s tests -p 'test_rnn_shared_length_queue.py' -v`. Queue tests use temporary state and mocked launches; they do not train experimental candidates.
