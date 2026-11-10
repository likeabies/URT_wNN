"""Three differentiable binary heads on a shared official-style LSTM encoder."""

import hashlib
import numpy as np
import torch
from torch import nn
from src.data import normalize_ann_series


RHOS = [.99, .95, .9, .5, .2]
BETAS = [.99, .95, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.]


class IntegrationOrderLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=30,
            num_layers=1,
            batch_first=True
        )

        self.stage1 = nn.Linear(30, 1)
        self.stage2 = nn.Linear(31, 1)
        self.stage3 = nn.Linear(31, 1)

    def forward(self, level, difference, second_difference):

        # Stage 1: I(0) vs I(1)+
        _, (h1, _) = self.lstm(level.unsqueeze(-1))
        logit1 = self.stage1(h1[-1]).squeeze(-1)

        # Stage 2: I(1) vs I(2)+
        _, (h2, _) = self.lstm(difference.unsqueeze(-1))
        logit2 = self.stage2(
            torch.cat(
                (
                    h2[-1],
                    torch.sigmoid(logit1).unsqueeze(1)
                ),
                dim=1
            )
        ).squeeze(-1)

        # Stage 3: I(2) vs I(3)
        _, (h3, _) = self.lstm(second_difference.unsqueeze(-1))
        logit3 = self.stage3(
            torch.cat(
                (
                    h3[-1],
                    torch.sigmoid(logit2).unsqueeze(1)
                ),
                dim=1
            )
        ).squeeze(-1)

        return torch.stack(
            (logit1, logit2, logit3),
            dim=1
        )


def balanced_weights(d):

    y = np.column_stack(
        (
            d >= 1,
            d >= 2,
            d >= 3
        )
    )

    n = len(d)

    return np.array(
        [
            [
                n / (2 * (~y[:, j]).sum()),
                n / (2 * y[:, j].sum())
            ]
            for j in range(3)
        ],
        dtype=np.float32
    )


def loss_parts(logits, d, weights):

    y = torch.stack(
        (
            d >= 1,
            d >= 2,
            d >= 3
        ),
        dim=1
    ).to(logits.dtype)

    w = torch.where(
        y.bool(),
        weights[:, 1],
        weights[:, 0]
    )

    return (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none"
        ) * w
    ).mean(0)


def parent_paths(errors, rho, beta, order):

    z = beta * errors[:, :-1] + errors[:, 1:]

    if order == 2:
        return np.cumsum(
            np.cumsum(z, axis=1),
            axis=1
        )

    if order == 3:
        return np.cumsum(
            np.cumsum(
                np.cumsum(z, axis=1),
                axis=1
            ),
            axis=1
        )

    values = np.empty_like(z)
    previous = np.zeros(len(z))

    for t in range(z.shape[1]):
        previous = rho * previous + z[:, t]
        values[:, t] = previous

    return values


def dataset(t, n_per_class, seed, orders=(0, 1, 2, 3)):

    derived = int(
        np.random.SeedSequence([seed, t]).generate_state(1)[0]
    )

    rng = np.random.default_rng(derived)

    n = n_per_class * len(orders)

    raw = dict(
        level=np.empty((n, t), np.float32),
        difference=np.empty((n, t - 1), np.float32),
        second_difference=np.empty((n, t - 2), np.float32),
        d=np.empty(n, np.int64),
        rho=np.empty(n, np.float64),
        beta=np.empty(n, np.float64)
    )

    pos = 0

    for order in orders:

        rhos = RHOS if order == 0 else [1.]

        assert n_per_class % (len(rhos) * len(BETAS)) == 0

        count = n_per_class // (len(rhos) * len(BETAS))

        for rho in rhos:
            for beta in BETAS:

                errors = rng.normal(
                    size=(count, t + 1)
                )

                values = parent_paths(
                    errors,
                    rho,
                    beta,
                    order
                )

                first_diff = np.diff(
                    values,
                    axis=1
                )

                second_diff = np.diff(
                    first_diff,
                    axis=1
                )

                end = pos + count

                raw["level"][pos:end] = normalize_ann_series(
                    values
                )

                raw["difference"][pos:end] = normalize_ann_series(
                    first_diff
                )

                raw["second_difference"][pos:end] = normalize_ann_series(
                    second_diff
                )

                raw["d"][pos:end] = order
                raw["rho"][pos:end] = rho
                raw["beta"][pos:end] = beta

                pos = end

    order = rng.permutation(n)

    return {
        k: v[order]
        for k, v in raw.items()
    }, derived


def fingerprint(raw):

    h = hashlib.sha256()

    for k in [
        "level",
        "difference",
        "second_difference",
        "d",
        "rho",
        "beta"
    ]:
        a = raw[k]

        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    return h.hexdigest()


def threshold(values, alpha=.05):

    values = np.asarray(values)

    assert np.isfinite(values).all()
    assert len(values) > 0

    # Strict > rejects the largest alpha*N observations
    # when there are no ties.
    k = (
        len(values)
        - int(round(alpha * len(values)))
        - 1
    )

    c = float(
        np.sort(values)[k]
    )

    return dict(
        threshold=c,
        n=len(values),
        target=alpha,
        achieved=float((values > c).mean()),
        rule=(
            "sorted[n-round(alpha*n)-1]; "
            "strict >; no random tie breaking"
        )
    )


def classification(d, pred):

    cm = np.zeros(
        (4, 4),
        dtype=np.int64
    )

    np.add.at(
        cm,
        (d, pred),
        1
    )

    recall = np.divide(
        cm.diagonal(),
        cm.sum(1),
        out=np.zeros(4, dtype=float),
        where=cm.sum(1) != 0
    )

    precision = np.divide(
        cm.diagonal(),
        cm.sum(0),
        out=np.zeros(4, dtype=float),
        where=cm.sum(0) != 0
    )

    return dict(
        confusion_matrix=cm.tolist(),
        recall=recall.tolist(),
        precision=precision.tolist(),
        accuracy=float((d == pred).mean())
    )