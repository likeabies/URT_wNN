"""Fine-tuning utilities; historical model/DGP code is deliberately reused."""
import numpy as np
import torch
from torch.nn import functional as F


def lower_null_threshold(scores, alpha=0.05):
    """Empirical lower-tail order statistic, inclusive rejection, no tie breaking."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected nonempty finite null-cohort scores')
    k = int(round(alpha * len(scores)))
    if not 1 <= k <= len(scores):
        raise ValueError('Null cohort too small for requested alpha')
    c = float(np.sort(scores)[k - 1])
    return dict(threshold=c, n=len(scores), alpha=alpha,
                achieved=float((scores <= c).mean()),
                rule='sorted[round(alpha*n)-1]; reject score <= threshold; no tie randomization')


def calibrate(scores, orders, alpha=0.05):
    result = {}
    for j in range(scores.shape[1]):
        mask = orders == j + 1
        result[f'c{j+1}'] = (dict(null_order=j+1, **lower_null_threshold(scores[mask, j], alpha))
                              if mask.any() else dict(null_order=j+1, threshold=None,
                              unavailable='Existing calibration cohort has no true I(%d)' % (j+1)))
    return result


def margin_loss(z, labels, margin=0.5):
    """One nearest opposite-label partner per anchor; mean squared hinge L2.

    All stationary rhos are candidates. The pair selection is label-only and
    detached; distances on selected pairs remain differentiable. Duplicate
    unordered pairs are intentional: this is a directed, per-anchor objective.
    """
    z = F.normalize(z, p=2, dim=1)
    opposite = labels[:, None] != labels[None, :]
    valid = opposite.any(1)
    if not valid.any():
        return z.sum() * 0
    with torch.no_grad():
        distances = (z.detach()[:, None, :] - z.detach()[None, :, :]).square().sum(-1)
        partner = distances.masked_fill(~opposite, float('inf')).argmin(1)
    distance = torch.linalg.vector_norm(z[valid] - z[partner[valid]], dim=1)
    return F.relu(margin - distance).square().mean()


def sample_indices(raw, mode, rhos, betas, rng, n=230400):
    """Stratified resampling of existing TRAIN ONLY; exact beta cell balance.

    hard: half I0/rho .99; half I1/rho1; no I2 or I3.
    replay: 35% I0/.99 + 35% I1/rho1 + 30% other stationary rhos.
    margin: original equal I0/I1/I2/I3, uniform I0 rho/beta cells.
    Balanced BCE weights are calculated from each selected cohort using the
    unchanged baseline N/(2*n_class) formula.
    """
    if mode == 'hard_only':
        cells = [(0, .99, n//2), (1, 1., n//2)]
    elif mode == 'hard_replay':
        other = [r for r in rhos if r != .99]
        cells = [(0, .99, n*35//100), (1, 1., n*35//100)]
        cells += [(0, r, n*30//(100*len(other))) for r in other]
    elif mode == 'hard_margin':
        cells = [(0, r, n//(4*len(rhos))) for r in rhos]
        cells += [(d, 1., n//4) for d in (1,2,3)]
    else:
        raise ValueError(mode)
    selected = []
    for d, rho, count in cells:
        if count % len(betas):
            raise ValueError('Sample budget must preserve exact beta balance')
        for beta in betas:
            pool = np.flatnonzero((raw['d']==d) & (raw['rho']==rho) & (raw['beta']==beta))
            if not len(pool):
                raise ValueError(f'Missing training cell {(d,rho,beta)}')
            selected.append(rng.choice(pool, count//len(betas), replace=True))
    indices = np.concatenate(selected)
    if len(indices) != n:
        raise ValueError('Incorrect sample budget')
    return rng.permutation(indices)


def stage_metrics(scores, raw, thresholds):
    d, rho = raw['d'], raw['rho']
    reject = scores[:, 0] <= thresholds['c1']['threshold']
    result = dict(stage1_size=float(reject[d==1].mean()),
                  stage1_I0_power=float(reject[d==0].mean()),
                  rho_power={str(float(r)):float(reject[(d==0)&(rho==r)].mean())
                             for r in sorted(np.unique(rho[d==0]))},
                  H0_N=int((d==1).sum()), H1_N=int((d==0).sum()))
    for j in range(scores.shape[1]):
        c = thresholds[f'c{j+1}']['threshold']
        result[f'stage{j+1}_null_size'] = (float((scores[d==j+1,j] <= c).mean())
                                         if c is not None and (d==j+1).any() else None)
    if thresholds['c2']['threshold'] is not None:
        bits = (scores[:,:2] > np.array([thresholds['c1']['threshold'],thresholds['c2']['threshold']])).astype(int)
        result['raw_stage12_patterns'] = {str(int(order)):{f'{a}{b}':int(((bits[:,0]==a)&(bits[:,1]==b)&(d==order)).sum())
                                          for a in (0,1) for b in (0,1)} for order in np.unique(d)}
    result['score_summary'] = {str(int(order)):{f'p{j+1}':dict(mean=float(scores[d==order,j].mean()),
                                  quantiles=np.quantile(scores[d==order,j],[0,.05,.5,.95,1]).tolist())
                                  for j in range(scores.shape[1])} for order in np.unique(d)}
    return result


def stage1_weights(d):
    y = np.asarray(d) >= 1
    return np.array([len(y)/(2*(~y).sum()), len(y)/(2*y.sum())], dtype=np.float32)


def boundary_margin(z, d, margin=0.5):
    keep = (d == 0) | (d == 1)
    return margin_loss(z[keep], d[keep] == 1, margin)


def stage1_only_loss(model, level, d, weight):
    _, (h, _) = model.lstm(level.unsqueeze(-1))
    logit = model.stage1(h[-1]).squeeze(-1)
    y = (d >= 1).to(logit.dtype)
    w = torch.where(y.bool(), weight[1], weight[0])
    return (F.binary_cross_entropy_with_logits(logit, y, reduction='none') * w).mean()
