"""Binary LSTM convention; ANN storage/metrics remain legacy and unchanged.

Public classes: stationary=0, unit_root=1. Model fc rows remain in historical
order for checkpoint/optimizer/initialization compatibility; forward flips logits.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from src import data as legacy_data

LABELS = {'stationary': 0, 'unit_root': 1}
LABEL_ORDER = ['stationary', 'unit_root']


def from_legacy(raw):
    """Return a new label array, never mutate archived data or its fingerprint."""
    return {**raw, 'y': 1 - raw['y']}


def make_binary_dataset(*args, **kwargs):
    return from_legacy(legacy_data.make_binary_dataset(*args, **kwargs))


def make_non_theoretical_binary_testset(*args, **kwargs):
    return from_legacy(legacy_data.make_non_theoretical_binary_testset(*args, **kwargs))


def make_lstm_dataset(*args, **kwargs):
    return from_legacy(legacy_data.make_ann_dataset(*args, **kwargs))


def dataset(*args, **kwargs):
    from scripts.run_ann_paper2019 import dataset as original
    raw, seed = original(*args, **kwargs)
    return from_legacy(raw), seed


def fingerprint(raw):
    """Legacy ANN-compatible identity of fixed-T samples (including legacy y)."""
    from scripts.run_ann_paper2019 import fingerprint as original
    return original(from_legacy(raw))


def weighted_cross_entropy(logits, targets, w1=1., w2=1.):
    """w1=unit root, w2=stationary; sample mean, NOT weight-normalized mean."""
    logp = logits.log_softmax(1)
    weights = logits.new_tensor([w2, w1])
    return -(weights[targets] * logp.gather(1, targets[:, None]).squeeze(1)).mean()


def predict(logits, softmax=False):
    """Preserve historical argmax tie rule: equal logits predict unit root."""
    legacy = logits.flip(1)
    return 1 - (legacy.softmax(1) if softmax else legacy).argmax(1)


def metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (2, 2) or (cm < 0).any() or not cm.sum():
        raise ValueError('Expected a nonempty 2x2 count matrix')
    tp = cm.diagonal(); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    def divide(a, b):
        return np.divide(a, b, out=np.zeros(2, dtype=float), where=b != 0)
    recall = divide(tp, tp+fn)
    return dict(label_order=LABEL_ORDER, confusion_matrix=cm.tolist(),
                accuracy=float(tp.sum()/cm.sum()),
                empirical_size=float(cm[1,0]/cm[1].sum()) if cm[1].sum() else 0.,
                empirical_power=float(recall[0]), precision=divide(tp,tp+fp).tolist(),
                recall=recall.tolist(), specificity=divide(tn,tn+fp).tolist(),
                f1=divide(2*tp,2*tp+fp+fn).tolist())


def binary_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all():
        raise ValueError('Expected aligned binary labels/predictions')
    cm = np.zeros((2,2),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    return metrics_from_cm(cm)


def evaluate_lstm(model, dataset, raw, criterion, batch_size=256, device='cpu'):
    from torch.utils.data import DataLoader
    model.eval(); predictions=[]; total_loss=0.
    with torch.no_grad():
        for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            logits=model(x.to(device))
            total_loss += criterion(logits,y.to(device)).item()*len(y)
            predictions.append(predict(logits, softmax=True).cpu().numpy())
    pred=np.concatenate(predictions); met=binary_metrics(raw['y'],pred)
    overall={k:met[k] for k in ('accuracy','empirical_size','empirical_power')}
    overall.update(loss=total_loss/len(pred),n=len(pred))
    cells=[]
    for rho in np.unique(raw['rho']):
        for beta in np.unique(raw['beta']):
            mask=(raw['rho']==rho)&(raw['beta']==beta)
            cells.append(dict(rho=float(rho),beta=float(beta),n=int(mask.sum()),
                              rejection_rate=float((pred[mask]==0).mean()),
                              interpretation='type_i_error' if rho==1 else 'power'))
    return dict(overall=overall,by_parameter=cells,class_metrics=met)


def evaluate(model, raw, criterion):
    state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        return evaluate_lstm(model,TensorDataset(torch.from_numpy(raw['x']),torch.from_numpy(raw['y'])),
                             raw,criterion,device=next(model.parameters()).device)
    finally:
        torch.set_rng_state(state)
        if mps_state is not None: torch.mps.set_rng_state(mps_state)
