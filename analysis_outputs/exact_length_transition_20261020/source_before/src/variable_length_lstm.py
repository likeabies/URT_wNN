"""Variable-length inputs; original DGP, normalization, labels and LSTM modules."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data import normalize_ann_series
from src.model import Paper2019LSTMClassifier


class VariableLengthLSTM(Paper2019LSTMClassifier):
    def forward(self, x, lengths):
        # A unidirectional LSTM cannot propagate right-padding backwards in time.
        output, _ = self.lstm(x.unsqueeze(-1))
        valid = output[torch.arange(len(x), device=x.device), lengths.to(x.device)-1]
        return self.fc(valid)


KEYS = ('x', 'y', 'rho', 'beta', 'lengths')


def fingerprint(raw):
    h = hashlib.sha256()
    for key in KEYS:
        a = raw[key]
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        for start in range(0, len(a), 2048):
            h.update(np.ascontiguousarray(a[start:start+2048]).tobytes())
    return h.hexdigest()


def load_data(path):
    return {key: np.load(Path(path)/(key+'.npy'), mmap_mode='r') for key in KEYS}


def generate(path, c, split):
    """Lengths and Gaussian innovations use independent child RNG streams.

    Simulate a path to each chunk's maximum length, then keep only its first T
    observations. Discarded Gaussian suffixes cannot affect the retained path.
    Centering/scaling uses only that retained path, never padding or the suffix.
    """
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    seed = c[{'train':'data_seed','calibration':'calibration_seed','test':'test_seed'}[split]]
    children = np.random.SeedSequence(seed).spawn(2)
    length_rng, noise_rng = [np.random.default_rng(child) for child in children]
    n0,n1 = c[split+'_counts']
    rhos = [1.] if split=='calibration' else c['rhos']
    n = sum(n0 if rho==1 else n1 for rho in rhos)*len(c['betas'])
    shapes = dict(x=(n,c['max_length']),y=(n,),rho=(n,),beta=(n,),lengths=(n,))
    dtypes = dict(x='float32',y='int64',rho='float64',beta='float64',lengths='int64')
    raw = {k:np.lib.format.open_memmap(path/(k+'.npy'), mode='w+', dtype=dtypes[k], shape=shapes[k]) for k in KEYS}
    offset = 0
    for rho in rhos:
        for beta in c['betas']:
            count = n0 if rho==1 else n1
            lengths = length_rng.integers(c['min_length'],c['max_length']+1,size=count)
            for start in range(0,count,c['generation_chunk']):
                lens = lengths[start:start+c['generation_chunk']]; batch=len(lens); longest=int(lens.max())
                errors=noise_rng.normal(size=(batch,longest+1))
                values=np.empty((batch,longest)); previous=np.zeros(batch)
                for t in range(longest):
                    previous=rho*previous+(beta*errors[:,t]+errors[:,t+1])
                    values[:,t]=previous
                x=np.zeros((batch,c['max_length']),dtype=np.float32)
                for length in np.unique(lens):
                    idx=np.flatnonzero(lens==length)
                    x[idx,:length]=normalize_ann_series(values[idx,:length])
                region=slice(offset,offset+batch)
                raw['x'][region]=x; raw['lengths'][region]=lens
                raw['y'][region]=int(rho!=1);raw['rho'][region]=rho;raw['beta'][region]=beta
                offset+=batch
    assert offset==n
    for a in raw.values():a.flush()
    meta=dict(status='complete',split=split,seed=seed,n=n,unit_root=int((raw['y']==0).sum()),
              stationary=int((raw['y']==1).sum()),fingerprint=fingerprint(raw),
              length_min=int(raw['lengths'].min()),length_max=int(raw['lengths'].max()),
              length_mean=float(raw['lengths'].mean()),
              rng='SeedSequence(seed).spawn(2): child0 lengths, child1 Gaussian innovations',
              generation_chunk=c['generation_chunk'])
    (path/'dataset.json').write_text(json.dumps(meta,indent=2))
    return load_data(path),meta


def batches(raw, batch_size, width, model_seed=None, epoch=0):
    caps=((raw['lengths']+width-1)//width)*width
    rng=np.random.default_rng(np.random.SeedSequence([model_seed,epoch,917])) if model_seed is not None else None
    blocks=[]
    for cap in np.unique(caps):
        indices=np.flatnonzero(caps==cap)
        if rng is not None:rng.shuffle(indices)
        blocks.extend((indices[i:i+batch_size],min(int(cap),raw['x'].shape[1])) for i in range(0,len(indices),batch_size))
    if rng is not None:rng.shuffle(blocks)
    for indices,cap in blocks:
        yield (torch.from_numpy(np.array(raw['x'][indices,:cap],copy=True)),
               torch.from_numpy(np.array(raw['y'][indices],copy=True)),
               torch.from_numpy(np.array(raw['lengths'][indices],copy=True)),indices)
