"""Exact-length batches, unpadded storage, unchanged DGP and h_n classifier."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.data import normalize_ann_series
from src.model import Paper2019LSTMClassifier
from src.lstm_binary import make_lstm_dataset as make_ann_dataset, LABEL_ORDER


class VariableLengthLSTM(Paper2019LSTMClassifier):
    def forward(self, x, lengths=None):
        if x.ndim != 3 or x.shape[2] != 1:
            raise ValueError('Expected an exact-length (B,T,1) batch')
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).flip(1)


KEYS=('x','y','rho','beta','lengths')


def fingerprint(raw):
    h=hashlib.sha256()
    for key in KEYS + (('offsets',) if 'offsets' in raw else ()):
        a=raw.get('_stored_y',raw[key]) if key=='y' else raw[key];h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
        for start in range(0,len(a),262144):h.update(np.ascontiguousarray(a[start:start+262144]).tobytes())
    return h.hexdigest()


def load_data(path):
    path=Path(path)
    keys=KEYS+(('offsets',) if (path/'offsets.npy').exists() else ())
    raw = {k:np.load(path/(k+'.npy'),mmap_mode='r') for k in keys}
    meta = json.loads((path/'dataset.json').read_text())
    order = meta.get('label_order', ['unit_root','stationary'])
    if order == ['unit_root','stationary']:
        raw['_stored_y'] = raw['y']; raw['y'] = 1 - raw['y']
    elif order != LABEL_ORDER:
        raise ValueError('Unknown dataset label order')
    if not np.array_equal(raw['y'], (raw['rho']==1).astype(np.int64)):
        raise ValueError('LSTM labels disagree with rho')
    return raw


def generate(path,c,split):
    """Flat concatenated valid series only: no padded storage or model inputs."""
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    seed=c[{'train':'data_seed','calibration':'calibration_seed','test':'test_seed'}[split]]
    n0,n1=c[split+'_counts'];rhos=[1.] if split=='calibration' else c['rhos']
    per_length=sum(n0 if rho==1 else n1 for rho in rhos)*len(c['betas'])
    grid=c.get('length_grid')
    if grid:
        lens=np.repeat(np.array(grid,dtype=np.int64),per_length)
    else:
        length_rng,noise_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        lens=np.concatenate([length_rng.integers(c['min_length'],c['max_length']+1,size=n0 if rho==1 else n1)
                             for rho in rhos for beta in c['betas']])
    n=len(lens);offsets=np.concatenate(([0],np.cumsum(lens,dtype=np.int64)))
    shapes=dict(x=(int(offsets[-1]),),y=(n,),rho=(n,),beta=(n,),lengths=(n,),offsets=(n+1,))
    dtypes=dict(x='float32',y='int64',rho='float64',beta='float64',lengths='int64',offsets='int64')
    raw={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype=dtypes[k],shape=v) for k,v in shapes.items()}
    raw['lengths'][:]=lens;raw['offsets'][:]=offsets
    position=0;derived={}
    if grid:
        for t in grid:
            child=int(np.random.SeedSequence([seed,t]).generate_state(1)[0]);derived[str(t)]=child
            data=make_ann_dataset(t,rhos,c['betas'],n0,n1,child)
            end=position+len(data['y']);raw['x'][offsets[position]:offsets[end]]=data['x'].reshape(-1)
            for k in ['y','rho','beta']:raw[k][position:end]=data[k]
            position=end
    else:
        # Same per-cell/chunk Gaussian stream as the earlier random-T DGP.
        for rho in rhos:
            for beta in c['betas']:
                count=n0 if rho==1 else n1
                for start in range(0,count,c['generation_chunk']):
                    size=min(c['generation_chunk'],count-start);ls=lens[position:position+size];longest=int(ls.max())
                    errors=noise_rng.normal(size=(size,longest+1));values=np.empty((size,longest));prev=np.zeros(size)
                    for t in range(longest):
                        prev=rho*prev+(beta*errors[:,t]+errors[:,t+1]);values[:,t]=prev
                    for length in np.unique(ls):
                        ids=np.flatnonzero(ls==length);normalized=normalize_ann_series(values[ids,:length]).astype(np.float32)
                        addresses=offsets[position+ids,None]+np.arange(length)
                        raw['x'][addresses]=normalized
                    raw['y'][position:position+size]=int(rho==1);raw['rho'][position:position+size]=rho;raw['beta'][position:position+size]=beta
                    position+=size
    assert position==n
    for a in raw.values():a.flush()
    by_t={str(t):dict(n=int((lens==t).sum()),unit_root=int(((lens==t)&(raw['y']==1)).sum()),
                     stationary=int(((lens==t)&(raw['y']==0)).sum())) for t in np.unique(lens)}
    meta=dict(label_order=LABEL_ORDER,status='complete',split=split,seed=seed,n=n,unit_root=int((raw['y']==1).sum()),stationary=int((raw['y']==0).sum()),
        fingerprint=fingerprint(raw),format='flat_valid_values_and_offsets',length_min=int(lens.min()),length_max=int(lens.max()),
        length_mean=float(lens.mean()),by_T=by_t,derived_seeds=derived,generation_chunk=c['generation_chunk'],
        rng='SeedSequence([split_seed,T]) for grid; independent length/noise child streams for random T')
    (path/'dataset.json').write_text(json.dumps(meta,indent=2));return load_data(path),meta


def batches(raw,batch_size,width=None,model_seed=None,epoch=0):
    """width is retained only for call compatibility; no length rounding is used."""
    rng=np.random.default_rng(np.random.SeedSequence([model_seed,epoch,917])) if model_seed is not None else None
    blocks=[]
    for length in np.unique(raw['lengths']):
        ids=np.flatnonzero(raw['lengths']==length)
        if rng is not None:rng.shuffle(ids)
        blocks.extend((ids[i:i+batch_size],int(length)) for i in range(0,len(ids),batch_size))
    if rng is not None:rng.shuffle(blocks)
    for ids,length in blocks:
        assert (raw['lengths'][ids]==length).all()
        if 'offsets' in raw:
            values=raw['x'][raw['offsets'][ids,None]+np.arange(length)]
        else:
            # Read-only compatibility for immutable historical arrays; use only actual observations.
            values=raw['x'][ids,:length]
        x=torch.from_numpy(np.array(values,copy=True)).unsqueeze(-1)
        yield x,torch.from_numpy(np.array(raw['y'][ids],copy=True)),torch.from_numpy(np.array(raw['lengths'][ids],copy=True)),ids
