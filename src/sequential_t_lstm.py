"""Grid25 data allocation and sequential exact-length batches for paired controls."""
import json
from pathlib import Path
import numpy as np
import torch
from src.lstm_binary import make_lstm_dataset as make_ann_dataset, LABEL_ORDER
from src.variable_length_lstm import load_data, fingerprint


def generate_shared(path, c, split):
    assert split == 'train'
    path = Path(path)
    if path.exists():
        meta = json.loads((path / 'dataset.json').read_text())
        assert meta['status'] == 'complete' and meta['seed'] == c['data_seed']
        assert meta['allocation'] == c['allocation']
        raw = load_data(path)
        assert fingerprint(raw) == meta['fingerprint']
        return raw, meta
    path.mkdir(parents=True, exist_ok=False)
    lens = np.concatenate([np.full(c['allocation'][str(t)], t, dtype=np.int64)
                           for t in c['length_grid']])
    offsets = np.concatenate(([0], np.cumsum(lens, dtype=np.int64)))
    n = len(lens)
    shapes = dict(x=(int(offsets[-1]),), y=(n,), rho=(n,), beta=(n,), lengths=(n,), offsets=(n+1,))
    dtypes = dict(x='float32', y='int64', rho='float64', beta='float64', lengths='int64', offsets='int64')
    raw = {k: np.lib.format.open_memmap(path/(k+'.npy'), mode='w+', dtype=dtypes[k], shape=s)
           for k, s in shapes.items()}
    raw['lengths'][:] = lens
    raw['offsets'][:] = offsets
    position = 0
    for t in c['length_grid']:
        half = c['allocation'][str(t)] // 2
        cell_index = 0
        for rhos in [[1.], [r for r in c['rhos'] if r != 1.]]:
            cells = [(r,b) for r in rhos for b in c['betas']]
            base, extra = divmod(half, len(cells))
            for j,(rho,beta) in enumerate(cells):
                count = base + int(j < extra)
                seed = int(np.random.SeedSequence([c['data_seed'],t,cell_index]).generate_state(1)[0])
                data = make_ann_dataset(t, [rho], [beta], count, count, seed)
                end = position + count
                raw['x'][offsets[position]:offsets[end]] = data['x'].reshape(-1)
                for k in ['y','rho','beta']: raw[k][position:end] = data[k]
                position = end
                cell_index += 1
    assert position == n == 600000
    for a in raw.values(): a.flush()
    by_t = {str(t): dict(n=int((lens==t).sum()),
                unit_root=int(((lens==t)&(raw['y']==1)).sum()),
                stationary=int(((lens==t)&(raw['y']==0)).sum())) for t in c['length_grid']}
    meta = dict(label_order=LABEL_ORDER, status='complete', seed=c['data_seed'], n=n, allocation=c['allocation'],
                by_T=by_t, fingerprint=fingerprint(raw),
                cell_allocation='Within each class: quotient per cell, remainder to first cells in config order',
                rng='SeedSequence([data_seed,T,cell_index])')
    tmp = path/'dataset.json.tmp'
    tmp.write_text(json.dumps(meta, indent=2)); tmp.replace(path/'dataset.json')
    return load_data(path), meta


def batch_plan(lengths, batch_size, model_seed, epoch, ordering):
    assert ordering in ('ascending','descending')
    rng = np.random.default_rng(np.random.SeedSequence([model_seed,epoch,917]))
    blocks = []
    # Draw all within-T permutations in ascending order for identical paired shuffles.
    for t in np.unique(lengths):
        ids = np.flatnonzero(lengths == t)
        rng.shuffle(ids)
        blocks.append([(ids[i:i+batch_size],int(t)) for i in range(0,len(ids),batch_size)])
    if ordering == 'descending': blocks.reverse()
    for block in blocks: yield from block


def batches(raw, batch_size, model_seed=7, epoch=0, ordering='ascending'):
    for ids,t in batch_plan(raw['lengths'],batch_size,model_seed,epoch,ordering):
        values = raw['x'][raw['offsets'][ids,None] + np.arange(t)]
        yield (torch.from_numpy(np.array(values,copy=True)).unsqueeze(-1),
               torch.from_numpy(np.array(raw['y'][ids],copy=True)),
               torch.from_numpy(np.array(raw['lengths'][ids],copy=True)),ids)
