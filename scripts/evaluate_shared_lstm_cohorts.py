"""Evaluation only: frozen completed shared LSTMs, new variable cohort, optional fixed arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.variable_length_lstm import VariableLengthLSTM, generate, load_data, fingerprint, batches
from src.lstm_binary import binary_metrics, predict, from_legacy


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2,allow_nan=False)


def discover():
    items=[]
    for p in Path('results').rglob('run_manifest.json'):
        m=read(p);c=m.get('protocol',{})
        if m.get('status') not in ['complete','completed'] or m.get('completed_epoch')!=200: continue
        if not c.get('length_grid') or len(c['length_grid'])<2: continue
        cp=Path(m['checkpoint'])
        assert cp.exists() and sha(cp)==m['checkpoint_sha256']
        items.append(dict(run=p.parent.name,ordering=c['ordering'],w2=c['w2'],batch_size=c['batch_size'],
                          seed=c['model_seed'],checkpoint=str(cp),sha256=sha(cp),manifest=str(p)))
    assert items
    return sorted(items,key=lambda r:(r['w2'],r['run']))


def evaluate(raw,items,out,label):
    rows=[]
    for item in items:
        cp=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['epoch']==200 and all(torch.isfinite(v).all() for v in cp['model'].values())
        model=VariableLengthLSTM().to('mps');model.load_state_dict(cp['model'],strict=True);model.eval()
        pred=np.full(len(raw['y']),-1,dtype=np.int64)
        with torch.inference_mode():
            for x,y,lengths,ids in batches(raw,256):
                logits=model(x.to('mps'))
                assert bool(torch.isfinite(logits).all())
                pred[ids]=predict(logits, softmax=True).cpu().numpy()
        assert np.all(pred>=0)
        met=binary_metrics(raw['y'],pred)
        row={k:item[k] for k in ['run','ordering','w2','batch_size','seed']}
        row.update(test_set=label,H0=int((raw['y']==1).sum()),H1=int((raw['y']==0).sum()),
                   size=met['empirical_size'],power=met['empirical_power'],accuracy=met['accuracy'])
        save(out/f'{item["run"]}_{label}.json',dict(metrics=row,checkpoint=item,test_only=True))
        np.save(out/f'{item["run"]}_{label}_predictions.npy',pred)
        assert sha(item['checkpoint'])==item['sha256']
        rows.append(row);print(json.dumps(row),flush=True)
        del model,cp;torch.mps.empty_cache()
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fixed-arrays',type=Path)
    args=parser.parse_args();out=args.output
    assert torch.backends.mps.is_available()
    if args.fixed_arrays:
        # Only evaluate already saved arrays. No fixed-T generation is performed here.
        items=read(out/'selected_completed_models.json');rows=[]
        from scripts.run_ann_paper2019 import fingerprint as fixed_fp
        from scripts.run_lstm_independent_test import FINGERPRINTS
        for t in [50,100,250]:
            path=args.fixed_arrays/f'T{t}.npz'
            digest=sha(path)
            with np.load(path) as z:raw={k:z[k] for k in ['x','y','rho','beta']}
            assert fixed_fp(raw)==FINGERPRINTS[t]
            raw=from_legacy(raw)
            raw['lengths']=np.full(len(raw['y']),t,dtype=np.int64)
            rows.extend(evaluate(raw,items,out,f'fixed_T{t}'))
            assert sha(path)==digest
        save(out/'fixed_results.json',rows)
        return
    out.mkdir(parents=True,exist_ok=False)
    items=discover();save(out/'selected_completed_models.json',items)
    c=dict(test_seed=20261008,min_length=50,max_length=500,test_counts=[3000,1000],
           rhos=[1.,.95,.9,.8],betas=[.9,.6,.3,0.],generation_chunk=2048)
    for item in items:
        train=read(item['manifest'])['protocol'];assert c['test_seed']!=train['data_seed']
        if 'calibration_seed' in train:assert c['test_seed']!=train['calibration_seed']
    raw,meta=generate(out/'variable_test',c,'test')
    assert meta['n']==24000 and meta['unit_root']==meta['stationary']==12000
    assert meta['length_min']>=50 and meta['length_max']<=500
    assert len(np.unique(raw['lengths']))==451
    for x,y,lengths,ids in batches(raw,256):
        assert torch.isfinite(x).all() and torch.max(x.mean(1).abs())<1e-6
        assert torch.allclose(x.abs().amax(1),torch.ones((len(ids),1)),atol=1e-6)
        assert np.all(raw['y'][ids]==(raw['rho'][ids]==1).astype(int))
    fp=fingerprint(raw);assert fp==meta['fingerprint']
    save(out/'test_protocol.json',dict(config=c,distribution='iid discrete uniform integers 50..500',
        counts='H0 3000/beta, H1 1000/(rho,beta); 12000 per class',dataset=meta,
        checkpoints_frozen_before_test=True,training_performed=False,test_based_selection=False,
        source_hashes={p:sha(p) for p in [__file__,'src/variable_length_lstm.py','src/data.py','src/evaluate.py']}))
    print('DATA VERIFIED '+json.dumps(dict(seed=20261008,n=24000,H0=12000,H1=12000,fingerprint=fp)),flush=True)
    rows=evaluate(raw,items,out,'variable_T50_500')
    assert fingerprint(raw)==fp
    save(out/'variable_results.json',rows)
    print('Variable cohort evaluation complete.',flush=True)

if __name__=='__main__':main()
