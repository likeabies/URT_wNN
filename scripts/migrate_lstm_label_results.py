"""Read archived binary LSTM results; write a new convention report only.

No training, data generation, calibration, or writes to source results/checkpoints.
Optional --verify-checkpoints performs inference on already-saved official arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.lstm_binary import binary_metrics, metrics_from_cm, predict, LABEL_ORDER
from src.model import Paper2019LSTMClassifier
from scripts.run_ann_paper2019 import fingerprint
from scripts.run_lstm_independent_test import FINGERPRINTS


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def from_cells(ev):
    old=np.zeros((2,2),dtype=np.int64)
    for cell in ev['by_parameter']:
        n=cell['n']; rejected=cell['rejection_rate']*n
        if abs(rejected-round(rejected))>1e-7:raise ValueError('Cannot recover exact counts')
        row=0 if cell['rho']==1 else 1
        old[row,1]+=round(rejected);old[row,0]+=n-round(rejected)
    return old


def check(old,new):
    for key, aliases in [('accuracy',['accuracy']),('empirical_size',['empirical_size','size']),
                         ('empirical_power',['empirical_power','power'])]:
        for alias in aliases:
            if alias in old:
                if not np.isclose(old[alias],new[key],rtol=0,atol=1e-12):
                    raise AssertionError((alias,old[alias],new[key]))
                break


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify-checkpoints',action='store_true')
    args=ap.parse_args();root=args.root;rows=[];sources={};verified=[]
    def record(path,identity,old_cm,previous):
        sources[str(path.relative_to(root))]=sha(path)
        new=metrics_from_cm(old_cm[::-1,::-1]);check(previous,new)
        rows.append(dict(identity=identity,source=str(path.relative_to(root)),old_confusion_matrix=old_cm.tolist(),**new))
    official=root/'results/lstm_paper2019_extension/official_independent_test_vs_ann_20261018'
    for p in sorted(official.glob('evaluation_T*_seed*.json')):
        ev=read(p)['result'];record(p,p.stem,from_cells(ev),ev['overall'])
    # All archived fixed-T LSTM calibration evaluation files, not ANN references.
    for p in sorted((root/'results/lstm_paper2019_extension').rglob('train_epoch_*.json')):
        ev=read(p)
        if 'by_parameter' in ev:record(p,'calibration/'+str(p.parent.name),from_cells(ev),ev['overall'])
    shared=root/'results/shared_lstm_independent_evaluation_20261029'
    variable_y=np.load(shared/'variable_test/y.npy')
    for p in sorted(shared.glob('*_predictions.npy')):
        name=p.stem.removesuffix('_predictions');previous=read(shared/(name+'.json'))['metrics']
        if 'variable_T' in name:y=variable_y
        else:
            t=int(name.split('fixed_T')[1])
            with np.load(shared/f'fixed_test_arrays/T{t}.npz') as z:y=z['y']
        pred=np.load(p);old_cm=np.zeros((2,2),dtype=np.int64);np.add.at(old_cm,(y,pred),1)
        record(p,name,old_cm,previous)
    fixed=root/'results/fixed_lstm_variable_test_seed7_20261029'
    for row in read(fixed/'overall.json'):
        t=row['training_T'];p=fixed/f'T{t}_predictions.npy';pred=np.load(p)
        cm=np.zeros((2,2),dtype=np.int64);np.add.at(cm,(variable_y,pred),1)
        record(p,f'fixed_training_T{t}_variable',cm,row)
    if args.verify_checkpoints:
        device='mps' if torch.backends.mps.is_available() else 'cpu'
        for item in read(official/'test_manifest.json')['checkpoints']:
            p=root/item['path'];before=sha(p);assert before==item['sha256']
            model=Paper2019LSTMClassifier().to(device)
            model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model'])
            model.eval();t=item['T'];seed=item['model_seed']
            array_path=shared/f'fixed_test_arrays/T{t}.npz';array_hash=sha(array_path)
            with np.load(array_path) as z:raw={k:z[k] for k in ['x','y','rho','beta']}
            assert fingerprint(raw)==FINGERPRINTS[t]
            preds=[];legacy_preds=[];native=[]
            hook=model.fc.register_forward_hook(lambda module,ins,outs:native.append(outs.detach()))
            with torch.inference_mode():
                for start in range(0,len(raw['y']),256):
                    logits=model(torch.from_numpy(raw['x'][start:start+256]).to(device))
                    native_logits=native.pop()
                    torch.testing.assert_close(logits,native_logits.flip(1),rtol=0,atol=0)
                    pred=predict(logits, softmax=True);old=native_logits.softmax(1).argmax(1)
                    assert torch.equal(pred,1-old)
                    preds.append(pred.cpu().numpy());legacy_preds.append(old.cpu().numpy())
            hook.remove();pred=np.concatenate(preds)
            new=binary_metrics(1-raw['y'],pred)
            old_result=read(official/f'evaluation_T{t}_seed{seed}.json')['result']['overall'];check(old_result,new)
            assert sha(p)==before and sha(array_path)==array_hash
            verified.append(dict(T=t,seed=seed,n=len(pred),sample_decisions_identical=True,
                                 official_size_power_accuracy_match=True,checkpoint_sha256=before,
                                 test_fingerprint=FINGERPRINTS[t],device=device))
            print('Verified checkpoint',t,seed,flush=True)
    for name,digest in sources.items():assert sha(root/name)==digest
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(label_order=LABEL_ORDER,rows=rows,checkpoint_verification=verified,
                invariant_tolerance=1e-12,source_hashes=sources,source_files_unchanged=True)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Binary LSTM label migration verification','',
           'New order: stationary=0, unit_root=1. CM rows=actual, columns=predicted.',
           f'{len(rows)} archived result records reconstructed; size/power/accuracy unchanged (tolerance 1e-12).',
           f'{len(verified)} official checkpoints verified against saved test arrays; no training or data generation.',
           '', '| Result | Size % | Power % | Accuracy % | Precision ST/UR % | Recall ST/UR % | F1 ST/UR % |',
           '|---|---:|---:|---:|---|---|---|']
    for r in rows:
        if r['identity'].startswith('calibration/'):continue
        pair=lambda k:' / '.join(f'{v*100:.3f}' for v in r[k])
        lines.append(f"| {r['identity']} | {100*r['empirical_size']:.3f} | {100*r['empirical_power']:.3f} | {100*r['accuracy']:.3f} | {pair('precision')} | {pair('recall')} | {pair('f1')} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Reconstructed records:',len(rows),'verified checkpoints:',len(verified),flush=True)


if __name__=='__main__':main()
