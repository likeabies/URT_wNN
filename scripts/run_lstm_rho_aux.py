"""One T250/seed7 fixed-200 rho auxiliary ablation; no calibration."""
import csv
import hashlib
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_ann_paper2019 import dataset, fingerprint, save
from src.data import BINARY_LABELS
from src.evaluate import binary_metrics
from src.model import Paper2019LSTMClassifier
from src.lstm_rho_aux import Paper2019LSTMRhoAux
from src.train import ann_weighted_cross_entropy
from src.utils import set_seed

OUT = Path('results/lstm_rho_aux_T250_20261113')
SOURCE = Path('results/lstm_paper2019_extension/w2_calibration_range_extension/resume_20261014_215318_786719')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_source():
    candidates=json.loads((SOURCE/'calibration_candidates.json').read_text())
    selected=[r for r in candidates if r['T']==250]
    assert len(selected)==1
    w2=selected[0]['w2']
    records=json.loads((SOURCE/'integrated_records.json').read_text())
    found=[r for r in records if r['T']==250 and r['model_seed']==7 and r['w2']==w2]
    assert len(found)==1
    record=found[0];path=Path(record['path']);cfg=json.loads((path/'config_snapshot.json').read_text())
    expected=dict(model='LSTM',T=250,hidden_size=30,num_layers=1,input_size=1,output_size=2,
                  epochs=200,lr=.001,batch_size=256,dropout=0,weight_decay=0,data_seed=20191007,
                  model_seed=7,w1=1.,w2=w2,optimizer='Adam',device='mps',shuffle=True,
                  validation=False,early_stopping=False,n_train=120000)
    for k,v in expected.items():
        if cfg[k]!=v:raise ValueError(f'Control protocol mismatch {k}: {cfg[k]} != {v}')
    cp=path/'epoch_200.pt'
    checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==200
    Paper2019LSTMClassifier().load_state_dict(checkpoint['model'])
    return cfg,checkpoint,dict(checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp),
        candidate_file=str((SOURCE/'calibration_candidates.json').resolve()),
        records_file=str((SOURCE/'integrated_records.json').resolve()),selected_candidate=selected[0],
        config_snapshot=cfg,record_training_metrics=record['metrics'])


def write_csv(path, rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def atomic_checkpoint(path,payload):
    if path.exists():raise FileExistsError(path)
    tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def outputs(model,raw,aux):
    model.eval();logits=[];rhos=[];hidden=[]
    with torch.inference_mode():
        for start in range(0,len(raw['y']),256):
            x=torch.from_numpy(raw['x'][start:start+256]).to('mps')
            if aux:
                a,r,h=model(x);rhos.append(r.cpu().numpy())
            else:
                _,(h_n,_)=model.lstm(x.unsqueeze(-1));h=h_n[-1];a=model.fc(h)
            if not torch.isfinite(a).all() or not torch.isfinite(h).all():
                raise FloatingPointError('Nonfinite inference')
            logits.append(a.cpu().numpy());hidden.append(h.cpu().numpy())
    result=dict(logits=np.concatenate(logits),hidden=np.concatenate(hidden))
    if aux:
        result['rho_hat']=np.concatenate(rhos)
        if not np.isfinite(result['rho_hat']).all():raise FloatingPointError('Nonfinite rho prediction')
    return result


def classification(raw,logits,w2):
    pred=logits.argmax(1)
    overall=binary_metrics(raw['y'],pred);overall['n']=len(pred)
    overall['loss']=float(ann_weighted_cross_entropy(torch.from_numpy(logits),torch.from_numpy(raw['y']),w1=1.,w2=w2))
    cells=[];by_rho=[]
    for rho in np.unique(raw['rho']):
        mask=raw['rho']==rho
        by_rho.append(dict(rho=float(rho),n=int(mask.sum()),rejection_rate=float((pred[mask]==1).mean()),
                           interpretation='type_i_error' if rho==1 else 'power'))
        for beta in np.unique(raw['beta']):
            mask=(raw['rho']==rho)&(raw['beta']==beta)
            if mask.any():cells.append(dict(rho=float(rho),beta=float(beta),n=int(mask.sum()),
                rejection_rate=float((pred[mask]==1).mean()),interpretation='type_i_error' if rho==1 else 'power'))
    return dict(overall=overall,by_parameter=cells,by_rho=by_rho)


def rho_metrics(raw,hat,t):
    rows=[]
    for rho in np.unique(raw['rho']):
        a=hat[raw['rho']==rho].astype(np.float64);error=a-rho
        rows.append(dict(T=t,true_rho=float(rho),n=len(a),mean=float(a.mean()),median=float(np.median(a)),
            sd=float(a.std(ddof=1)),MAE=float(np.abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),bias=float(error.mean())))
    one=next(r for r in rows if r['true_rho']==1.);hard=next(r for r in rows if r['true_rho']==.99)
    return rows,dict(mean_gap=one['mean']-hard['mean'],median_gap=one['median']-hard['median'],
        overall_MAE=float(np.abs(hat.astype(np.float64)-raw['rho']).mean()),
        overall_RMSE=float(np.sqrt(np.mean((hat.astype(np.float64)-raw['rho'])**2))))


def main():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('MPS fallback must explicitly equal 0')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no CPU fallback')
    assert BINARY_LABELS==dict(unit_root=0,stationary=1)
    cfg,control_cp,provenance=control_source()
    c=json.loads(Path('configs/ann_paper2019_reproduction.json').read_text())
    assert c['data_seed']==20191007 and c['test_seed']==20261008
    assert c['rhos']==[1,.99,.95,.9,.5,.2] and c['betas']==[.99,.95,.9,.8,.7,.6,.5,.4,.3,.2,.1,0]
    raw,derived=dataset(c,250)
    assert fingerprint(raw)==cfg['data_fingerprint']
    assert len(raw['y'])==120000 and np.bincount(raw['y']).tolist()==[60000,60000]
    assert np.array_equal(raw['y'],(raw['rho']<1).astype(np.int64))
    OUT.mkdir(exist_ok=False)
    state=dict(status='initializing',pid=os.getpid(),started_at=now(),completed_epoch=0)
    save(OUT/'run_manifest.json',state)
    protocol=dict(control=cfg,lambda_rho=.1,rho_head='Linear(30,1); raw scalar; no activation',
        label_convention=BINARY_LABELS,train_T=250,test_T=[50,100,250],
        train_data_fingerprint=fingerprint(raw),derived_data_seed=derived,test_seed=c['test_seed'],
        dataset_config=c,prediction='classification logits argmax only',calibration=False,
        treatment_initialization='from scratch; seed7 base construction then rho head; restore CPU RNG after base',
        PYTORCH_ENABLE_MPS_FALLBACK='0',rho_loss='unweighted sample mean MSE over all training samples')
    save(OUT/'protocol.json',protocol);save(OUT/'control_provenance.json',provenance)
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path('src/lstm_rho_aux.py'),Path('src/model.py'),Path('src/data.py'),
              Path('src/train.py'),Path('src/evaluate.py'),Path('src/utils.py'),Path('scripts/run_ann_paper2019.py')]:
        shutil.copyfile(p,snapshot/p.name)
    save(OUT/'source_hashes.json',{str(p):digest(p) for p in snapshot.iterdir()})
    try:
        # Exactly the baseline initialization/optimizer/loader order.
        set_seed(7);model=Paper2019LSTMRhoAux().to('mps')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        loader=DataLoader(TensorDataset(torch.from_numpy(raw['x']),torch.from_numpy(raw['y']),
             torch.from_numpy(raw['rho']).float()),batch_size=256,shuffle=True)
        history=[];state['status']='training';save(OUT/'run_manifest.json',state)
        print('TRAINING FINGERPRINT VERIFIED',fingerprint(raw),'w2',cfg['w2'],flush=True)
        for epoch in range(1,201):
            torch.mps.synchronize();tick=perf_counter();model.train();sums=np.zeros(3);correct=0;n=0
            for x,y,rho in loader:
                x,y,rho=x.to('mps'),y.to('mps'),rho.to('mps')
                logits,hat,_=model(x)
                cls=ann_weighted_cross_entropy(logits,y,w1=1.,w2=cfg['w2']);mse=F.mse_loss(hat,rho)
                loss=cls+.1*mse
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
                size=len(y);sums+=np.array([loss.item(),cls.item(),mse.item()])*size
                correct+=int((logits.argmax(1)==y).sum());n+=size
            torch.mps.synchronize()
            if n!=120000 or not np.isfinite(sums).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Invalid epoch count or nonfinite state')
            row=dict(epoch=epoch,total_loss=sums[0]/n,classification_loss=sums[1]/n,rho_loss=sums[2]/n,
                     classification_accuracy=correct/n,train_seconds=perf_counter()-tick)
            history.append(row);save(OUT/'training_history.json',history)
            atomic_checkpoint(OUT/f'epoch_{epoch:03d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),protocol=protocol))
            state.update(completed_epoch=epoch,checkpoint=str(OUT/f'epoch_{epoch:03d}.pt'),last_epoch_seconds=row['train_seconds'])
            save(OUT/'run_manifest.json',state);print(json.dumps(row),flush=True)
        state['status']='evaluation';save(OUT/'run_manifest.json',state)
        train_out=outputs(model,raw,True)
        train_class=classification(raw,train_out['logits'],cfg['w2'])
        train_rho,train_gap=rho_metrics(raw,train_out['rho_hat'],250)
        save(OUT/'training_classification_metrics.json',train_class)
        save(OUT/'training_rho_metrics.json',dict(by_rho=train_rho,overall=train_gap))
        del loader,raw,train_out
        control=Paper2019LSTMClassifier().to('mps');control.load_state_dict(control_cp['model'])
        comparison={};rho_rows=[];cell_rows=[];diagnostics=[];gaps={};arrays={}
        for t in [50,100,250]:
            test,test_derived=dataset(c,t,test=True)
            fp=fingerprint(test)
            save(OUT/f'test_provenance_T{t}.json',dict(seed=c['test_seed'],derived_seed=test_derived,
                fingerprint=fp,n=len(test['y']),class_counts=np.bincount(test['y']).tolist(),
                rhos=np.unique(test['rho']).tolist(),betas=np.unique(test['beta']).tolist(),
                generator='scripts.run_ann_paper2019.dataset(c,T,test=True)',same_array_for_both_models=True))
            # Preserve generated arrays for exact downstream reuse.
            np.savez(OUT/f'test_dataset_T{t}.npz',**test)
            co=outputs(control,test,False);tr=outputs(model,test,True)
            comparison[str(t)]={}
            for name,pack in [('control',co),('rho_aux',tr)]:
                metrics=classification(test,pack['logits'],cfg['w2']);comparison[str(t)][name]=metrics
                rho_rows.extend(dict(T=t,model=name,**r) for r in metrics['by_rho'])
                cell_rows.extend(dict(T=t,model=name,**r) for r in metrics['by_parameter'])
            rr,gap=rho_metrics(test,tr['rho_hat'],t);diagnostics.extend(rr);gaps[str(t)]=gap
            arrays.update({f'T{t}_'+k:v for k,v in dict(T=np.full(len(test['y']),t,dtype=np.int64),
                true_y=test['y'],true_rho=test['rho'],beta=test['beta'],control_logits=co['logits'],
                treatment_logits=tr['logits'],rho_hat=tr['rho_hat']).items()})
            np.save(OUT/f'control_hidden_T{t}.npy',co['hidden']);np.save(OUT/f'treatment_hidden_T{t}.npy',tr['hidden'])
            assert fingerprint(test)==fp
        save(OUT/'classification_comparison.json',comparison);save(OUT/'rho_prediction_gaps.json',gaps)
        write_csv(OUT/'classification_by_rho.csv',rho_rows);write_csv(OUT/'classification_by_rho_beta.csv',cell_rows)
        write_csv(OUT/'rho_prediction_summary.csv',diagnostics);np.savez(OUT/'test_predictions.npz',**arrays)
        assert digest(Path(provenance['checkpoint']))==provenance['checkpoint_sha256']
        state.update(status='complete',finished_at=now());save(OUT/'run_manifest.json',state)
        print('RHO AUX COMPLETE',flush=True)
    except Exception as e:
        state.update(status='failed',error=repr(e),finished_at=now());save(OUT/'run_manifest.json',state)
        (OUT/'failure_traceback.log').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
