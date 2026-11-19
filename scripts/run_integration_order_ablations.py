"""Nine-run continuous ablation queue, preserving the original fixed-cycle protocol."""
import argparse
import json
import os
import traceback
import hashlib
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import wandb
from src.integration_order_lstm import IntegrationOrderLSTM,dataset,fingerprint,balanced_weights,loss_parts,threshold,classification
from src.adf_utils import adf_is_stationary
from src.utils import set_seed
from src.integration_order_ablations import ARCHITECTURES, make_model, total_loss


def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False));temp.replace(p)
def telemetry(fn,*args,**kw):
    try:return fn(*args,**kw)
    except Exception as e:print('W&B warning:',repr(e),flush=True)
def tensors(raw):return TensorDataset(*(torch.from_numpy(raw[k]) for k in ['level','difference','d']))
def save_data(p,raw,seed,derived):
    p.mkdir(exist_ok=False)
    for k,v in raw.items():np.save(p/(k+'.npy'),v)
    meta=dict(seed=seed,derived_seed=derived,n=len(raw['d']),class_counts={str(d):int((raw['d']==d).sum()) for d in np.unique(raw['d'])},fingerprint=fingerprint(raw),created_at=now())
    save(p/'dataset.json',meta);return meta

def probabilities(model,raw,batch):
    model.eval();parts=[]
    with torch.inference_mode():
        for level,diff,d in DataLoader(tensors(raw),batch_size=batch,shuffle=False):
            logits=model(level.to('mps'),diff.to('mps'))
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite evaluation logits')
            parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts)

def sequential_adf(pair):
    level,diff=pair
    if adf_is_stationary(level,.05,'n'):return 0
    return 1 if adf_is_stationary(diff,.05,'n') else 2

def one(c,t,out):
    started=perf_counter()
    out.mkdir(exist_ok=False);save(out/'config_snapshot.json',c)
    state=dict(git_sha=c['git_sha'],architecture=c['architecture'],status='initializing',T=t,protocol=c,pid=os.getpid(),started_at=now(),completed_epoch=0)
    save(out/'run_manifest.json',state);wr=None;context={}
    def finite(ok,stage):
        if not bool(ok):
            save(out/'numeric_failure.json',dict(stage=stage,**context));raise FloatingPointError(stage)
    try:
        raw,derived=dataset(t,c['train_per_class'],c['data_seed'])
        state['train_dataset']=save_data(out/'train',raw,c['data_seed'],derived)
        w=balanced_weights(raw['d']);assert np.array_equal(w,np.array([[1.5,.75],[.75,1.5]],np.float32))
        save(out/'loss_weights.json',dict(weights=w.tolist(),stage_order=[1,2],class_order=['negative','positive']))
        wr=telemetry(wandb.init,project=c['wandb_project'],group=c['wandb_group'],name=f"{c['architecture']}_T{t}_ms7",config=dict(c,T=t,loss_weights=w.tolist()),mode='online')
        state.update(status='training',wandb_url=wr.url if wr else None);save(out/'run_manifest.json',state)
        set_seed(c['model_seed']);model=make_model(c['architecture']).to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=0.)
        state['parameter_count']=sum(p.numel() for p in model.parameters());save(out/'run_manifest.json',state)
        weight=torch.tensor(w,device='mps');loader=DataLoader(tensors(raw),batch_size=c['batch_size'],shuffle=True)
        history=[]
        for epoch in range(1,c['epochs']+1):
            tick=perf_counter();model.train();losses=np.zeros(2);correct=np.zeros(2);count=0
            for b,(level,diff,d) in enumerate(loader,1):
                context=dict(T=t,epoch=epoch,batch=b)
                level=level.to('mps');diff=diff.to('mps');d=d.to('mps')
                optimizer.zero_grad(set_to_none=True);logits=model(level,diff);parts=loss_parts(logits,d,weight);loss=total_loss(parts,c['stage2_weight'])
                finite(torch.isfinite(loss),'nonfinite_loss');loss.backward()
                finite(torch.isfinite(torch.cat([p.grad.flatten() for p in model.parameters()])).all(),'nonfinite_gradient')
                optimizer.step()
                finite(torch.isfinite(torch.cat([p.flatten() for p in model.parameters()])).all(),'nonfinite_parameter')
                finite(torch.isfinite(torch.cat([v.flatten() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.device.type=='mps'])).all(),'nonfinite_adam_state')
                n=len(d);losses+=parts.detach().cpu().numpy()*n
                targets=torch.stack((d>=1,d>=2),1);correct+=((logits>0)==targets).sum(0).cpu().numpy();count+=n
            assert count==180000
            torch.mps.synchronize();row=dict(epoch=epoch,loss=float((losses[0]+c['stage2_weight']*losses[1])/count),loss1=float(losses[0]/count),loss2=float(losses[1]/count),acc1=float(correct[0]/count),acc2=float(correct[1]/count),seconds=perf_counter()-tick)
            history.append(row);save(out/'history.json',history)
            cp=out/f'epoch_{epoch:03d}.pt';temp=cp.with_suffix('.tmp');assert not cp.exists()
            with temp.open('xb') as f:
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),protocol=c,T=t),f);f.flush();os.fsync(f.fileno())
            temp.replace(cp)
            state.update(completed_epoch=epoch,checkpoint=str(cp),last_epoch_seconds=row['seconds']);save(out/'run_manifest.json',state)
            if wr:telemetry(wr.log,row)
            print(f'T={t} '+json.dumps(row),flush=True)
        del loader,raw
        state['status']='calibration';save(out/'run_manifest.json',state)
        cal,derived=dataset(t,c['calibration_per_class'],c['calibration_seed'],orders=(0,1))
        state['calibration_dataset']=save_data(out/'calibration',cal,c['calibration_seed'],derived)
        probs=probabilities(model,cal,c['batch_size']);np.save(out/'calibration_probabilities.npy',probs)
        c1=threshold(probs[cal['d']==0,0]);c2=threshold(probs[cal['d']==1,1])
        selected=dict(c1=c1,c2=c2,selected_at=now(),cohort='held-out calibration only',test_accessed=False)
        save(out/'thresholds.json',selected)
        del cal,probs
        state['status']='independent_test';save(out/'run_manifest.json',state)
        test,derived=dataset(t,c['test_per_class'],c['test_seed'])
        state['test_dataset']=save_data(out/'test',test,c['test_seed'],derived)
        assert len({state[k]['fingerprint'] for k in ['train_dataset','calibration_dataset','test_dataset']})==3
        probs=probabilities(model,test,c['batch_size']);np.save(out/'test_probabilities.npy',probs)
        reject1=probs[:,0]>c1['threshold'];reject2=probs[:,1]>c2['threshold'];d=test['d']
        pred=np.where(reject1,np.where(reject2,2,1),0);np.save(out/'test_predictions.npy',pred)
        metrics=dict(T=t,c1=c1['threshold'],c2=c2['threshold'],stage1_size=float(reject1[d==0].mean()),stage1_I1_power=float(reject1[d==1].mean()),stage1_I2_power=float(reject1[d==2].mean()),stage2_size=float(reject2[d==1].mean()),stage2_I2_power=float(reject2[d==2].mean()),neural=classification(d,pred))
        save(out/'neural_test.json',metrics)
        if wr:telemetry(wr.log,{k:v for k,v in metrics.items() if isinstance(v,(float,int))})
        state['status']='adf_benchmark';save(out/'run_manifest.json',state)
        with ProcessPoolExecutor(max_workers=4) as pool:
            adf=np.fromiter(pool.map(sequential_adf,zip(test['level'],test['difference']),chunksize=128),dtype=np.int64,count=len(d))
        np.save(out/'adf_predictions.npy',adf);metrics['adf']=classification(d,adf);save(out/'adf_test.json',metrics['adf'])
        assert fingerprint(test)==state['test_dataset']['fingerprint']
        save(out/'final_results.json',metrics);state.update(status='complete',finished_at=now(),runtime_seconds=perf_counter()-started);save(out/'run_manifest.json',state)
        print(f'T={t} COMPLETE '+json.dumps(metrics),flush=True)
        return metrics
    except Exception as e:
        state.update(status='failed',failure_kind='numerical' if isinstance(e,FloatingPointError) else 'infrastructure_or_code',failure_stage=state['status'],error=repr(e),finished_at=now(),runtime_seconds=perf_counter()-started,context=context);save(out/'run_manifest.json',state)
        (out/'failure_traceback.log').write_text(traceback.format_exc());traceback.print_exc()
        if isinstance(e,FloatingPointError) and not (out/'numeric_failure.json').exists():
            save(out/'numeric_failure.json',dict(stage=state['failure_stage'],error=repr(e),**context))
        if not isinstance(e,FloatingPointError):raise
        return dict(T=t,status='failed',error=repr(e),context=context)
    finally:
        if wr:telemetry(wr.finish)
        torch.mps.empty_cache()


def comparison(root, references):
    rows=[]
    for architecture,path in list(references.items())+[(a,str(root/a)) for a in ARCHITECTURES]:
        for t in (50,100,250):
            folder=Path(path)/f'T{t}';m=json.loads((folder/'run_manifest.json').read_text())
            row=dict(architecture=architecture,T=t,status=m['status'],source=str(folder),
                     parameter_count=m.get('parameter_count'),wandb_url=m.get('wandb_url'))
            if m['status']=='complete':
                assert m['completed_epoch']==200
                r=json.loads((folder/'final_results.json').read_text())
                row.update({k:r[k] for k in ['c1','c2','stage1_size','stage1_I1_power','stage1_I2_power','stage2_size','stage2_I2_power']})
                row.update(neural=r['neural'],adf=r['adf'])
                cm=np.array(r['neural']['confusion_matrix']);n=cm[1].sum()
                row.update(I1_to_I0=float(cm[1,0]/n),I1_to_I2=float(cm[1,2]/n),
                           I1_to_I0_count=int(cm[1,0]),I1_to_I2_count=int(cm[1,2]),
                           accuracy=r['neural']['accuracy'])
                for k,v in enumerate(r['neural']['recall']):row[f'I{k}_recall']=v
                # Matched cohorts are required for comparisons; no test-driven selection.
                base=Path(references['original_chained'])/f'T{t}'
                for split in ('train','calibration','test'):
                    a=json.loads((folder/split/'dataset.json').read_text())
                    b=json.loads((base/split/'dataset.json').read_text())
                    assert a['fingerprint']==b['fingerprint'], (architecture,t,split)
                assert json.loads((folder/'thresholds.json').read_text())['selected_at'] < json.loads((folder/'test/dataset.json').read_text())['created_at']
            else:row.update(error=m.get('error'),failure_stage=m.get('failure_stage'),context=m.get('context'))
            rows.append(row)
    keys=['stage1_I1_power','I1_to_I0','I1_to_I2','I0_recall','I1_recall','I2_recall','accuracy']
    for row in rows:
        if row['status']!='complete':continue
        row['deltas']={}
        for ref in references:
            base=next(x for x in rows if x['architecture']==ref and x['T']==row['T'])
            if base['status']=='complete':row['deltas'][ref]={k:row[k]-base[k] for k in keys}
    save(root/'comparison.json',dict(rows=rows,units='rates; deltas are absolute rate differences',
         independent_encoders_capacity_note='Two separate hidden30 encoders; increased model capacity vs shared encoder.'))
    lines=['# Integration-order ablations','',
           'Rates in percent; deltas in percentage points. Independent encoders have increased model capacity.', '',
           '| Architecture | T | Status | Stage1 I1 detection | I1 → I0 | I1 → I2 | I0 recall | I1 recall | I2 recall | Accuracy |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[f"{100*row[k]:.3f}" if k in row else '—' for k in keys]
        lines.append('| '+' | '.join([row['architecture'],str(row['T']),row['status']]+values)+' |')
    for ref in references:
        lines+=['',f'## Change vs {ref}', '', '| Architecture | T | Δ Stage1 I1 | Δ I1 → I0 | Δ I1 → I2 | Δ I0 recall | Δ I1 recall | Δ I2 recall | Δ Accuracy |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if ref in row.get('deltas',{}):lines.append('| '+' | '.join([row['architecture'],str(row['T'])]+[f"{100*row['deltas'][ref][k]:+.3f}" for k in keys])+' |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def execute_queue(c, root, references, run_one=one):
    queue=[(a,t) for a in ARCHITECTURES for t in (50,100,250)]
    state=dict(status='running',pid=os.getpid(),started_at=now(),queue=queue,runs=[],git_sha=c['git_sha'])
    save(root/'queue_manifest.json',state)
    try:
        for architecture,t in queue:
            dest=root/architecture;dest.mkdir(exist_ok=True)
            cfg=dict(c,architecture=architecture,stage2_weight=.5 if architecture=='reduced_l2' else 1.,
                     wandb_group=root.name+'_'+architecture)
            cfg['loss']='balanced L1 + 0.5 * balanced L2' if architecture=='reduced_l2' else 'balanced L1 + balanced L2'
            state.update(current_architecture=architecture,current_T=t);save(root/'queue_manifest.json',state)
            result=run_one(cfg,t,dest/f'T{t}')
            state['runs'].append(dict(architecture=architecture,T=t,result=result))
            save(root/'queue_manifest.json',state)
        comparison(root,references)
        state.update(status='finished',finished_at=now());save(root/'queue_manifest.json',state)
    except Exception as exc:
        state.update(status='blocked',error=repr(exc),finished_at=now());save(root/'queue_manifest.json',state)
        (root/'queue_traceback.log').write_text(traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    c=json.loads(Path('configs/integration_order_lstm_fixed_cycle_20261029.json').read_text())
    assert torch.backends.mps.is_available()
    assert (c['epochs'],c['batch_size'],c['model_seed'])==(200,256,7)
    refs=dict(original_chained=str(Path('results/integration_order_lstm_fixed_cycle_20261029').resolve()),
              independent_head=str(Path('.worktrees/integration-order-independent/results/integration_order_lstm_independent_fixed_cycle_20261103').resolve()))
    for path in refs.values():
        for t in (50,100,250):
            m=json.loads((Path(path)/f'T{t}/run_manifest.json').read_text())
            assert m['status']=='complete' and m['completed_epoch']==200
    root.mkdir(parents=True,exist_ok=False)
    c['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    c['lambda_provenance']='No agreed lambda found in local source/config/notes; requested default 0.5, no search'
    sources=[__file__,'src/integration_order_ablations.py','src/integration_order_lstm.py','src/data.py','src/adf_utils.py','src/utils.py',
             'configs/integration_order_lstm_fixed_cycle_20261029.json','tests/test_integration_order_ablations.py']
    hashes={}
    for name in sources:
        path=Path(name).resolve();relative=path.relative_to(Path.cwd());target=root/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(root/'provenance.json',dict(git_sha=c['git_sha'],source_hashes=hashes,references=refs))
    save(root/'config_snapshot.json',c)
    execute_queue(c,root,refs)


if __name__=='__main__':main()
