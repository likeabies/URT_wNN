"""Three fixed-T runs: train, held-out thresholds, independent test, sequential ADF."""
import argparse
import json
import os
import traceback
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
    out.mkdir(exist_ok=False);state=dict(status='initializing',T=t,protocol=c,pid=os.getpid(),started_at=now(),completed_epoch=0)
    save(out/'run_manifest.json',state);wr=None;context={}
    def finite(ok,stage):
        if not bool(ok):
            save(out/'numeric_failure.json',dict(stage=stage,**context));raise FloatingPointError(stage)
    try:
        raw,derived=dataset(t,c['train_per_class'],c['data_seed'])
        state['train_dataset']=save_data(out/'train',raw,c['data_seed'],derived)
        w=balanced_weights(raw['d']);assert np.array_equal(w,np.array([[1.5,.75],[.75,1.5]],np.float32))
        save(out/'loss_weights.json',dict(weights=w.tolist(),stage_order=[1,2],class_order=['negative','positive']))
        wr=telemetry(wandb.init,project=c['wandb_project'],group=c['wandb_group'],name=f'I012_T{t}_ms7',config=dict(c,T=t,loss_weights=w.tolist()),mode='online')
        state.update(status='training',wandb_url=wr.url if wr else None);save(out/'run_manifest.json',state)
        set_seed(c['model_seed']);model=IntegrationOrderLSTM().to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=0.)
        weight=torch.tensor(w,device='mps');loader=DataLoader(tensors(raw),batch_size=c['batch_size'],shuffle=True)
        history=[]
        for epoch in range(1,c['epochs']+1):
            tick=perf_counter();model.train();losses=np.zeros(2);correct=np.zeros(2);count=0
            for b,(level,diff,d) in enumerate(loader,1):
                context=dict(T=t,epoch=epoch,batch=b)
                level=level.to('mps');diff=diff.to('mps');d=d.to('mps')
                optimizer.zero_grad(set_to_none=True);logits=model(level,diff);parts=loss_parts(logits,d,weight);loss=parts.sum()
                finite(torch.isfinite(loss),'nonfinite_loss');loss.backward()
                finite(torch.isfinite(torch.cat([p.grad.flatten() for p in model.parameters()])).all(),'nonfinite_gradient')
                optimizer.step()
                finite(torch.isfinite(torch.cat([p.flatten() for p in model.parameters()])).all(),'nonfinite_parameter')
                finite(torch.isfinite(torch.cat([v.flatten() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.device.type=='mps'])).all(),'nonfinite_adam_state')
                n=len(d);losses+=parts.detach().cpu().numpy()*n
                targets=torch.stack((d>=1,d>=2),1);correct+=((logits>0)==targets).sum(0).cpu().numpy();count+=n
            assert count==180000
            torch.mps.synchronize();row=dict(epoch=epoch,loss=float(losses.sum()/count),loss1=float(losses[0]/count),loss2=float(losses[1]/count),acc1=float(correct[0]/count),acc2=float(correct[1]/count),seconds=perf_counter()-tick)
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
        save(out/'final_results.json',metrics);state.update(status='complete',finished_at=now());save(out/'run_manifest.json',state)
        print(f'T={t} COMPLETE '+json.dumps(metrics),flush=True)
        return metrics
    except Exception as e:
        state.update(status='failed',error=repr(e),finished_at=now(),context=context);save(out/'run_manifest.json',state)
        (out/'failure_traceback.log').write_text(traceback.format_exc());traceback.print_exc()
        return dict(T=t,status='failed',error=repr(e),context=context)
    finally:
        if wr:telemetry(wr.finish)
        torch.mps.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    c=json.loads(Path('configs/integration_order_lstm_fixed_cycle_20261029.json').read_text())
    assert torch.backends.mps.is_available() and c['T_values']==[50,100,250] and c['epochs']==200
    assert len({c['data_seed'],c['calibration_seed'],c['test_seed']})==3
    a.output.mkdir(exist_ok=False);save(a.output/'config_snapshot.json',c)
    state=dict(status='running',pid=os.getpid(),started_at=now(),runs=[]);save(a.output/'cycle_manifest.json',state)
    for t in c['T_values']:
        state['current_T']=t;save(a.output/'cycle_manifest.json',state)
        result=one(c,t,a.output/f'T{t}');state['runs'].append(result);save(a.output/'cycle_manifest.json',state)
    state.update(status='finished',finished_at=now());save(a.output/'cycle_manifest.json',state)
    print('CYCLE FINISHED',flush=True)
if __name__=='__main__':main()
