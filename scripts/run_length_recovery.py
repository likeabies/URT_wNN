"""Train the prescribed fixed-T length-recovery queue once; no calibration."""
import os,json,csv,hashlib,shutil,traceback,argparse
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader
from src.length_recovery import paired_data,short_input,Completion,Encoder,LogitStudent
from src.data import normalize_ann_series,BINARY_LABELS
from src.model import Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy
from src.utils import set_seed
from scripts.run_ann_paper2019 import save,fingerprint
from scripts.run_lstm_rho_aux import control_source,classification

OUT=None

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(m):m.requires_grad_(False);m.eval();return m

def csvfile(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def infer(model,x,fn=None):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            xb=torch.tensor(x[start:start+256],dtype=torch.float32,device='mps')
            a=fn(xb) if fn else model(xb)
            if not torch.isfinite(a).all():raise FloatingPointError('Nonfinite inference')
            parts.append(a.cpu().numpy())
    return np.concatenate(parts)

def train(model,arrays,kind,path,w2,state):
    path.mkdir(exist_ok=False);model.to('mps')
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    loader=DataLoader(TensorDataset(*[torch.from_numpy(a) for a in arrays]),batch_size=256,shuffle=True)
    history=[];local=dict(status='training',completed_epoch=0,kind=kind,started_at=now())
    save(path/'run_manifest.json',local)
    for epoch in range(1,201):
        tick=perf_counter();model.train();total=0.;n=0
        for batch in loader:
            batch=[a.to('mps') for a in batch]
            if kind=='forecast':pred=model(batch[0],batch[1]);loss=F.mse_loss(pred,batch[2])
            elif kind=='head':pred=model(batch[0]);loss=ann_weighted_cross_entropy(pred,batch[1],w1=1.,w2=w2)
            else:pred=model(batch[0]);loss=F.mse_loss(pred,batch[1])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(batch[0]);n+=len(batch[0])
        if n!=120000 or not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Invalid epoch')
        torch.mps.synchronize();row=dict(epoch=epoch,loss=total/n,seconds=perf_counter()-tick);history.append(row)
        save(path/'history.json',history)
        cp=path/f'epoch_{epoch:03d}.pt';tmp=cp.with_suffix('.tmp')
        with tmp.open('xb') as f:
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                           torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),f);f.flush();os.fsync(f.fileno())
        tmp.replace(cp);local.update(completed_epoch=epoch,checkpoint=str(cp));save(path/'run_manifest.json',local)
        state['completed_epoch']=epoch;save(OUT/'run_manifest.json',state)
        print(str(path),json.dumps(row),flush=True)
    local.update(status='complete',finished_at=now());save(path/'run_manifest.json',local);frozen(model)
    return model


def parse_args():
    p=argparse.ArgumentParser(description='Fixed-T length recovery with configurable T250 teacher and recovery seed.')
    p.add_argument('--teacher-seed',type=int,choices=[7,17,27],default=7)
    p.add_argument('--recovery-seed',type=int,choices=[7,17,27],default=7)
    p.add_argument('--out',type=Path,default=None)
    return p.parse_args()

def main():
    global OUT
    args=parse_args();teacher_seed=args.teacher_seed;recovery_seed=args.recovery_seed
    OUT=args.out or Path(f'results/length_recovery_T250_teacher{teacher_seed}_recovery{recovery_seed}_20261119')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
    assert BINARY_LABELS=={'unit_root':0,'stationary':1}
    cfg,cp,provenance=control_source(model_seed=teacher_seed);c=json.loads(Path('configs/ann_paper2019_reproduction.json').read_text())
    OUT.mkdir(exist_ok=False);snapshot=OUT/'source_snapshot';snapshot.mkdir()
    sources=['src/length_recovery.py','scripts/run_length_recovery.py','src/model.py','src/data.py','src/train.py',
             'src/utils.py','src/evaluate.py','scripts/run_ann_paper2019.py','scripts/run_lstm_rho_aux.py']
    for s in sources:shutil.copyfile(s,snapshot/Path(s).name)
    save(OUT/'source_hashes.json',{s:sha(s) for s in sources})
    save(OUT/'protocol.json',dict(control=provenance,data=c,epochs=200,batch_size=256,lr=.001,teacher_model_seed=teacher_seed,recovery_seed=recovery_seed,
        short_lengths=[50,100],teacher_frozen=True,no_calibration=True,
        forecast='raw float32 prefix + shifted raw suffix teacher forcing 1.0; raw suffix MSE; autoregressive inference',
        preprocessing='exact original normalize_ann_series separately on prefix; completion normalized after concatenation',
        method2='copy teacher encoder; hidden MSE; no projection',method3='copy teacher encoder; fresh Linear(30,2); raw logit MSE',
        heads='encoder frozen; fresh Linear(30,2), WCE selected control w2; 200 epochs each',
        paired_full_path='same 250-step source path and permutation; validated official full fingerprint',device='mps',fallback=0))
    state=dict(status='preparing',pid=os.getpid(),started_at=now(),completed=[]);save(OUT/'run_manifest.json',state)
    try:
        teacher=frozen(Paper2019LSTMClassifier().to('mps'));teacher.load_state_dict(cp['model'])
        teacher_state={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
        training,derived=paired_data(c);assert fingerprint(training)==cfg['data_fingerprint']
        test,testderived=paired_data(c,test=True)
        save(OUT/'data_provenance.json',dict(train_fingerprint=fingerprint(training),test_fingerprint=fingerprint(test),
            train_derived_seed=derived,test_derived_seed=testderived,train_n=len(training['y']),test_n=len(test['y']),
            test_rhos=np.unique(test['rho']).tolist()))
        np.savez(OUT/'paired_test_metadata.npz',y=test['y'],rho=test['rho'],beta=test['beta'])
        teacher_h=lambda x:teacher.lstm(x.unsqueeze(-1))[1][0][-1]
        targets_h=infer(teacher,training['x'],teacher_h);targets_z=infer(teacher,training['x'])
        oracle_h=infer(teacher,test['x'],teacher_h);oracle=infer(teacher,test['x']);oracle_dec=oracle.argmax(1)
        np.save(OUT/'train_h250.npy',targets_h);np.save(OUT/'train_logits250.npy',targets_z)
        np.save(OUT/'oracle_test_logits.npy',oracle)
        comparison={};rho_rows=[];forecast_rows=[];hidden_rows=[];logit_rows=[]
        for t in [50,100]:
            p=OUT/f'T{t}';p.mkdir();short=short_input(training['raw'],t);shorttest=short_input(test['raw'],t)
            comparison[str(t)]={}
            def record(name,logits):
                metrics=classification(test,logits,cfg['w2']);metrics['oracle_agreement']=float((logits.argmax(1)==oracle_dec).mean())
                comparison[str(t)][name]=metrics
                rho_rows.extend(dict(T=t,method=name,**r) for r in metrics['by_rho'])
                np.save(p/(name+'_logits.npy'),logits)
                save(OUT/'comparison.json',comparison)
            record('oracle_T250',oracle);record('direct_short_baseline',infer(teacher,shorttest))
            state.update(status='training',T=t,method='method1_forecast',completed_epoch=0);save(OUT/'run_manifest.json',state)
            set_seed(recovery_seed);forecast=Completion()
            raw=training['raw'].astype(np.float32)
            train(forecast,[raw[:,:t].copy(),raw[:,t-1:249].copy(),raw[:,t:].copy()],'forecast',p/'method1_forecast',cfg['w2'],state)
            prefix=test['raw'][:,:t].astype(np.float32)
            suffix=infer(forecast,prefix,lambda x:forecast.generate(x,250-t))
            forecast_rows.append(dict(T=t,autoregressive_raw_suffix_MSE=float(np.mean((suffix.astype(np.float64)-test['raw'][:,t:])**2))))
            completed=np.concatenate([test['raw'][:,:t],suffix],1)
            record('method1_sequence_teacher_head',infer(teacher,normalize_ann_series(completed).astype(np.float32)))
            np.save(p/'predicted_suffix.npy',suffix)
            fh=infer(forecast.encoder,raw[:,:t]);fht=infer(forecast.encoder,prefix)
            state.update(method='method1_forecast_hidden_new_head',completed_epoch=0);set_seed(recovery_seed);head=nn.Linear(30,2)
            train(head,[fh,training['y']],'head',p/'method1_forecast_hidden_new_head',cfg['w2'],state)
            record('method1_forecast_hidden_new_head',infer(head,fht));del head,fh,fht,forecast,raw,completed,suffix
            state.update(method='method2_hidden',completed_epoch=0);set_seed(recovery_seed);student=Encoder(teacher.lstm.state_dict())
            train(student,[short,targets_h],'hidden',p/'method2_hidden',cfg['w2'],state)
            h=infer(student,short);ht=infer(student,shorttest)
            hidden_rows.append(dict(T=t,hidden_state_MSE=float(np.mean((ht-oracle_h)**2))))
            record('method2_hidden_teacher_head',infer(teacher.fc,ht))
            np.save(p/'predicted_hidden.npy',ht)
            state.update(method='method2_hidden_new_head',completed_epoch=0);set_seed(recovery_seed);head=nn.Linear(30,2)
            train(head,[h,training['y']],'head',p/'method2_hidden_new_head',cfg['w2'],state)
            record('method2_hidden_new_head',infer(head,ht));del head,h,ht,student
            state.update(method='method3_logit',completed_epoch=0);set_seed(recovery_seed);student=LogitStudent(teacher.lstm.state_dict())
            train(student,[short,targets_z],'logit',p/'method3_logit',cfg['w2'],state)
            logits=infer(student,shorttest)
            logit_rows.append(dict(T=t,logit_MSE=float(np.mean((logits-oracle)**2))))
            record('method3_logit_prediction',logits)
            assert all(torch.equal(v.cpu(),teacher_state[k]) for k,v in teacher.state_dict().items())
            state['completed'].append(t);save(OUT/'run_manifest.json',state);del student;torch.mps.empty_cache()
        csvfile(OUT/'comparison_by_rho.csv',rho_rows)
        csvfile(OUT/'method1_forecast_metrics.csv',forecast_rows);csvfile(OUT/'method2_hidden_metrics.csv',hidden_rows)
        csvfile(OUT/'method3_logit_metrics.csv',logit_rows)
        assert sha(provenance['checkpoint'])==provenance['checkpoint_sha256']
        state.update(status='complete',finished_at=now());save(OUT/'run_manifest.json',state)
    except Exception as e:
        state.update(status='failed',error=repr(e),finished_at=now());save(OUT/'run_manifest.json',state)
        (OUT/'failure_traceback.log').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
