"""Method1 head-only cohort/distribution ablation. No test training/calibration."""
import os,json,csv,hashlib,traceback,shutil
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.length_recovery import paired_data,Completion
from src.data import make_ann_dataset,normalize_ann_series
from src.model import Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy
from src.utils import set_seed
from scripts.run_ann_paper2019 import save,fingerprint
from scripts.run_lstm_rho_aux import classification

ROOT=Path('results/length_recovery_method1_head_ft_20261119')
BASE=Path('results/length_recovery_T250_20261118')
B_SEED=20261119

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze(m):m.requires_grad_(False);m.eval();return m

def cohort_b(c):
    seed=int(np.random.SeedSequence([B_SEED,250]).generate_state(1)[0]);rng=np.random.default_rng(seed)
    rhos=sorted(c['rhos'],reverse=True);raw=[]
    for rho in rhos:
        for beta in c['betas']:
            n=500 if rho==1 else 100;e=rng.normal(size=(n,251));innov=beta*e[:,:-1]+e[:,1:]
            values=np.empty((n,250),np.float64);previous=np.zeros(n)
            for t in range(250):previous=rho*previous+innov[:,t];values[:,t]=previous
            raw.append(values)
    a=np.concatenate(raw);order=rng.permutation(len(a));a=a[order]
    result=make_ann_dataset(250,rhos,c['betas'],500,100,seed)
    np.testing.assert_array_equal(result['x'],normalize_ann_series(a).astype(np.float32))
    result['raw']=a;assert np.bincount(result['y']).tolist()==[6000,6000]
    return result,seed

def infer(x,fn):
    values=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            v=fn(torch.tensor(x[start:start+256],device='mps',dtype=torch.float32))
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite frozen inference')
            values.append(v.cpu().numpy())
    return np.concatenate(values)

def real_hidden(teacher,x):return infer(x,lambda v:teacher.lstm(v.unsqueeze(-1))[1][0][-1])

def reconstructed_hidden(teacher,forecast,raw,t,suffix=None):
    if suffix is None:suffix=infer(raw[:,:t],lambda v:forecast.generate(v,250-t))
    full=np.concatenate([raw[:,:t],suffix],1)
    return real_hidden(teacher,normalize_ann_series(full).astype(np.float32))

def head_fit(initial,h,y,path,w2):
    assert len(h)==len(y) and h.shape[1]==30
    path.mkdir();set_seed(7);head=nn.Linear(30,2).to('mps');head.load_state_dict(initial)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=0)
    loader=DataLoader(TensorDataset(torch.from_numpy(h),torch.from_numpy(y)),batch_size=256,shuffle=True)
    hist=[];state=dict(status='training',n=len(y),completed_epoch=0,updates=0,started_at=now())
    save(path/'run_manifest.json',state)
    for epoch in range(1,201):
        tick=perf_counter();total=0.;n=0;head.train()
        for x,label in loader:
            x,label=x.to('mps'),label.to('mps');opt.zero_grad()
            loss=ann_weighted_cross_entropy(head(x),label,w1=1.,w2=w2)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
            loss.backward();opt.step();total+=loss.item()*len(x);n+=len(x);state['updates']+=1
        assert n==len(y) and all(torch.isfinite(p).all() for p in head.parameters())
        hist.append(dict(epoch=epoch,loss=total/n,seconds=perf_counter()-tick));save(path/'history.json',hist)
        state['completed_epoch']=epoch;save(path/'run_manifest.json',state)
        if epoch%50==0:print(path,'epoch',epoch,flush=True)
    tmp=path/'checkpoint.tmp'
    torch.save(dict(model=head.state_dict(),optimizer=opt.state_dict(),epoch=200,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()),tmp)
    tmp.replace(path/'epoch_200.pt');state.update(status='complete',finished_at=now());save(path/'run_manifest.json',state)
    return freeze(head)

def main():
    assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    protocol=json.loads((BASE/'protocol.json').read_text());provenance=json.loads((BASE/'data_provenance.json').read_text())
    c=protocol['data'];w2=protocol['control']['config_snapshot']['w2']
    assert B_SEED not in [c['data_seed'],c['test_seed']]
    ROOT.mkdir(exist_ok=False);snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for source in [Path(__file__),Path('src/length_recovery.py'),Path('src/data.py'),Path('src/train.py')]:shutil.copyfile(source,snapshot/source.name)
    settings=dict(baseline=str(BASE.resolve()),teacher_checkpoint=protocol['control']['checkpoint'],cohort_B_seed=B_SEED,
        epochs=200,lr=.001,batch=256,seed=7,optimizer='fresh Adam per independent condition; weight_decay=0',w1=1.,w2=w2,
        frozen='teacher encoder; forecast encoder and decoder; only Linear(30,2) optimized',
        real_representation_1_2='same frozen forecast encoder applied to raw real250; final hidden, no other encoder mixed',
        method1_1=dict(distribution_only='A reconstructed250 only',new_cohort_only='B real250 only',both='B real250 + B reconstructed250'),
        method1_2=dict(distribution_only='A prefix forecast encoder hidden + A raw real250 forecast encoder hidden',new_cohort_only='B prefix forecast encoder hidden',both='B prefix hidden + B raw real250 hidden'),
        fixed_epochs_not_fixed_updates=True,test_used_for_training=False,calibration=False,decision='raw logits argmax')
    save(ROOT/'protocol.json',settings)
    state=dict(status='preparing',pid=os.getpid(),started_at=now(),completed=[]);save(ROOT/'run_manifest.json',state)
    try:
        A,_=paired_data(c);B,derived=cohort_b(c);test,_=paired_data(c,test=True)
        assert fingerprint(A)==provenance['train_fingerprint'] and fingerprint(test)==provenance['test_fingerprint']
        assert len({fingerprint(p) for p in [A,B,test]})==3
        np.savez(ROOT/'cohort_B.npz',**B)
        save(ROOT/'cohort_provenance.json',dict(B_seed=B_SEED,B_derived_seed=derived,B_n=len(B['y']),B_fingerprint=fingerprint(B),
            A_fingerprint=fingerprint(A),test_fingerprint=fingerprint(test)))
        teacher_path=Path(settings['teacher_checkpoint']);hashes={str(teacher_path):sha(teacher_path)}
        teacher=Paper2019LSTMClassifier().to('mps');teacher.load_state_dict(torch.load(teacher_path,map_location='cpu',weights_only=True)['model']);freeze(teacher)
        initial_teacher={k:v.cpu().clone() for k,v in teacher.fc.state_dict().items()}
        Breal=real_hidden(teacher,B['x']);baseline=json.loads((BASE/'comparison.json').read_text())
        comparison={};rows=[]
        for t in [50,100]:
            p=ROOT/f'T{t}';p.mkdir();comparison[str(t)]={}
            forecast_path=BASE/f'T{t}/method1_forecast/epoch_200.pt';head_path=BASE/f'T{t}/method1_forecast_hidden_new_head/epoch_200.pt'
            hashes[str(forecast_path)]=sha(forecast_path);hashes[str(head_path)]=sha(head_path)
            forecast=Completion().to('mps');forecast.load_state_dict(torch.load(forecast_path,map_location='cpu',weights_only=True)['model']);freeze(forecast)
            state.update(T=t,status='extracting_features');save(ROOT/'run_manifest.json',state)
            print('EXTRACTING',t,flush=True)
            Arec=reconstructed_hidden(teacher,forecast,A['raw'],t)
            Brec=reconstructed_hidden(teacher,forecast,B['raw'],t)
            test_rec=reconstructed_hidden(teacher,forecast,test['raw'],t,np.load(BASE/f'T{t}/predicted_suffix.npy'))
            Aprefix=infer(A['raw'][:,:t],forecast.encoder);Afull=infer(A['raw'],forecast.encoder)
            Bprefix=infer(B['raw'][:,:t],forecast.encoder);Bfull=infer(B['raw'],forecast.encoder)
            test_prefix=infer(test['raw'][:,:t],forecast.encoder)
            old_head=torch.load(head_path,map_location='cpu',weights_only=True)['model']
            for method,initial,features,test_features,old_name in [
                ('method1_1',initial_teacher,[(Arec,A['y']),(Breal,B['y']),(np.concatenate([Breal,Brec]),np.tile(B['y'],2))],test_rec,'method1_sequence_teacher_head'),
                ('method1_2',old_head,[(np.concatenate([Aprefix,Afull]),np.tile(A['y'],2)),(Bprefix,B['y']),(np.concatenate([Bprefix,Bfull]),np.tile(B['y'],2))],test_prefix,'method1_forecast_hidden_new_head')]:
                existing=nn.Linear(30,2).to('mps');existing.load_state_dict(initial);freeze(existing)
                before=infer(test_features,existing);saved=np.load(BASE/f'T{t}/{old_name}_logits.npy')
                np.testing.assert_allclose(before,saved,rtol=1e-5,atol=2e-6)
                comparison[str(t)][method]={'baseline':baseline[str(t)][old_name]}
                for r in baseline[str(t)][old_name]['by_rho']:rows.append(dict(T=t,method=method,condition='baseline',**r))
                for condition,(h,y) in zip(['distribution_only','new_cohort_only','both'],features):
                    state.update(status='training',method=method,condition=condition);save(ROOT/'run_manifest.json',state)
                    directory=p/f'{method}_{condition}'
                    head=head_fit(initial,h,y,directory,w2)
                    logits=infer(test_features,head);metrics=classification(test,logits,w2)
                    comparison[str(t)][method][condition]=metrics;save(directory/'metrics.json',metrics)
                    np.save(directory/'test_logits.npy',logits)
                    rows.extend(dict(T=t,method=method,condition=condition,**r) for r in metrics['by_rho'])
                    state['completed'].append(f'T{t}/{method}/{condition}');save(ROOT/'run_manifest.json',state)
                    save(ROOT/'comparison.json',comparison)
            del forecast,Arec,Brec,test_rec,Aprefix,Afull,Bprefix,Bfull,test_prefix
            torch.mps.empty_cache()
        with (ROOT/'comparison_by_rho.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        assert all(sha(p)==h for p,h in hashes.items());save(ROOT/'frozen_source_checkpoint_hashes.json',hashes)
        state.update(status='complete',finished_at=now());save(ROOT/'run_manifest.json',state);print('ALL COMPLETE',flush=True)
    except Exception as e:
        state.update(status='failed',error=repr(e),finished_at=now());save(ROOT/'run_manifest.json',state)
        (ROOT/'failure_traceback.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
