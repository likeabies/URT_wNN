"""Continue frozen CPU training-grid checkpoints; no selection or new weights."""
import argparse, copy, fcntl, json, math, os, random, statistics as st, time, traceback, uuid
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
from scripts.run_rnn_paper2019_fixed200 import (read, sha, require, immutable, atomic_checkpoint, save,
    csv_save, slopes, evaluate, dataset, fingerprint, make_ann_dataset, seed_for,
    Paper2019RNNClassifier, train_one_epoch, ann_weighted_cross_entropy)

CONFIG=Path('configs/rnn_training_grid_continuation_20261027.json')

def rng():
    return dict(torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state())

def restore_rng(c):
    torch.set_rng_state(c['torch_rng']);random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng'])

def restore(c):
    model=Paper2019RNNClassifier().cpu()
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);restore_rng(c)
    return model,opt

def pack(model,opt,history,identity):
    return dict(identity=identity,epoch=len(history),history=history,model=model.state_dict(),
                optimizer=opt.state_dict(),mps_rng=None,**rng())

def schedule(source):
    manifest=read(source/'run_manifest.json');jobs=[]
    require(manifest['status']=='complete','Source workflow incomplete')
    for t in (100,250):
        grid=read(source/'training_grid'/f'T{t}.json')
        for point in sorted(grid['points'].values(),key=lambda x:x['w2']):
            require(sorted(x['model_seed'] for x in point['per_seed'])==[7,17,27],'Seed coverage')
            for row in sorted(point['per_seed'],key=lambda x:x['model_seed']):
                r=manifest['runs'][row['fit_key']];cp=Path(r['checkpoint'])
                require(r['T']==t and r['w2']==point['w2'] and r['purpose']=='grid','Wrong source')
                require(r['status']=='finished' and r['epochs']==200 and sha(cp)==r['checkpoint_sha256'],'Source integrity')
                c=read(source/'fits'/r['key']/'config.json');require(c['device']=='cpu','CPU source required')
                jobs.append(dict(key=r['key'],T=t,w2=r['w2'],model_seed=r['model_seed'],
                    source_checkpoint=str(cp.resolve()),source_sha256=r['checkpoint_sha256']))
    require(len(jobs)==66 and len({j['key'] for j in jobs})==66,'Expected 66 unique training-grid fits')
    return jobs,manifest

def snapshot(folder,epoch,model,history,job,train,test):
    path=folder/f'metrics_{epoch}.json'
    if path.exists():return read(path)
    state=rng();mode=model.training
    try:
        a=evaluate(model,train,job['w2'],'cpu');b=evaluate(model,test,job['w2'],'cpu')
        row=dict(T=job['T'],w2=job['w2'],model_seed=job['model_seed'],epoch=epoch,
          loss=history[-1]['train_loss'],**slopes(history),calibration_size=a['overall']['empirical_size'],
          calibration_loss=a['overall']['loss'],test_size=b['overall']['empirical_size'],
          test_power=b['overall']['empirical_power'],
          **{f'power_rho_{rho}':st.mean(x['rejection_rate'] for x in b['by_parameter'] if x['rho']==rho) for rho in (.95,.9,.8)})
        save(path,dict(**row,checkpoint_sha256=sha(folder/f'epoch_{epoch}.pt')))
        return row
    finally:restore_rng(state);model.train(mode)

def summaries(base,jobs):
    rows=[read(base/'fits'/j['key']/f'metrics_{e}.json') for j in jobs for e in (200,300,400)]
    csv_save(base/'metrics_by_seed.csv',rows)
    metrics=['loss','last_50_loss_slope','last_20_loss_slope','calibration_size','calibration_loss','test_size','test_power','power_rho_0.95','power_rho_0.9','power_rho_0.8'];agg=[]
    for t in (100,250):
        for w in sorted({j['w2'] for j in jobs if j['T']==t}):
            for e in (200,300,400):
                rr=[r for r in rows if (r['T'],r['w2'],r['epoch'])==(t,w,e)];require(len(rr)==3,'Missing metrics')
                agg.append(dict(T=t,w2=w,epoch=e,**{f'{k}_{f}':fn(r[k] for r in rr) for k in metrics for f,fn in [('mean',st.mean),('sd',st.stdev)]}))
    csv_save(base/'summary.csv',agg);diagnostics=[]
    for t in (100,250):
        for e in (200,300,400):
            rr=[r for r in agg if r['T']==t and r['epoch']==e]
            reversals=[dict(a=a['w2'],b=b['w2'],change=b['calibration_size_mean']-a['calibration_size_mean']) for a,b in zip(rr,rr[1:]) if b['calibration_size_mean']<a['calibration_size_mean']]
            raw=[r for r in rows if r['T']==t and r['epoch']==e]
            diagnostics.append(dict(T=t,epoch=e,reversals=reversals,reversal_count=len(reversals),
              mean_calibration_size_sd=st.mean(r['calibration_size_sd'] for r in rr),
              mean_test_size_sd=st.mean(r['test_size_sd'] for r in rr),mean_test_power_sd=st.mean(r['test_power_sd'] for r in rr),
              mean_abs_last50_slope=st.mean(abs(r['last_50_loss_slope']) for r in raw),
              mean_abs_last20_slope=st.mean(abs(r['last_20_loss_slope']) for r in raw)))
    save(base/'diagnostics.json',diagnostics)
    report=['# RNN training-grid continuation','', 'No model/weight/epoch selection: test metrics are diagnostic only. Original 200-epoch checkpoints are unchanged.','', '|T|w2|epoch|Calibration size mean±SD|Test size mean±SD|Test power mean±SD|last50 slope mean±SD|','|---|---|---|---|---|---|---|']
    for r in agg:
        report.append('|'+ '|'.join([str(r['T']),str(r['w2']),str(r['epoch'])]+[f"{r[k+'_mean']:.6g} ± {r[k+'_sd']:.6g}" for k in ['calibration_size','test_size','test_power','last_50_loss_slope']])+'|')
    (base/'report.md').write_text('\n'.join(report)+'\n')

def smoke():
    torch.manual_seed(71);x=torch.randn(12,100);y=torch.arange(12)%2
    model=Paper2019RNNClassifier();opt=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(TensorDataset(x,y),batch_size=4,shuffle=True);criterion=partial(ann_weighted_cross_entropy,w1=1.,w2=.45)
    h=[]
    for e in range(1,3):
        loss,acc=train_one_epoch(model,loader,criterion,opt,'cpu');h.append(dict(epoch=e,train_loss=loss,train_acc=acc))
    cp=copy.deepcopy(pack(model,opt,h,'smoke'))
    train_one_epoch(model,loader,criterion,opt,'cpu');expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict());expected_rng=rng()
    resumed,ropt=restore(cp);train_one_epoch(resumed,loader,criterion,ropt,'cpu')
    require(all(torch.equal(v,resumed.state_dict()[k]) for k,v in expected.items()),'Resume model mismatch')
    for k,v in expected_opt['state'].items():
        for name,value in v.items():require(torch.equal(value,ropt.state_dict()['state'][k][name]),'Resume optimizer mismatch')
    require(torch.equal(expected_rng['torch_rng'],torch.get_rng_state()),'Resume RNG mismatch')
    print('SMOKE PASS: exact model/Adam/RNG resume equivalence',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:smoke();return
    c=read(CONFIG);source=Path(c['source']);base=Path(c['output']);base.mkdir(parents=True,exist_ok=True)
    with (base/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs,old=schedule(source);scientific=read(source/'protocol.json')['config']
        protocol=dict(config=c,jobs=jobs,source_manifest_sha256=sha(source/'run_manifest.json'),
          source_hashes={str(f):sha(f) for f in [CONFIG,Path(__file__),Path('scripts/run_rnn_paper2019_fixed200.py'),Path('src/train.py'),Path('src/paper2019_rnn.py'),Path('src/data.py'),Path('src/evaluate.py')]},test_selection=False)
        immutable(base/'protocol.json',protocol)
        for j in jobs:
            cp=torch.load(j['source_checkpoint'],map_location='cpu',weights_only=False)
            require(cp['epoch']==200 and len(cp['history'])==200,'Source epoch/history')
            require(all(k in cp for k in ('optimizer','torch_rng','numpy_rng','python_rng')),'Missing resume state')
        if args.preflight:print('PREFLIGHT PASS: 66 source checkpoints and hashes, epochs, RNG/optimizer states');return
        state=read(base/'run_manifest.json') if (base/'run_manifest.json').exists() else dict(runs={},errors=[])
        state.update(pid=os.getpid(),status='running',total_fits=len(jobs))
        def persist():state['updated_at']=time.time();save(base/'run_manifest.json',state)
        persist()
        try:
            for t in (100,250):
                train,_=dataset(scientific,t,smoke=False)
                test=make_ann_dataset(t,[1.,.95,.9,.8],[.9,.6,.3,0.],1000,1000,seed_for(scientific['test_seed'],t))
                require(fingerprint(train)==old['datasets'][f'training_T{t}']['fingerprint'],'Training fingerprint')
                require(fingerprint(test)==old['datasets'][f'test_T{t}']['fingerprint'],'Test fingerprint')
                for job in [j for j in jobs if j['T']==t]:
                    folder=base/'fits'/job['key'];folder.mkdir(parents=True,exist_ok=True)
                    if job['key'] in state['runs']:
                        require(sha(folder/'epoch_400.pt')==state['runs'][job['key']]['checkpoint_sha256'],'Finished hash mismatch');continue
                    source_cp=torch.load(job['source_checkpoint'],map_location='cpu',weights_only=False)
                    if not (folder/'epoch_200.pt').exists():
                        import shutil
                        shutil.copyfile(job['source_checkpoint'],folder/'epoch_200.pt')
                    require(sha(folder/'epoch_200.pt')==job['source_sha256'],'Baseline changed')
                    # Recover a milestone if interrupted between latest and milestone writes.
                    if (folder/'latest.pt').exists():
                        latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
                        if latest['epoch'] in (300,400) and not (folder/f"epoch_{latest['epoch']}.pt").exists():
                            atomic_checkpoint(folder/f"epoch_{latest['epoch']}.pt",latest)
                    # Recover any milestone evaluation interrupted before it was saved.
                    for e in (200,300,400):
                        if (folder/f'epoch_{e}.pt').exists() and not (folder/f'metrics_{e}.json').exists():
                            cp=torch.load(folder/f'epoch_{e}.pt',weights_only=False,map_location='cpu');model,opt=restore(cp)
                            snapshot(folder,e,model,cp['history'],job,train,test)
                    require(abs(read(folder/'metrics_200.json')['calibration_size']-old['runs'][job['key']]['training']['overall']['empirical_size'])<1e-12,'Baseline calibration mismatch')
                    path=folder/'latest.pt';cp=torch.load(path if path.exists() else folder/'epoch_200.pt',map_location='cpu',weights_only=False)
                    model,opt=restore(cp);history=cp['history'];identity=cp['identity']
                    require(cp['epoch']==len(history) and 200<=len(history)<=400,'Resume epoch mismatch')
                    loader=DataLoader(TensorDataset(torch.from_numpy(train['x']),torch.from_numpy(train['y'])),batch_size=256,shuffle=True)
                    rid_path=folder/'wandb_id.json';rid=read(rid_path)['id'] if rid_path.exists() else uuid.uuid4().hex[:8];immutable(rid_path,dict(id=rid))
                    before=rng()
                    with wandb.init(project=scientific['project'],entity=scientific['entity'],group=c['group'],name='continue_'+job['key'],id=rid,resume='allow',mode='online',config=dict(**job,device='cpu',start_epoch=200,end_epoch=400,lr=.001,batch_size=256,test_diagnostic_only=True)) as run:
                        restore_rng(before)
                        for epoch in range(len(history)+1,401):
                            tick=time.perf_counter();loss,acc=train_one_epoch(model,loader,partial(ann_weighted_cross_entropy,w1=1.,w2=job['w2']),opt,'cpu')
                            require(math.isfinite(loss) and all(torch.isfinite(v).all() for v in model.parameters()),'Nonfinite optimization')
                            history.append(dict(epoch=epoch,train_loss=loss,train_acc=acc,seconds=time.perf_counter()-tick))
                            checkpoint=pack(model,opt,history,identity);atomic_checkpoint(path,checkpoint)
                            if epoch in (300,400):
                                atomic_checkpoint(folder/f'epoch_{epoch}.pt',checkpoint);snapshot(folder,epoch,model,history,job,train,test)
                            save(folder/'history.json',history);run.log(history[-1],step=epoch)
                            state.update(active_run=job['key'],active_epoch=epoch);persist()
                            if epoch==201 or epoch%10==0:print(job['key'],history[-1],flush=True)
                        run.summary['completed_epoch']=400
                    state['runs'][job['key']]=dict(**job,epoch=400,status='finished',checkpoint_sha256=sha(folder/'epoch_400.pt'));persist()
                    require(sha(Path(job['source_checkpoint']))==job['source_sha256'],'Original checkpoint changed')
            summaries(base,jobs);state['status']='complete';persist();print('COMPLETE 66 continuation fits',flush=True)
        except BaseException:
            state['status']='failed';state['errors'].append(traceback.format_exc());persist();raise

if __name__=='__main__':main()
