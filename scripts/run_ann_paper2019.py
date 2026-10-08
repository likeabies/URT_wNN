"""Separate fixed-epoch training-only calibration; no validation or ADF calls."""
import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_ann_dataset
from src.evaluate import evaluate_ann
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy, train_one_epoch
from src.utils import set_seed


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(raw):
    h = hashlib.sha256()
    for k in ['x', 'y', 'rho', 'beta']:
        h.update(k.encode()); h.update(np.ascontiguousarray(raw[k]).tobytes())
    return h.hexdigest()


def dataset(c, t, test=False, smoke=False):
    # Distinct primary streams; no prior validation or test samples reused.
    seed = int(np.random.SeedSequence([c['test_seed'] if test else c['data_seed'], t]).generate_state(1)[0])
    rhos = sorted(set(c['rhos'] + ([0.8] if test else [])), reverse=True)
    n0, n1 = ((2, 2) if test else (10, 2)) if smoke else ((1000, 1000) if test else (5000, 1000))
    return make_ann_dataset(t, rhos, c['betas'], n0, n1, seed), seed


def evaluate(model, raw, criterion):
    # DataLoader evaluation must not perturb subsequent training shuffle RNG.
    state = torch.get_rng_state(); mps_state = torch.mps.get_rng_state()
    try:
        result = evaluate_ann(model, TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y'])), raw, criterion, batch_size=256, device='mps')
    finally:
        torch.set_rng_state(state); torch.mps.set_rng_state(mps_state)
    assert all(math.isfinite(v) for v in result['overall'].values())
    return result


def train(c, base, stage, t, seed, raw, w2, epochs, lr=.001, batch=256, checkpoints=None):
    name = f"{c['family']}_{stage}_T{t}_w2-{w2:.10g}_ms-{seed}_lr-{lr}_bs-{batch}"
    p = base / name; p.mkdir(exist_ok=False)
    cfg = dict(family=c['family'], stage=stage, T=t, model_seed=seed, w1=1., w2=w2,
               epochs=epochs, optimizer='Adam', lr=lr, batch_size=batch, dropout=0,
               weight_decay=0, shuffle=True, device='mps', validation=False,
               data_seed=c['data_seed'], data_fingerprint=fingerprint(raw), n_train=len(raw['y']))
    save(p/'config_snapshot.json', cfg)
    criterion = partial(ann_weighted_cross_entropy, w1=1., w2=w2)
    checkpoints = checkpoints or [epochs]
    with wandb.init(project='URT_wNN', group=c['family'], name=name, config=cfg, mode='online') as run:
        set_seed(seed)
        model = ANNClassifier(t, 20 if t == 50 else 50).to('mps')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
        loader = DataLoader(TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y'])), batch_size=batch, shuffle=True)
        history=[]; evaluations={}; start=perf_counter()
        for epoch in range(1, epochs+1):
            tick=perf_counter()
            loss,acc=train_one_epoch(model,loader,criterion,optimizer,'mps')
            if not math.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            rec=dict(epoch=epoch,train_loss=loss,train_acc=acc,train_seconds=perf_counter()-tick)
            history.append(rec); run.log(rec)
            if epoch in checkpoints:
                ev=evaluate(model,raw,criterion)
                evaluations[str(epoch)]=ev
                save(p/f'train_epoch_{epoch}.json',ev)
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                            'torch_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state()},p/f'epoch_{epoch}.pt')
                run.log({'checkpoint_epoch':epoch,'checkpoint_train':ev['overall']})
                print(name, 'checkpoint',epoch,ev['overall'],flush=True)
            save(p/'history.json',history)
        info=dict(name=name,path=str(p),T=t,model_seed=seed,w2=w2,lr=lr,batch=batch,
                  evaluations=evaluations,seconds=perf_counter()-start,id=run.id,url=run.url,status='finished')
        save(p/'summary.json',info)
    return info


def select_epoch(c, runs):
    checks=[]
    for epoch in c['epoch_candidates']:
        worst_loss=0.; worst_rate=0.
        for r in runs:
            origin=r['evaluations'][str(epoch)]['overall']
            for later in c['checkpoints']:
                if later <= epoch: continue
                v=r['evaluations'][str(later)]['overall']
                worst_loss=max(worst_loss,abs(v['loss']-origin['loss'])/max(origin['loss'],1e-12))
                worst_rate=max(worst_rate,*[abs(v[k]-origin[k]) for k in ['accuracy','empirical_size','empirical_power']])
        ok=worst_loss<=c['stability_relative_loss'] and worst_rate<=c['stability_absolute_rate']
        checks.append(dict(epoch=epoch,worst_relative_loss_change=worst_loss,worst_absolute_rate_change=worst_rate,stable=ok))
        if ok: return epoch,checks
    return None,checks


def polynomial(c, x, y):
    brackets=[(float(a),float(b)) for a,b,ya,yb in zip(x,x[1:],y,y[1:]) if (ya-.05)*(yb-.05)<=0]
    result={'observed_w2':list(x),'mean_train_type_i':list(y),'empirical_brackets':brackets,'fits':{}}
    for degree in [2,3]:
        coef=np.polyfit(x,y,degree); shifted=coef.copy(); shifted[-1]-=.05
        roots=np.roots(shifted)
        valid=[float(z.real) for z in roots if abs(z.imag)<1e-8 and any(lo<=z.real<=hi for lo,hi in brackets)]
        result['fits'][str(degree)]={'coefficients_descending':coef.tolist(),'roots_real_imag':[[float(z.real),float(z.imag)] for z in roots],
                                    'valid_roots':valid,'rmse':float(np.sqrt(np.mean((np.polyval(coef,x)-y)**2)))}
    a=result['fits']['2']['valid_roots']; b=result['fits']['3']['valid_roots']
    result['acceptable']=len(a)==len(b)==1
    if result['acceptable']:
        result['root_difference']=abs(a[0]-b[0])
        result['tolerance']=max(c['polynomial_absolute_root_difference'],c['polynomial_relative_root_difference']*abs(a[0]))
        result['acceptable']=result['root_difference']<=result['tolerance']
    if result['acceptable']: result['chosen_w2']=a[0]
    return result


def runtime(c,epoch,diagnostic_seeds=3):
    speed=c['estimated_train_seconds_per_epoch']; t100=speed['100']
    # 3 diagnostics to 300; 12 new sensitivity runs (baseline reused); 78 new
    # calibration runs (T100/w2=1 checkpoints reused); 9 fresh final fits.
    seconds=900*t100 + epoch*(4.5*diagnostic_seeds*t100 + 27*sum(speed.values())-3*t100 + 3*sum(speed.values()))
    return {'fixed_epoch':epoch,'planned_new_ANN_runs':90+4*diagnostic_seeds,'sensitivity_seeds_per_alternative':diagnostic_seeds,'logical_calibration_runs':81,
            'estimated_hours_with_10pct_overhead':seconds*1.10/3600}


def workflow(c,base,state):
    def status(stage,**extra):
        state.update(stage=stage,**extra); save(base/'workflow_status.json',state)
    raw100,_=dataset(c,100)
    status('convergence_running',runtime_scenarios=[runtime(c,e) for e in c['epoch_candidates']])
    diagnostics=[]
    for seed in c['model_seeds']:
        diagnostics.append(train(c,base,'convergence',100,seed,raw100,1.,300,checkpoints=c['checkpoints']))
        status('convergence_running',convergence_completed=len(diagnostics))
    epoch,checks=select_epoch(c,diagnostics)
    save(base/'convergence_decision.json',{'selected_epoch':epoch,'checks':checks,'runs':diagnostics})
    if epoch is None:
        status('needs_attention',reason='Neither 150 nor 200 epochs meets the documented convergence checks; no arbitrary 300-epoch selection.'); return
    # Adjust the old speed estimate using the newly observed diagnostic speed.
    measured=[h['train_seconds'] for r in diagnostics for h in json.loads((Path(r['path'])/'history.json').read_text())]
    ratio=float(np.mean(measured))/c['estimated_train_seconds_per_epoch']['100']
    effective=dict(c,estimated_train_seconds_per_epoch={t:v*ratio for t,v in c['estimated_train_seconds_per_epoch'].items()})
    sensitivity_seeds=list(c['model_seeds'])
    estimate=runtime(effective,epoch,len(sensitivity_seeds))
    while estimate['estimated_hours_with_10pct_overhead']>c['runtime_budget_hours'] and len(sensitivity_seeds)>1:
        sensitivity_seeds=sensitivity_seeds[:-1]
        estimate=runtime(effective,epoch,len(sensitivity_seeds))
    status('sensitivity_running',selected_epoch=epoch,runtime=estimate)
    if estimate['estimated_hours_with_10pct_overhead']>c['runtime_budget_hours']:
        status('needs_attention',reason='Runtime estimate still exceeds 16h after reducing alternative sensitivity settings to one seed; paper sample and calibration grid preserved.'); return
    baseline={r['model_seed']:r['evaluations'][str(epoch)]['overall'] for r in diagnostics}
    sensitivity=[]
    for lr,batch in [(.0005,256),(.002,256),(.001,128),(.001,512)]:
        for seed in sensitivity_seeds:
            r=train(c,base,'sensitivity',100,seed,raw100,1.,epoch,lr,batch)
            v=r['evaluations'][str(epoch)]['overall']; b=baseline[seed]
            sensitivity.append(dict(run=r,relative_loss_difference=abs(v['loss']-b['loss'])/b['loss'],
                max_rate_difference=max(abs(v[k]-b[k]) for k in ['empirical_size','empirical_power'])))
    unstable=any(s['relative_loss_difference']>c['sensitivity_relative_loss'] or s['max_rate_difference']>c['sensitivity_absolute_rate'] for s in sensitivity)
    save(base/'sensitivity_decision.json',dict(baseline=baseline,model_seeds=sensitivity_seeds,comparisons=sensitivity,major_instability=unstable))
    if unstable:
        status('needs_attention',reason='Sensitivity screen exceeds documented major-change threshold; calibration not launched.'); return
    calibration=[]
    for t in c['T_values']:
        raw=raw100 if t==100 else dataset(c,t)[0]
        for w in c['w2_values']:
            for seed in c['model_seeds']:
                if t==100 and w==1.:
                    r=next(r for r in diagnostics if r['model_seed']==seed)
                    entry=dict(T=t,w2=w,model_seed=seed,source_run=r['name'],reused_checkpoint=True,evaluation=r['evaluations'][str(epoch)])
                else:
                    r=train(c,base,'calibration',t,seed,raw,w,epoch)
                    entry=dict(T=t,w2=w,model_seed=seed,source_run=r['name'],reused_checkpoint=False,evaluation=r['evaluations'][str(epoch)])
                calibration.append(entry); save(base/'calibration.json',calibration)
                status('calibration_running',calibration_completed=len(calibration))
    fits={}
    for t in c['T_values']:
        x=c['w2_values']; y=[float(np.mean([r['evaluation']['overall']['empirical_size'] for r in calibration if r['T']==t and r['w2']==w])) for w in x]
        fits[str(t)]=polynomial(c,x,y)
    save(base/'polynomial_comparison.json',fits)
    if not all(v['acceptable'] for v in fits.values()):
        status('needs_attention',reason='Degree-2/3 roots differ materially or lack a unique root within an empirical 0.05 bracket. No averaging or test generation.'); return
    # Freeze weights for all T BEFORE generating or reading any test data.
    save(base/'selected_weights.json',{t:f['chosen_w2'] for t,f in fits.items()})
    final=[]
    for t in c['T_values']:
        raw=raw100 if t==100 else dataset(c,t)[0]
        w=fits[str(t)]['chosen_w2']
        test,test_seed=dataset(c,t,test=True)
        save(base/f'test_provenance_T{t}.json',dict(seed=test_seed,fingerprint=fingerprint(test),n=len(test['y']),generated_after_calibration=True))
        for seed in c['model_seeds']:
            r=train(c,base,'final',t,seed,raw,w,epoch)
            model=ANNClassifier(t,20 if t==50 else 50).to('mps')
            ckpt=torch.load(Path(r['path'])/f'epoch_{epoch}.pt',map_location='mps',weights_only=True)
            model.load_state_dict(ckpt['model'])
            ev=evaluate(model,test,partial(ann_weighted_cross_entropy,w1=1.,w2=w))
            save(Path(r['path'])/'evaluation_test.json',ev)
            with wandb.init(project='URT_wNN',id=r['id'],resume='must',mode='online') as run:
                run.summary['test']=ev
            final.append(dict(T=t,model_seed=seed,w2=w,training=r['evaluations'][str(epoch)],test=ev))
            save(base/'final_results.json',final); status('final_running',final_completed=len(final))
    aggregates=[]
    for t in c['T_values']:
        rr=[r for r in final if r['T']==t]
        for split in ['training','test']:
            params=rr[0][split]['by_parameter']
            for cell in params:
                rates=[next(v['rejection_rate'] for v in r[split]['by_parameter'] if v['rho']==cell['rho'] and v['beta']==cell['beta']) for r in rr]
                aggregates.append(dict(T=t,split=split,rho=cell['rho'],beta=cell['beta'],n_per_seed=cell['n'],
                    mean=float(np.mean(rates)),sd=float(np.std(rates,ddof=1)),min=min(rates),max=max(rates),
                    paper_reported_cell=cell['rho'] in [1,.95,.9,.8] and cell['beta'] in [.9,.6,.3,0]))
    save(base/'parameter_summary.json',aggregates)
    status('complete',ADF='Deferred: existing implementation is AIC-autolag, not paper GtS.')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    c=json.loads(Path('configs/ann_paper2019_reproduction.json').read_text())
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS is required; CPU fallback is forbidden')
    if args.smoke: c['family']+='_smoke'
    base=Path('results')/c['family']; base.mkdir(exist_ok=False)
    save(base/'protocol.json',c); state={'stage':'initializing','family':c['family']}
    try:
        if args.smoke:
            for t in c['T_values']:
                raw,_=dataset(c,t,smoke=True)
                assert len(raw['y'])==240 and (raw['y']==0).sum()==120
                r=train(c,base,'smoke',t,7,raw,1.,2,checkpoints=[1,2])
                assert set(r['evaluations'])=={'1','2'}
                ckpt=torch.load(Path(r['path'])/'epoch_2.pt',map_location='mps',weights_only=True)
                assert ckpt['epoch']==2
            x=c['w2_values']; y=[.01+.1*w for w in x]
            p=polynomial(c,x,y); assert p['acceptable'] and abs(p['chosen_w2']-.4)<1e-8
            assert not polynomial(c,x,[.2]*len(x))['acceptable']
            state['stage']='smoke_passed'; save(base/'workflow_status.json',state)
            print('SMOKE PASSED: three T values, checkpoint reload, polynomial root and no-bracket guards.',flush=True)
        else: workflow(c,base,state)
    except Exception as exc:
        state.update(stage='failed',error=repr(exc)); save(base/'workflow_status.json',state); raise


if __name__=='__main__': main()
