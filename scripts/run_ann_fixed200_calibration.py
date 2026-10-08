"""Fixed-200 training-only coarse -> fine calibration. No diagnostic/test gates."""
import argparse
import csv
import json
import math
import statistics as st
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.run_ann_paper2019 import dataset, fingerprint, save, train


def local_grid(points, target=.05):
    """Choose one observed adjacent interval, never extrapolate."""
    points = sorted(points, key=lambda p: p['w2'])
    if len(points) < 2 or any(not math.isfinite(p['mean_type_i_error']) for p in points):
        raise ValueError('Need at least two finite calibration points')
    nearest = min(range(len(points)), key=lambda i: (abs(points[i]['mean_type_i_error']-target), points[i]['w2']))
    crossings = [i for i in range(len(points)-1)
                 if (points[i]['mean_type_i_error']-target)*(points[i+1]['mean_type_i_error']-target) <= 0]
    def distance(i):
        return (min(abs(points[j]['mean_type_i_error']-target) for j in [i,i+1]),
                sum(abs(points[j]['mean_type_i_error']-target) for j in [i,i+1]), i)
    if crossings:
        adjacent = [i for i in crossings if nearest in [i,i+1]]
        chosen = min(adjacent or crossings, key=distance)
        reason = 'Observed mean Type I error brackets target'
    else:
        adjacent = [i for i in [nearest-1,nearest] if 0 <= i < len(points)-1]
        chosen = min(adjacent, key=distance)
        reason = 'No observed crossing; closest-point adjacent interval only, no extrapolation'
    lo, hi = points[chosen]['w2'], points[chosen+1]['w2']
    grid = [round(float(v), 10) for v in np.linspace(lo, hi, 5)]
    return dict(interval=[lo,hi], w2_values=grid, bracketed=bool(crossings), reason=reason,
                closest_coarse_w2=points[nearest]['w2'])


def summarize(records, stage):
    result=[]
    for t,w in sorted({(r['T'],r['w2']) for r in records}):
        rs=[r for r in records if (r['T'],r['w2'])==(t,w)]
        size=[r['metrics']['empirical_size'] for r in rs]
        power=[r['metrics']['empirical_power'] for r in rs]
        result.append(dict(stage=stage,T=t,w2=w,n=len(rs),mean_type_i_error=st.mean(size),
            sd_type_i_error=st.stdev(size),mean_power=st.mean(power),sd_power=st.stdev(power)))
    return result


def write_csv(path, rows):
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def candidates(rows, records, target):
    result=[]
    for t in sorted({r['T'] for r in rows}):
        best=min((r for r in rows if r['T']==t),key=lambda r:(abs(r['mean_type_i_error']-target),r['w2']))
        rs=sorted([r for r in records if (r['T'],r['w2'])==(t,best['w2'])],key=lambda r:r['model_seed'])
        betas=sorted({b['beta'] for r in rs for b in r['beta_type_i_error']})
        beta_summary=[]
        for beta in betas:
            vals=[next(b['rejection_rate'] for b in r['beta_type_i_error'] if b['beta']==beta) for r in rs]
            beta_summary.append(dict(beta=beta,mean_type_i_error=st.mean(vals),sd_type_i_error=st.stdev(vals)))
        result.append(dict(**best,label='fine_grid_candidate',final=False,
            absolute_error=abs(best['mean_type_i_error']-target),beta_type_i_error=beta_summary,
            per_seed=[dict(model_seed=r['model_seed'],type_i_error=r['metrics']['empirical_size'],power=r['metrics']['empirical_power'],beta_type_i_error=r['beta_type_i_error']) for r in rs]))
    return result


def execute(c, base, state):
    raw_by_t={}
    for t in c['T_values']:
        raw,derived_seed=dataset(c,t)
        if len(raw['y'])!=120000 or int((raw['y']==0).sum())!=60000:
            raise RuntimeError('Unexpected training composition')
        for rho in c['rhos']:
            for beta in c['betas']:
                if int(((raw['rho']==rho)&(raw['beta']==beta)).sum()) != (5000 if rho==1 else 1000):
                    raise RuntimeError('Unexpected training cell count')
        raw_by_t[t]=raw
        state['datasets'][str(t)] = dict(data_seed=c['data_seed'],derived_seed=derived_seed,
                                        fingerprint=fingerprint(raw),n=120000,unit_root=60000,stationary=60000)
    save(base/'run_manifest.json',state)
    def one(stage,t,w,seed):
        raw=raw_by_t[t]
        info=train(c,base,stage,t,seed,raw,w,200,lr=.001,batch=256)
        # A failed/corrupt/nonfinite checkpoint is a genuine execution error.
        checkpoint=torch.load(Path(info['path'])/'epoch_200.pt',map_location='cpu',weights_only=True)
        if checkpoint['epoch']!=200 or not all(torch.isfinite(v).all().item() for v in checkpoint['model'].values()):
            raise RuntimeError('Invalid final checkpoint')
        ev=info['evaluations']['200']
        if not all(math.isfinite(v) for v in ev['overall'].values()):
            raise RuntimeError('Nonfinite final metrics')
        if any(not math.isfinite(v['rejection_rate']) for v in ev['by_parameter']):
            raise RuntimeError('Nonfinite cell rejection rate')
        record=dict(stage=stage,T=t,w2=w,model_seed=seed,data_seed=c['data_seed'],
            derived_data_seed=state['datasets'][str(t)]['derived_seed'],data_fingerprint=state['datasets'][str(t)]['fingerprint'],
            epochs=200,metrics=ev['overall'],beta_type_i_error=[v for v in ev['by_parameter'] if v['rho']==1],
            by_parameter=ev['by_parameter'],runtime_seconds=info['seconds'],path=info['path'],
            wandb_id=info['id'],wandb_url=info['url'],status='finished')
        save(Path(info['path'])/'calibration_record.json',record)
        state['runs'].append(record); save(base/'run_manifest.json',state)
        return record
    state['status']='coarse_running'
    for t in c['T_values']:
        for w in c['coarse_w2']:
            for seed in c['model_seeds']: one('coarse',t,w,seed)
    coarse=summarize(state['runs'],'coarse'); write_csv(base/'coarse_summary.csv',coarse)
    plans={str(t):local_grid([p for p in coarse if p['T']==t],c['target']) for t in c['T_values']}
    state.update(status='fine_running',fine_plans=plans)
    save(base/'fine_plan.json',plans); save(base/'run_manifest.json',state)
    existing={(r['T'],r['w2'],r['model_seed']):r for r in state['runs']}
    fine=[]
    for t in c['T_values']:
        for w in plans[str(t)]['w2_values']:
            for seed in c['model_seeds']:
                key=(t,w,seed)
                if key in existing:
                    record=dict(existing[key],stage='fine',reused=True,source_stage=existing[key]['stage'])
                else:
                    record=dict(one('fine',t,w,seed),reused=False,source_stage='fine')
                    existing[key]=record
                fine.append(record)
    assert len(state['runs'])==99 and len(existing)==99 and len(fine)==45
    fine_summary=summarize(fine,'fine'); write_csv(base/'fine_summary.csv',fine_summary)
    save(base/'fine_records.json',fine)
    save(base/'calibration_candidates.json',candidates(fine_summary,fine,c['target']))
    state.update(status='complete',new_runs_completed=len(state['runs']),fine_reused_results=sum(r['reused'] for r in fine))
    save(base/'run_manifest.json',state)
    print('Completed 72 coarse + 27 new fine runs; 18 coarse endpoints reused. No independent test.',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dry-run',action='store_true'); args=parser.parse_args()
    cfg=json.loads(Path('configs/ann_fixed200_w2_calibration.json').read_text())
    c=json.loads(Path(cfg['base_config']).read_text()); c.update(cfg)
    assert c['epochs']==200 and c['fine_points']==5 and c['T_values']==[50,100,250]
    assert c['model_seeds']==[7,17,27] and c['coarse_w2']==[.2,.25,.3,.35,.4,.45,.5,.55]
    if args.dry_run:
        print(json.dumps(dict(epochs=200,coarse_runs=72,max_new_fine_runs=27,total_new_runs=99,
            training_only=True,data_seed=c['data_seed'],model_seeds=c['model_seeds'],coarse_w2=c['coarse_w2'],
            output_dir=c['output_dir'],wandb_group=c['family']),indent=2)); return
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required; no CPU fallback')
    base=Path(c['output_dir']); base.mkdir(parents=True,exist_ok=False)
    save(base/'protocol.json',c)
    state=dict(status='initializing',family=c['family'],epochs=200,expected_coarse_runs=72,max_new_fine_runs=27,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),datasets={},runs=[])
    save(base/'run_manifest.json',state); start=perf_counter()
    try:
        execute(c,base,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc)); save(base/'run_manifest.json',state); raise
    finally:
        state['total_runtime_seconds']=perf_counter()-start; save(base/'run_manifest.json',state)


if __name__=='__main__': main()
