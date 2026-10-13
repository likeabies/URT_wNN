"""Supplement: fixed-200 ANN training, held-out null selection, fresh test only after selection."""
import csv
import json
import math
import statistics as st
import subprocess
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset
from src.data import make_ann_dataset
from src.evaluate import evaluate_ann
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy
from scripts.run_ann_paper2019 import dataset, fingerprint, save, train
from scripts.run_ann_independent_test import (preflight as original_preflight, compare, paper_values,
                                            sha, require, RHOS, BETAS, SEEDS, SELECTED as OLD_W2)

ROOT=Path('results/ann_paper2019_reproduction')
OUT=ROOT/'fixed200_validation_calibration'
GROUP='ann_fixed200_validation_calibration'
ARTIFACT='ann_fixed200_validation_calibration_results'
INITIAL={50:[.32,.37],100:[.32,.40],250:[.40,.52]}
TRAIN_SEED=20191007
VAL_SEED=20261013
OLD_TEST_SEED=20261008
NEW_TEST_SEED=20261014
TARGET=.05
SAFETY=(.05,1.)
VAL_BETAS=[.99,.95,.9,.8,.7,.6,.5,.4,.3,.2,.1,0.]


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def derived(seed,t):return int(np.random.SeedSequence([seed,t]).generate_state(1)[0])
def weight(w):return round(float(w),10)
def write_csv(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def point_summary(records):
    require(sorted(r['model_seed'] for r in records)==SEEDS,'Three model seeds required')
    values=[r['validation_size'] for r in records]
    beta=[dict(beta=b,mean_size=st.mean(next(v['rejection_rate'] for v in r['validation_by_beta'] if v['beta']==b) for r in records),
               sd_size=st.stdev(next(v['rejection_rate'] for v in r['validation_by_beta'] if v['beta']==b) for r in records)) for b in VAL_BETAS]
    return dict(w2=records[0]['w2'],mean_size=st.mean(values),sd_size=st.stdev(values),
                per_seed=[dict(model_seed=r['model_seed'],size=r['validation_size']) for r in records],beta_specific=beta)


def bracket(points):
    ordered=sorted(points.values(),key=lambda p:p['w2'])
    intervals=[(a['w2'],b['w2']) for a,b in zip(ordered,ordered[1:]) if (a['mean_size']-TARGET)*(b['mean_size']-TARGET)<=0]
    return list(intervals[0]) if intervals else None


def next_action(points,current,confirmed=False):
    """Use observed direction, one point at a time; diagnose ambiguous slopes once."""
    found=bracket(points)
    if found:return dict(action='fine',interval=found)
    a,b=current;lo=points[a]['mean_size'];hi=points[b]['mean_size'];delta=hi-lo
    ordered=sorted(points.values(),key=lambda p:p['w2'])
    diffs=[q['mean_size']-p['mean_size'] for p,q in zip(ordered,ordered[1:])]
    reversal=any(d*delta<0 and abs(d)>=.0005 for d in diffs)
    ambiguous=abs(delta)<.0005 or reversal
    if ambiguous and not confirmed:
        mid=weight((a+b)/2)
        if mid in points:return dict(action='stop',reason='Unstable direction after available midpoint evidence')
        return dict(action='diagnostic',w2=mid,reason='Endpoint slope <0.0005 or observed reversal')
    if ambiguous:
        mid=weight((a+b)/2);v=points[mid]['mean_size']
        monotone=(lo<=v<=hi) or (lo>=v>=hi)
        if not monotone or abs(delta)<1e-6:return dict(action='stop',reason='Midpoint did not resolve unstable direction')
    # Below target: move along positive response; above target: move against it.
    upward=(lo<TARGET and delta>0) or (lo>TARGET and delta<0)
    d=b-a;new=weight(min(SAFETY[1],b+d) if upward else max(SAFETY[0],a-d))
    if new in points:return dict(action='stop',reason='Safety boundary or previously visited point reached without bracket')
    return dict(action='extend',w2=new,interval=[b,new] if upward else [new,a],observed_slope=delta/d)


def choose_candidate(points,fine):
    return min((points[w] for w in fine),key=lambda p:(abs(p['mean_size']-TARGET),p['w2']))


def evaluate(raw,item):
    require(sha(item['checkpoint'])==item['checkpoint_sha256'],'Checkpoint changed')
    t=item['T'];model=ANNClassifier(t,20 if t==50 else 50).to('mps')
    cp=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
    require(cp['epoch']==200 and all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Invalid model checkpoint')
    model.load_state_dict(cp['model'],strict=True)
    return evaluate_ann(model,TensorDataset(torch.from_numpy(raw['x']),torch.from_numpy(raw['y'])),raw,
                        partial(ann_weighted_cross_entropy,w1=1.,w2=item['w2']),batch_size=256,device='mps')


def old_index():
    original_preflight()  # Read-only verification of scientific source and historical fixed candidates.
    old_test=read(ROOT/'independent_test/test_manifest.json')
    require(old_test['status']=='complete','Original independent evaluation not complete')
    index={}
    for dirname in ['w2_calibration','w2_calibration_t250_extension']:
        manifest=read(ROOT/dirname/'run_manifest.json')
        require(manifest['status']=='complete','Historical calibration incomplete')
        for r in manifest['runs']:
            index[(r['T'],weight(r['w2']),r['model_seed'])]=r
    return index


def validate_reuse(r,fp):
    p=Path(r['path']);c=read(p/'config_snapshot.json')
    expected=dict(T=r['T'],w2=r['w2'],model_seed=r['model_seed'],epochs=200,data_seed=TRAIN_SEED,
                  lr=.001,batch_size=256,optimizer='Adam',dropout=0,weight_decay=0,validation=False,w1=1.)
    require(all(c[k]==v for k,v in expected.items()),'Reuse configuration mismatch')
    require(r['status']=='finished' and r['data_fingerprint']==c['data_fingerprint']==fp,'Reuse provenance mismatch')
    require(read(p/'summary.json')['status']=='finished','Reuse run summary incomplete')
    cp=torch.load(p/'epoch_200.pt',map_location='cpu',weights_only=True)
    require(cp['epoch']==200 and all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Reuse checkpoint invalid')
    ANNClassifier(r['T'],20 if r['T']==50 else 50).load_state_dict(cp['model'],strict=True)
    return dict(checkpoint=str(p/'epoch_200.pt'),checkpoint_sha256=sha(p/'epoch_200.pt'),source_run=str(p),source_wandb_id=r['wandb_id'])


def record_config(t,w,seed,stage,reused):
    return dict(T=t,w2=w,model_seed=seed,epochs=200,training_seed=TRAIN_SEED,validation_seed=VAL_SEED,
                stage=stage,reused_checkpoint=reused,validation_used_in_training=False,early_stopping=False)


def calibrate_t(t,c,raw,val,index,state):
    records={};points={};stages={};train_fp=fingerprint(raw);val_fp=fingerprint(val)
    def persist():
        save(OUT/'run_manifest.json',state);save(OUT/'bracket_history.json',state['bracket_history'])
        endpoint=[];fine_rows=[]
        for tt,packs in state['points'].items():
            for p in packs:
                row=dict(T=int(tt),w2=p['w2'],stage=p['stage'],mean_validation_size=p['mean_size'],sd_validation_size=p['sd_size'],
                    absolute_error=abs(p['mean_size']-TARGET),per_seed=json.dumps(p['per_seed']),beta_specific=json.dumps(p['beta_specific']))
                (fine_rows if p['stage']=='fine' else endpoint).append(row)
        write_csv(OUT/'endpoint_results.csv',endpoint);write_csv(OUT/'fine_grid_results.csv',fine_rows)
    def point(w,stage):
        w=weight(w)
        if w in points:return points[w]
        rs=[]
        for seed in SEEDS:
            key=(t,w,seed);reused=key in index
            cfg=record_config(t,w,seed,stage,reused)
            if reused:
                item=dict(T=t,w2=w,model_seed=seed,**validate_reuse(index[key],train_fp))
                resume={}
            else:
                # Only the original training arrays enter the unchanged training function.
                info=train(c,OUT/'training_runs',stage,t,seed,raw,w,200,lr=.001,batch=256)
                item=dict(T=t,w2=w,model_seed=seed,checkpoint=str(Path(info['path'])/'epoch_200.pt'),
                          checkpoint_sha256=sha(Path(info['path'])/'epoch_200.pt'),source_run=info['path'],source_wandb_id=info['id'])
                resume=dict(id=info['id'],resume='must')
            with wandb.init(project='URT_wNN',group=GROUP,job_type='calibration-evaluation',
                name=f'heldout_{stage}_T{t}_w{w}_seed{seed}',config=cfg,mode='online',**resume) as wr:
                ev=evaluate(val,item);size=ev['overall']['empirical_size']
                require(math.isfinite(size) and len(ev['by_parameter'])==12,'Invalid validation result')
                require(all(v['n']==5000 and v['rho']==1 for v in ev['by_parameter']),'Invalid validation composition')
                wr.log({'validation_size':size,'validation_by_beta':wandb.Table(columns=['beta','n','size'],data=[[v['beta'],v['n'],v['rejection_rate']] for v in ev['by_parameter']])})
                item.update(stage=stage,reused_checkpoint=reused,epochs=200,training_seed=TRAIN_SEED,validation_seed=VAL_SEED,training_fingerprint=train_fp,validation_fingerprint=val_fp,
                    validation_size=size,validation_by_beta=ev['by_parameter'],evaluation_wandb_id=wr.id,evaluation_wandb_url=wr.url,status='finished')
            rs.append(item);state['records'].append(item);persist()
        records[w]=rs;points[w]=point_summary(rs);stages[w]=stage
        state['points'][str(t)].append(dict(points[w],stage=stage));persist()
        print('validation',t,w,points[w]['mean_size'],flush=True)
        return points[w]
    current=[weight(w) for w in INITIAL[t]]
    for w in current:point(w,'endpoint')
    confirmed=False
    for _ in range(50):
        action=next_action(points,current,confirmed)
        state['bracket_history'].append(dict(T=t,current_interval=current,endpoint_sizes=[points[w]['mean_size'] for w in current],**action));persist()
        if action['action']=='stop':
            state.update(status='needs_attention',reason=action['reason'],blocked_T=t);persist();return None
        if action['action']=='fine':break
        point(action['w2'],'bracket_extension')
        if action['action']=='diagnostic':confirmed=True
        else:current=action['interval'];confirmed=False
    else:raise RuntimeError('Adaptive search exceeded finite-step safety limit')
    interval=action['interval'];fine=[weight(w) for w in np.linspace(*interval,5)]
    state.update(status='fine_running',current_T=t);persist()
    for w in fine:point(w,'fine')
    # Complete fine table includes reused endpoints; never count them as new evaluations.
    fine_entries=[]
    for tt,existing in state.get('fine_points',{}).items():fine_entries.extend(existing)
    fine_entries.extend([dict(T=t,**points[w]) for w in fine])
    state.setdefault('fine_points',{})[str(t)]=[dict(T=t,**points[w]) for w in fine]
    write_csv(OUT/'fine_grid_results.csv',[dict(T=r['T'],w2=r['w2'],mean_validation_size=r['mean_size'],sd_validation_size=r['sd_size'],
        absolute_error=abs(r['mean_size']-TARGET),per_seed=json.dumps(r['per_seed']),beta_specific=json.dumps(r['beta_specific'])) for r in fine_entries])
    best=choose_candidate(points,fine)
    selected=dict(T=t,selected_w2=best['w2'],mean_validation_size=best['mean_size'],sd_validation_size=best['sd_size'],
        per_seed=best['per_seed'],beta_specific=best['beta_specific'],bracket=interval,fine_grid=fine,
        old_training_w2=OLD_W2[t],w2_difference=best['w2']-OLD_W2[t],
        reused_checkpoint_count=sum(r['reused_checkpoint'] for rs in records.values() for r in rs),
        newly_trained_checkpoint_count=sum(not r['reused_checkpoint'] for rs in records.values() for r in rs),
        selected_checkpoints=records[best['w2']],selection_rule='absolute distance to 0.05 on 12-beta validation mean; lower w2 on exact tie')
    require(fingerprint(raw)==train_fp and fingerprint(val)==val_fp,'Shared dataset mutated')
    save(OUT/f'calibration_T{t}.json',dict(points=list(points.values()),candidate=selected))
    return selected


def fresh_test(candidates,state):
    require(len(candidates)==3 and state['status']=='selection_complete','Test before frozen selection')
    frozen=sha(OUT/'validation_candidates.json');all_rows=[];fps={}
    for selected in candidates:
        t=selected['T'];d=derived(NEW_TEST_SEED,t)
        require(len({derived(seed,t) for seed in [TRAIN_SEED,VAL_SEED,OLD_TEST_SEED,NEW_TEST_SEED]})==4,'Random streams collide')
        raw=make_ann_dataset(t,RHOS,BETAS,1000,1000,d);fp=fingerprint(raw)
        require(len(raw['y'])==16000 and fp not in [state['datasets'][str(t)]['training_fingerprint'],state['datasets'][str(t)]['validation_fingerprint']],'Test data mismatch')
        fps[str(t)]=dict(seed=NEW_TEST_SEED,derived_seed=d,fingerprint=fp,n=16000)
        save(OUT/'new_test_dataset_fingerprints.json',fps)
        for item in selected['selected_checkpoints']:
            cfg=record_config(t,selected['selected_w2'],item['model_seed'],'independent_test',True)
            cfg.update(test_seed=NEW_TEST_SEED,training_performed=False,test_only=True,test_fingerprint=fp)
            with wandb.init(project='URT_wNN',group=GROUP,job_type='independent-test',name=f'heldout_test_T{t}_seed{item["model_seed"]}',config=cfg,mode='online') as wr:
                ev=evaluate(raw,item)
                require(len(ev['by_parameter'])==16 and all(v['n']==1000 for v in ev['by_parameter']),'Final test cells mismatch')
                wr.log({'test_size':ev['overall']['empirical_size'],'test_power':ev['overall']['empirical_power'],
                    'test_cells':wandb.Table(columns=['rho','beta','rate'],data=[[v['rho'],v['beta'],v['rejection_rate']] for v in ev['by_parameter']])})
                save(OUT/f'test_T{t}_seed{item["model_seed"]}.json',dict(config=cfg,checkpoint=item,result=ev,wandb_url=wr.url))
                state['test_runs'].append(dict(T=t,seed=item['model_seed'],id=wr.id,url=wr.url,status='finished'))
            all_rows.extend([dict(T=t,rho=v['rho'],beta=v['beta'],seed=item['model_seed'],metric_type='size' if v['rho']==1 else 'power',n=v['n'],rejection_rate=v['rejection_rate']) for v in ev['by_parameter']])
            save(OUT/'run_manifest.json',state)
        require(fingerprint(raw)==fp,'Test dataset mutated')
    require(sha(OUT/'validation_candidates.json')==frozen,'Selection changed during test')
    return all_rows


def finish_results(candidates,rows,state):
    means,paper,new_summary,overall=compare(rows)
    write_csv(OUT/'new_test_rejection_rates_by_seed.csv',rows);write_csv(OUT/'new_test_rejection_rates_mean.csv',means)
    with (ROOT/'independent_test/paper_comparison.csv').open() as f:
        old_cells={(int(r['T']),float(r['rho']),float(r['beta'])):r for r in csv.DictReader(f)}
    for row in paper:
        prior=old_cells[(row['T'],row['rho'],row['beta'])]
        row['old_training_based_mean_percent']=float(prior['our_mean_percent'])
        row['old_training_based_SD_pp']=float(prior['our_SD_percentage_points'])
        row['new_minus_old_pp']=row['our_mean_percent']-row['old_training_based_mean_percent']
    write_csv(OUT/'paper_comparison.csv',paper)
    old=read(ROOT/'independent_test/summary_by_T.json')
    training=read(ROOT/'independent_test/protocol_comparison.json')['selected_training'];primary=[];power=[]
    for c in candidates:
        t=str(c['T']);a=old[t];b=new_summary[t];train_size=next(r['mean_size'] for r in training if r['T']==c['T'])
        old_size=a['size']['our_mean_percent']/100;new_size=b['size']['our_mean_percent']/100
        matched=st.mean(r['mean_size'] for r in c['beta_specific'] if r['beta'] in BETAS)
        primary.append(dict(T=c['T'],old_training_w2=c['old_training_w2'],old_training_size=train_size,
            old_independent_size=old_size,new_validation_w2=c['selected_w2'],new_validation_size=c['mean_validation_size'],
            new_validation_SD=c['sd_validation_size'],validation_size_final_4_betas=matched,new_independent_size=new_size,
            paper_size_mean=b['size']['paper_mean_percent']/100,old_size_minus_005=old_size-.05,new_size_minus_005=new_size-.05,
            absolute_gap_improved=abs(new_size-.05)<abs(old_size-.05)))
        for rho in RHOS[1:]:
            power.append(dict(T=c['T'],rho=rho,old_power=a['power_by_rho'][str(rho)]['our_mean_percent']/100,
                new_power=b['power_by_rho'][str(rho)]['our_mean_percent']/100,paper_power=b['power_by_rho'][str(rho)]['paper_mean_percent']/100))
    write_csv(OUT/'training_vs_validation_comparison.csv',primary);write_csv(OUT/'power_comparison.csv',power)
    summary=dict(primary_comparison=primary,power_comparison=power,new_test_summary=new_summary,paper_comparison_summary=overall,
        improved_T_count=sum(r['absolute_gap_improved'] for r in primary),total_T=3,
        newly_trained_checkpoints=sum(not r['reused_checkpoint'] for r in state['records']),
        historical_checkpoints_reused=sum(r['reused_checkpoint'] for r in state['records']),
        assessment='Descriptive results available; interpret Cases 1/2/3 from validation, new test and seed/beta variability. No test-based adjustments.',
        caveats=['Calibration averages 12 beta values, final size averages 4. Matched 4-beta validation size is reported.',
                 'Old and new independent datasets use different random streams; this is an unpaired descriptive comparison.',
                 'Model-seed SD on shared samples is not uncertainty across independent datasets; one data seed is used.'])
    save(OUT/'final_summary.json',summary)
    lines=['# Fixed-200 ANN held-out calibration supplementary experiment','',
        '기존 primary reproduction 결과를 보존하고, 독립 null validation 평균만으로 w2를 선택한 후 새 test stream으로 평가했습니다. Epoch/optimizer/model/loss는 고정했습니다.',
        '', 'Training seed=20191007, validation seed=20261013, old test seed=20261008, new test seed=20261014. Validation은 T별 12 beta × 5000=60000 null series. Test는 T별 16 cells × 1000=16000 series, 총 48000개이며 model seeds 7/17/27이 공유합니다.',
        '', '## Calibration','', '| T | initial endpoints | final bracket | fine grid | selected w2 | validation mean ± SD | old w2 difference |','|---|---|---|---|---|---|---|']
    for c in candidates:lines.append(f"| {c['T']} | {INITIAL[c['T']]} | {c['bracket']} | {c['fine_grid']} | {c['selected_w2']} | {c['mean_validation_size']:.6f} ± {c['sd_validation_size']:.6f} | {c['w2_difference']:+.6f} |")
    lines+=['','## Endpoint 및 adaptive 경로','']
    for h in state['bracket_history']:lines.append('- '+json.dumps(h,ensure_ascii=False))
    lines+=['','## Primary size comparison','', '| T | old train w2 | old train size | old test size | new w2 | validation size | new test size | paper size | old gap | new gap |','|---|---|---|---|---|---|---|---|---|---|']
    for r in primary:lines.append('| '+str(r['T'])+' | '+' | '.join(f'{r[k]:.6f}' for k in ['old_training_w2','old_training_size','old_independent_size','new_validation_w2','new_validation_size','new_independent_size','paper_size_mean','old_size_minus_005','new_size_minus_005'])+' |')
    lines+=['','## Power comparison (rates)','', '| T | rho | old | new | paper |','|---|---|---|---|---|']
    for r in power:lines.append(f"| {r['T']} | {r['rho']} | {r['old_power']:.6f} | {r['new_power']:.6f} | {r['paper_power']:.6f} |")
    lines+=['','## 해석','',f"새 test에서 0.05까지의 절대 거리가 줄어든 T: {summary['improved_T_count']}/3. Size 감소에 동반된 power 감소는 단독으로 실패를 의미하지 않습니다.",
        '', 'Case 1은 validation과 test 모두 약 5%로 유지되는 경우, Case 2는 validation은 근접하지만 test에서 상승하는 경우, Case 3은 validation 단계부터 seed/beta 변동이 큰 경우입니다. 고정된 formal cutoff 없이 mean/SD와 beta별 결과를 함께 검토해야 합니다.',
        '', *summary['caveats'], '',f"신규 학습 {summary['newly_trained_checkpoints']}개, 역사적 checkpoint 재사용 {summary['historical_checkpoints_reused']}개. Fine endpoints와 최종 선택 checkpoint는 재학습하지 않았습니다."]
    (OUT/'final_report.md').write_text('\n'.join(lines)+'\n')


def upload():
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']}
    receipt=dict(status='uploading',artifact_name=ARTIFACT,files=files)
    path=ROOT/'fixed200_validation_calibration_artifact_receipt.json'
    try:
        with wandb.init(project='URT_wNN',group=GROUP,job_type='artifact-upload',name='heldout_final_archive',mode='online') as wr:
            artifact=wandb.Artifact(ARTIFACT,type='reproduction-supplement')
            for name in files:artifact.add_file(str(OUT/name),name=name)
            obj=wr.log_artifact(artifact);obj.wait()
            remote=wandb.Api().artifact(f'{wr.entity}/{wr.project}/{obj.name}',type='reproduction-supplement')
            require(set(remote.manifest.entries)==set(files),'Artifact file list mismatch')
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(remote.download(root=tmp));require(all(sha(root/n)==digest for n,digest in files.items()),'Artifact SHA256 mismatch')
            receipt.update(status='verified',version=remote.version,url=remote.url,verified_at=now())
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));save(path,receipt);raise
    save(path,receipt)


def main():
    require(len({TRAIN_SEED,VAL_SEED,OLD_TEST_SEED,NEW_TEST_SEED})==4,'Seeds must be distinct')
    index=old_index();require(torch.backends.mps.is_available(),'MPS required')
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'training_runs').mkdir();(OUT/'datasets').mkdir()
    c=read('configs/ann_paper2019_reproduction.json');c.update(family=GROUP)
    source=read(ROOT/'w2_calibration/run_manifest.json')
    historical=[ROOT/'w2_calibration/run_manifest.json',ROOT/'w2_calibration_t250_extension/run_manifest.json',ROOT/'independent_test/test_manifest.json',ROOT/'independent_test/summary_by_T.json']
    state=dict(status='initializing',started_at=now(),historical_file_hashes={str(p):sha(p) for p in historical},git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        datasets={},records=[],points={str(t):[] for t in INITIAL},bracket_history=[],test_runs=[])
    save(OUT/'protocol.json',dict(family=GROUP,T=list(INITIAL),initial_endpoints=INITIAL,model_seeds=SEEDS,epochs=200,
        training_seed=TRAIN_SEED,validation_seed=VAL_SEED,old_test_seed=OLD_TEST_SEED,new_test_seed=NEW_TEST_SEED,
        target=TARGET,safety_range=SAFETY,validation_betas=VAL_BETAS,validation_n_per_beta=5000,new_test_rhos=RHOS,
        new_test_betas=BETAS,new_test_n_per_cell=1000,validation_in_optimizer=False,early_stopping=False,
        small_endpoint_delta=.0005,diagnostic_rule='One midpoint for small/reversed slope; continue only if midpoint resolves direction; otherwise needs_attention',
        safety_boundary_rule='Clip one proposed endpoint to boundary, never exceed range',
        candidate_rule='Nearest 12-beta validation mean to 0.05; exact tie lower w2',training_config=c))
    save(OUT/'run_manifest.json',state);start=perf_counter()
    try:
        selected=[]
        for t in INITIAL:
            raw,d=dataset(c,t);train_fp=fingerprint(raw)
            require(train_fp==source['datasets'][str(t)]['fingerprint'],'Training arrays differ from historical data')
            require(len({derived(seed,t) for seed in [TRAIN_SEED,VAL_SEED,OLD_TEST_SEED,NEW_TEST_SEED]})==4,'Derived random streams collide')
            val=make_ann_dataset(t,[1.],VAL_BETAS,5000,5000,derived(VAL_SEED,t));val_fp=fingerprint(val)
            require(len(val['y'])==60000 and np.all(val['y']==0) and val_fp!=train_fp,'Validation dataset mismatch')
            np.savez(OUT/f'datasets/training_T{t}.npz',**raw);np.savez(OUT/f'datasets/validation_T{t}.npz',**val)
            state['datasets'][str(t)]=dict(training_fingerprint=train_fp,validation_fingerprint=val_fp,
                training_derived_seed=d,validation_derived_seed=derived(VAL_SEED,t),validation_n=60000)
            save(OUT/'validation_dataset_fingerprints.json',state['datasets'])
            state.update(status='calibration_running',current_T=t)
            candidate=calibrate_t(t,c,raw,val,index,state)
            if candidate is None:
                state['total_runtime_seconds']=perf_counter()-start;save(OUT/'run_manifest.json',state);return
            selected.append(candidate);save(OUT/'validation_candidates.json',selected)
        state.update(status='selection_complete',selected_at=now(),selection_sha256=sha(OUT/'validation_candidates.json'))
        save(OUT/'run_manifest.json',state)
        rows=fresh_test(selected,state);finish_results(selected,rows,state)
        require(all(sha(p)==digest for p,digest in state['historical_file_hashes'].items()),'Historical result changed')
        state.update(status='complete',finished_at=now(),total_runtime_seconds=perf_counter()-start,
                     newly_trained_checkpoints=sum(not r['reused_checkpoint'] for r in state['records']),
                     historical_checkpoints_reused=sum(r['reused_checkpoint'] for r in state['records']))
        save(OUT/'run_manifest.json',state)
        final=read(OUT/'final_summary.json');final.update(total_runtime_seconds=state['total_runtime_seconds'],git_sha=state['git_sha']);save(OUT/'final_summary.json',final)
        with (OUT/'final_report.md').open('a') as report:
            report.write(f"\n실행 시간(artifact 업로드 전): {state['total_runtime_seconds']/3600:.3f}시간. 실행 main SHA: `{state['git_sha']}`.\n")
    except Exception as exc:
        state.update(status='failed',error=repr(exc),total_runtime_seconds=perf_counter()-start);save(OUT/'run_manifest.json',state);raise
    # Upload failures never relabel or delete a completed scientific experiment.
    upload()


if __name__=='__main__':main()
