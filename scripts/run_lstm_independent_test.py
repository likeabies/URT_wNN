"""Evaluate frozen LSTM checkpoints against the archived official ANN test."""
import argparse
import math
import shutil
import statistics as st
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset

from scripts.run_ann_independent_test import (
    make_test, read, require, sha, now, write_csv, SEEDS, RHOS, BETAS, TEST_SEED,
)
from scripts.run_ann_paper2019 import fingerprint, save
from src.lstm_binary import evaluate_lstm as evaluate_ann, from_legacy
from src.model import Paper2019LSTMClassifier
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy

SELECTED = {50: .55625, 100: .425, 250: .74375}
FINGERPRINTS = {
    50: 'd1513a67d48f6f689ceabe1f84c383b7e7518a235529f07c9cc5d119f36230e0',
    100: '04aaa1699cc91799f9c3ae32421f58a5d679584d24d1d5f3e62d4071046a64aa',
    250: 'e95d95a0efc66894dcde26bd9d9cda1ca2cff288389605f189cf8ac78b1cd7ce',
}
ROOT = Path('results/lstm_paper2019_extension')
RESUME = ROOT/'w2_calibration_range_extension/resume_20261014_215318_786719'
ANN_ARTIFACT = 'rdces-chung-ang-university/URT_wNN/ann_paper2019_reproduction_final_results:v0'
GROUP = 'lstm_fixed200_official_independent_test_vs_ann_20261018'


def preflight(ann):
    am = read(ann/'test_manifest.json')
    af = read(ann/'test_dataset_fingerprints.json')
    require(am['status']=='complete' and am['test_seed']==TEST_SEED and am['model_seeds']==SEEDS,
            'ANN official protocol mismatch')
    for name, digest in am['source_hashes'].items():
        # Model/loss/metric sources intentionally changed for the LSTM label migration.
        # Keep strict DGP, normalization, RNG, config and generator hash checks;
        # current evaluator sources are recorded in the new test manifest.
        if name not in {'src/model.py','src/lstm_binary.py','src/train.py','src/evaluate.py'}:
            require(sha(name)==digest, 'Scientific source differs from official ANN: '+name)
    require(np.__version__==am['environment']['numpy'] and torch.__version__==am['environment']['torch'],
            'Numerical library versions differ from official ANN')
    manifests = [read(ROOT/'w2_calibration/run_manifest.json'), read(RESUME/'run_manifest.json')]
    require(all(m['status']=='complete' for m in manifests), 'LSTM calibration incomplete')
    selected = read(RESUME/'calibration_candidates.json')
    items, data, ann_results = [], {}, []
    for t,w in SELECTED.items():
        require(next(r for r in selected if r['T']==t)['w2']==w, 'Training-selected weight mismatch')
        raw,derived = make_test(t)
        fp = fingerprint(raw)
        require(fp==FINGERPRINTS[t]==af[str(t)]['fingerprint'], 'Test fingerprint mismatch: '+str(t))
        require(derived==af[str(t)]['derived_seed'], 'Derived test seed mismatch')
        data[t] = from_legacy(raw)
        records = [r for m in manifests for r in m['runs'] if (r['T'],r['w2'])==(t,w)]
        require(sorted(r['model_seed'] for r in records)==SEEDS, 'Missing or duplicate LSTM checkpoint')
        for r in sorted(records,key=lambda r:r['model_seed']):
            p=Path(r['path']);cfg=read(p/'config_snapshot.json');ev=read(p/'train_epoch_200.json')
            expected=dict(model='LSTM',T=t,w1=1.,w2=w,model_seed=r['model_seed'],epochs=200,
                          hidden_size=30,num_layers=1,input_size=1,output_size=2,data_seed=20191007,
                          optimizer='Adam',lr=.001,batch_size=256,weight_decay=0.,dropout=0.,
                          shuffle=True,validation=False,early_stopping=False)
            require(all(cfg.get(k)==v for k,v in expected.items()),'LSTM config mismatch')
            ap=next(a for a in am['checkpoints'] if a['T']==t and a['model_seed']==r['model_seed'])
            require(cfg['data_fingerprint']==r['data_fingerprint']==ap['training_fingerprint']==af[str(t)]['training_fingerprint'],
                    'ANN/LSTM training sample mismatch')
            require(r['status']=='finished' and ev['overall']==r['metrics'], 'Invalid calibration record')
            require(read(p/'summary.json')['status']=='finished','Incomplete saved run')
            require([h['epoch'] for h in read(p/'history.json')]==list(range(1,201)), 'Incomplete epoch history')
            cp=torch.load(p/'epoch_200.pt',map_location='cpu',weights_only=True)
            require(cp['epoch']==200 and all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Invalid checkpoint')
            Paper2019LSTMClassifier().load_state_dict(cp['model'],strict=True)
            items.append(dict(T=t,w2=w,model_seed=r['model_seed'],path=str(p/'epoch_200.pt'),
                              sha256=sha(p/'epoch_200.pt'),training_fingerprint=r['data_fingerprint']))
            ae=read(ann/f'evaluation_T{t}_seed{r["model_seed"]}.json')
            require(ae['config']['test_fingerprint']==fp and ae['config']['selected_w2']==float(am['selected_w2'][str(t)]),
                    'ANN evaluation provenance mismatch')
            ann_results.append(dict(T=t,Model='ANN',w2=ae['config']['selected_w2'],seed=r['model_seed'],result=ae['result']))
    return items,data,ann_results


def seed_metrics(r):
    cells=r['result']['by_parameter']
    require(len(cells)==16 and {(c['rho'],c['beta']) for c in cells}=={(a,b) for a in RHOS for b in BETAS},'Cell grid mismatch')
    require(all(c['n']==1000 and 0<=c['rejection_rate']<=1 for c in cells),'Invalid cell')
    size=st.mean(c['rejection_rate'] for c in cells if c['rho']==1)
    power=st.mean(c['rejection_rate'] for c in cells if c['rho']<1)
    require(math.isclose(size,r['result']['overall']['empirical_size'],abs_tol=1e-12),'Overall size mismatch')
    require(math.isclose(power,r['result']['overall']['empirical_power'],abs_tol=1e-12),'Overall power mismatch')
    return dict(T=r['T'],Model=r['Model'],w2=r['w2'],seed=r['seed'],size=size,power=power)


def tables(results):
    seedrows=[seed_metrics(r) for r in results]
    require(len(seedrows)==18 and len({(r['T'],r['Model'],r['seed']) for r in seedrows})==18,'Seed coverage mismatch')
    one,two,three,four,paired,cells=[],[],[],[],[],[]
    for t in SELECTED:
        for model in ['ANN','LSTM']:
            rs=[r for r in results if r['T']==t and r['Model']==model]
            require(sorted(r['seed'] for r in rs)==SEEDS,'Missing seeds')
            sr=[r for r in seedrows if r['T']==t and r['Model']==model]
            one.append(dict(T=t,Model=model,w2=sr[0]['w2'],Mean_Size=st.mean(r['size'] for r in sr),
                            Size_SD=st.stdev(r['size'] for r in sr),Mean_Power=st.mean(r['power'] for r in sr),
                            Power_SD=st.stdev(r['power'] for r in sr)))
            for rho in RHOS:
                for beta in BETAS:
                    vals=[next(c['rejection_rate'] for c in r['result']['by_parameter'] if (c['rho'],c['beta'])==(rho,beta)) for r in rs]
                    cells.append(dict(T=t,Model=model,rho=rho,beta=beta,mean=st.mean(vals),SD=st.stdev(vals)))
            for rho in RHOS[1:]:
                vals=[st.mean(c['rejection_rate'] for c in r['result']['by_parameter'] if c['rho']==rho) for r in rs]
                two.append(dict(T=t,Model=model,rho=rho,Mean_Power=st.mean(vals),Power_SD=st.stdev(vals)))
            for beta in BETAS:
                row=next(c for c in cells if (c['T'],c['Model'],c['rho'],c['beta'])==(t,model,1,beta))
                three.append(dict(T=t,Model=model,beta=beta,Mean_Size=row['mean'],Size_SD=row['SD']))
        for rho in RHOS[1:]:
            for beta in BETAS:
                a,l=[next(c['mean'] for c in cells if (c['T'],c['Model'],c['rho'],c['beta'])==(t,mod,rho,beta)) for mod in ['ANN','LSTM']]
                four.append(dict(T=t,rho=rho,beta=beta,ANN_Power=a,LSTM_Power=l,LSTM_minus_ANN=l-a))
        for seed in SEEDS:
            a,l=[next(r for r in seedrows if (r['T'],r['Model'],r['seed'])==(t,mod,seed)) for mod in ['ANN','LSTM']]
            paired.append(dict(T=t,seed=seed,ANN_size=a['size'],LSTM_size=l['size'],size_difference=l['size']-a['size'],
                               ANN_power=a['power'],LSTM_power=l['power'],power_difference=l['power']-a['power']))
    return dict(table1_overall=one,table2_power_by_rho=two,table3_size_by_beta=three,
                table4_cell_power=four,paired_by_seed=paired,metrics_by_seed=seedrows,cell_mean_sd=cells)


def report(ts):
    text=['# ANN vs LSTM official independent test','',
          '동일 test sample, training data fingerprint, model seeds 7/17/27 및 fixed-200 protocol 비교. '
          '모델별 training-based w2는 평가 전에 고정했으며 추가 학습·calibration·threshold tuning은 없다.',
          '', '먼저 size를 확인해야 한다. nominal 5%와의 차이 또는 모델 간 size 차이가 크면 raw power를 '
          'size-adjusted 우열로 해석할 수 없다. 이 비교에서는 test 기반 size 보정도 수행하지 않았다.',
          '', '같은 seed 번호끼리의 차이는 기술적 비교다. 다른 구조의 초기화가 동일하다는 의미가 아니며, '
          '동일 test sample을 공유하는 3개 model seed의 sample SD(ddof=1)는 독립 Monte Carlo 반복의 불확실성이 아니다.',
          '', '모든 rate와 SD는 0–1 단위. ANN은 official artifact v0의 저장 결과만 사용했다.']
    for name,rows in ts.items():
        if name=='cell_mean_sd':continue
        fields=list(rows[0]);text+=['',f'## {name}','','| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
        for r in rows:
            text.append('| '+' | '.join(f'{r[k]:.8f}' if isinstance(r[k],float) else str(r[k]) for k in fields)+' |')
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ann-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'official_independent_test_vs_ann_20261018')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    items,data,results=preflight(args.ann_results)
    if args.preflight:
        print('Preflight passed: 9 epoch-200 checkpoints; all test and training fingerprints match official ANN.')
        return
    require(torch.backends.mps.is_available(),'MPS required')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(args.ann_results,out/'ann_official_reference_v0')
    shutil.copyfile(__file__,out/'evaluation_script.py')
    manifest=dict(status='running',started_at=now(),device='mps',test_seed=TEST_SEED,test_fingerprints=FINGERPRINTS,
                  selected_w2=SELECTED,model_seeds=SEEDS,checkpoints=items,ann_artifact=ANN_ARTIFACT,
                  training_performed=False,test_based_selection=False,wandb_group=GROUP,evaluation_runs=[],
                  source_hashes={n:sha(n) for n in ['src/data.py','src/model.py','src/lstm_binary.py','src/train.py','src/evaluate.py',
                    'scripts/run_ann_independent_test.py',__file__]},
                  environment=dict(numpy=np.__version__,torch=torch.__version__,wandb=wandb.__version__))
    save(out/'test_manifest.json',manifest)
    try:
        for item in items:
            t=item['T'];seed=item['model_seed'];raw=data[t]
            require(sha(item['path'])==item['sha256'],'Checkpoint changed')
            model=Paper2019LSTMClassifier().to('mps')
            model.load_state_dict(torch.load(item['path'],map_location='cpu',weights_only=True)['model'])
            cfg=dict(model='LSTM',T=t,model_seed=seed,w1=1.,w2=item['w2'],epochs=200,device='mps',
                     test_seed=TEST_SEED,test_fingerprint=FINGERPRINTS[t],checkpoint_sha256=item['sha256'],
                     training_fingerprint=item['training_fingerprint'],test_only=True,training_performed=False)
            with wandb.init(project='URT_wNN',group=GROUP,job_type='evaluation',name=f'LSTM_official_T{t}_seed{seed}',config=cfg,mode='online') as wr:
                ev=evaluate_ann(model,TensorDataset(torch.from_numpy(raw['x']),torch.from_numpy(raw['y'])),raw,
                                partial(ann_weighted_cross_entropy,w1=1.,w2=item['w2']),batch_size=256,device='mps')
                r=dict(T=t,Model='LSTM',w2=item['w2'],seed=seed,result=ev)
                metrics=seed_metrics(r)
                metrics['power_by_rho']={str(rho):st.mean(c['rejection_rate'] for c in ev['by_parameter'] if c['rho']==rho) for rho in RHOS[1:]}
                metrics['size_by_beta']={str(c['beta']):c['rejection_rate'] for c in ev['by_parameter'] if c['rho']==1}
                metrics['cells']={f"rho={c['rho']:g},beta={c['beta']:g}":c['rejection_rate'] for c in ev['by_parameter']}
                wr.log(metrics)
                save(out/f'evaluation_T{t}_seed{seed}.json',dict(config=cfg,result=ev,metrics=metrics,wandb_id=wr.id,wandb_url=wr.url))
                run=dict(T=t,seed=seed,id=wr.id,url=wr.url,status='finished')
            manifest['evaluation_runs'].append(run);save(out/'test_manifest.json',manifest)
            results.append(r)
            require(fingerprint(from_legacy(raw))==FINGERPRINTS[t],'Test sample mutated')
            print(f'Finished T={t}, seed={seed}: size={metrics["size"]:.8f}, power={metrics["power"]:.8f}',flush=True)
        ts=tables(results)
        for name,rows in ts.items():write_csv(out/(name+'.csv'),rows)
        save(out/'all_evaluation_results.json',results)
        save(out/'comparison_tables.json',ts)
        (out/'comparison_report.md').write_text(report(ts))
        require(all(sha(i['path'])==i['sha256'] for i in items),'Checkpoint modified')
        manifest.update(status='complete',finished_at=now(),evaluation_count=9,cell_seed_records=144)
        save(out/'test_manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));save(out/'test_manifest.json',manifest);raise
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    with wandb.init(project='URT_wNN',group=GROUP,job_type='comparison',name='ANN_vs_LSTM_official_comparison',
                    config={'training_performed':False,'test_based_selection':False,'ann_artifact':ANN_ARTIFACT},mode='online') as wr:
        wr.log({name:wandb.Table(columns=list(rows[0]),data=[list(r.values()) for r in rows]) for name,rows in ts.items()})
        artifact=wandb.Artifact(GROUP+'_results',type='controlled-comparison',metadata={'test_seed':TEST_SEED,'evaluation_only':True})
        artifact.add_dir(str(out));uploaded=wr.log_artifact(artifact);uploaded.wait()
        qualified=f'{wr.entity}/{wr.project}/{uploaded.name}'
        remote=wandb.Api().artifact(qualified,type='controlled-comparison')
        require(set(remote.manifest.entries)==set(files),'Artifact file set mismatch')
        with tempfile.TemporaryDirectory() as tmp:
            downloaded=Path(remote.download(root=tmp))
            require(all(sha(downloaded/n)==digest for n,digest in files.items()),'Artifact hash mismatch')
        receipt=dict(status='verified',qualified_name=qualified,url=remote.url,run_url=wr.url,files_sha256=files)
    save(out.with_name(out.name+'_artifact_receipt.json'),receipt)
    print(receipt['url'],flush=True)


if __name__=='__main__':
    main()
