"""Evaluation-only paper comparison. Fixed checkpoints; never trains or selects models."""
import argparse
import ast
import csv
import hashlib
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
from src.data import make_ann_dataset, BINARY_LABELS
from src.evaluate import evaluate_ann
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy
from scripts.run_ann_paper2019 import fingerprint, save

ROOT = Path('results/ann_paper2019_reproduction')
OUT = ROOT/'independent_test'
GROUP = 'ann_paper2019_reproduction_independent_test'
ARTIFACT = 'ann_paper2019_reproduction_final_results'
SELECTED = {50:.375, 100:.4375, 250:.6375}
SEEDS = [7,17,27]
TEST_SEED = 20261008
RHOS = [1.,.95,.9,.8]
BETAS = [.9,.6,.3,0.]
N = 1000
REFERENCE = Path('configs/ann_paper2019_test_reference.json')
SCIENTIFIC = ['src/data.py','src/model.py','src/train.py','src/evaluate.py',
              'src/utils.py','scripts/run_ann_paper2019.py']


def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise RuntimeError(message)


def write_csv(p, rows):
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def paper_values():
    p=read(REFERENCE)
    require(p['rhos']==RHOS and p['betas']==BETAS,'Reference grid mismatch')
    return {(int(t),rho,beta):p['ANN_percent'][t][i][j]
            for t in p['ANN_percent'] for i,rho in enumerate(RHOS) for j,beta in enumerate(BETAS)}


def check_scientific_source(commit):
    for name in SCIENTIFIC:
        old=subprocess.check_output(['git','show',commit+':'+name]); current=Path(name).read_bytes()
        if name=='src/model.py':
            def ann(source):
                return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ANNClassifier'))
            same=ann(old)==ann(current)
        else: same=old==current
        require(same,'Changed scientific source: '+name)


def preflight():
    config=read('configs/ann_paper2019_reproduction.json')
    require(config['test_seed']==TEST_SEED and config['data_seed']!=TEST_SEED,'Seed separation failed')
    require(BINARY_LABELS=={'unit_root':0,'stationary':1},'Label mismatch')
    coarse=read(ROOT/'w2_calibration/run_manifest.json')
    ext=read(ROOT/'w2_calibration_t250_extension/run_manifest.json')
    require(coarse['status']==ext['status']=='complete','Calibration not complete')
    check_scientific_source(coarse['git_sha']); check_scientific_source(ext['git_sha'])
    initial=read(ROOT/'w2_calibration/calibration_candidates.json')
    extension=read(ROOT/'w2_calibration_t250_extension/calibration_candidate_t250.json')
    selected_training=[]; checkpoints=[]
    for t,w in SELECTED.items():
        candidate=extension if t==250 else next(c for c in initial if c['T']==t)
        require(candidate['w2']==w,'Training-selected weight mismatch')
        source=ext if t==250 else coarse
        records=[r for r in source['runs'] if (r['T'],r['w2'])==(t,w)]
        require(sorted(r['model_seed'] for r in records)==SEEDS,'Checkpoint seed coverage mismatch')
        sizes=[]
        for r in sorted(records,key=lambda x:x['model_seed']):
            p=Path(r['path']); cfg=read(p/'config_snapshot.json')
            expected=dict(T=t,w2=w,w1=1.,model_seed=r['model_seed'],epochs=200,data_seed=20191007,
                          optimizer='Adam',lr=.001,batch_size=256,dropout=0,weight_decay=0,validation=False)
            require(all(cfg[k]==v for k,v in expected.items()),'Checkpoint configuration mismatch')
            fp=coarse['datasets'][str(t)]['fingerprint']
            require(r['data_fingerprint']==cfg['data_fingerprint']==fp,'Training fingerprint mismatch')
            require(r['status']=='finished','Checkpoint run not finished')
            ev=read(p/'train_epoch_200.json')
            require(ev['overall']==r['metrics'],'Calibration metrics mismatch')
            require([h['epoch'] for h in read(p/'history.json')]==list(range(1,201)),'Epoch history mismatch')
            cp=torch.load(p/'epoch_200.pt',map_location='cpu',weights_only=True)
            require(cp['epoch']==200 and all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Invalid checkpoint')
            model=ANNClassifier(t,20 if t==50 else 50)
            model.load_state_dict(cp['model'],strict=True)
            require(tuple(cp['model']['hidden.weight'].shape)==(20 if t==50 else 50,t),'Hidden layer mismatch')
            summary=read(p/'summary.json')
            require(summary['status']=='finished','Saved run summary not finished')
            checkpoints.append(dict(T=t,w2=w,model_seed=r['model_seed'],epoch=200,hidden_size=20 if t==50 else 50,
                path=str(p/'epoch_200.pt'),sha256=sha(p/'epoch_200.pt'),training_fingerprint=fp,
                calibration_manifest=str((ROOT/('w2_calibration_t250_extension' if t==250 else 'w2_calibration'))/'run_manifest.json'),
                calibration_manifest_sha256=sha((ROOT/('w2_calibration_t250_extension' if t==250 else 'w2_calibration'))/'run_manifest.json'),
                training_type_i_error=r['metrics']['empirical_size'],training_power=r['metrics']['empirical_power']))
            sizes.append(r['metrics']['empirical_size'])
        require(math.isclose(st.mean(sizes),candidate['mean_type_i_error'],abs_tol=1e-12),'Candidate mean mismatch')
        selected_training.append(dict(T=t,w2=w,mean_size=st.mean(sizes),sd_size=st.stdev(sizes)))
    paper=read(REFERENCE)
    protocol=dict(source_url=paper['source_url'],source_tables=paper['source_tables'],source_checked_date=paper['verified_date'],
        published_values_verified=48,corrections_to_user_published_values=paper['corrections_to_user_published_values'],
        paper_internal_inconsistencies=paper['paper_internal_inconsistencies'],
        matched=['DGP with alpha=0 and Y0=0, iid Gaussian innovations','per-series mean centering then maximum absolute centered value scaling',
                 'one ReLU hidden layer, 20/50/50 hidden nodes, two output classes','literal WCE class weights and sample-mean denominator',
                 'training rho grid and 120000 series per T (beta=0 supported by Table 3 and totals)',
                 'final 16 test cells per T, 1000 fresh series per cell'],
        label_convention={'class_0':'unit root','class_1':'stationary','reject_H0':'predicted class 1','decision':'existing softmax(logits).argmax, mathematically same ordering as logits.argmax'},
        implementation_choices={'epochs':200,'optimizer':'Adam','lr':.001,'batch_size':256,'dropout':0,'weight_decay':0,
            'validation':False,'early_stopping':False,'model_seeds':SEEDS,'test_seed':TEST_SEED,
            'innovation':'N(0,1), with iid e0','note':'These precise optimization, initialization, RNG and stopping settings are not specified in the paper.'},
        methodological_differences=['Observed training-grid calibration rather than the paper polynomial fit; weights fixed before testing.',
            'Reuse calibrated 200-epoch checkpoints per user protocol; paper describes training final ANNs after estimating weights.',
            'Report mean and sample SD of 3 fixed model seeds sharing a test sample; original table does not report this seed ensemble.'],
        known_w2_discrepancy={'paper_reference':{'50':2.44,'100':1.99,'250':1.94},'our_fixed':SELECTED,
            'note':'Printed formula/class definitions retained; published w2 and Figure 2 direction are inconsistent with our observed size response. No reciprocal substitution.'},
        selected_training=selected_training,training_performed=False,test_based_selection=False,
        excluded=['ADF including GtS','LSTM','GRU','polynomial calibration','threshold tuning'])
    return protocol,checkpoints


def make_test(t):
    derived=int(np.random.SeedSequence([TEST_SEED,t]).generate_state(1)[0])
    training=int(np.random.SeedSequence([20191007,t]).generate_state(1)[0])
    require(derived!=training,'Derived RNG seed collision')
    raw=make_ann_dataset(t,RHOS,BETAS,N,N,derived)
    require(len(raw['y'])==16000 and np.isfinite(raw['x']).all(),'Invalid test data')
    require(np.max(np.abs(raw['x'].mean(axis=1)))<1e-6,'Centering failed')
    require(np.allclose(np.max(np.abs(raw['x']),axis=1),1),'Scaling failed')
    for rho in RHOS:
        for beta in BETAS:
            mask=(raw['rho']==rho)&(raw['beta']==beta)
            require(int(mask.sum())==N,'Test cell count mismatch')
            require(np.all(raw['y'][mask]==(0 if rho==1 else 1)),'Test labels incorrect')
    return raw,derived


def evaluate_checkpoints(raw, items, evaluator):
    # The same raw object is passed for all three seeds; no test-driven choices.
    require(sorted(r['model_seed'] for r in items)==SEEDS,'Must evaluate all seeds')
    return [evaluator(raw,item) for item in sorted(items,key=lambda x:x['model_seed'])]


def compare(by_seed):
    require(len(by_seed)==144,'Expected 144 cell/seed records')
    paper=paper_values(); means=[]; comparison=[]
    require(len({(r['T'],r['rho'],r['beta'],r['seed']) for r in by_seed})==144,'Duplicate cell/seed')
    for (t,rho,beta),published in paper.items():
        rows=sorted([r for r in by_seed if (r['T'],r['rho'],r['beta'])==(t,rho,beta)],key=lambda x:x['seed'])
        require([r['seed'] for r in rows]==SEEDS,'Missing cell seeds')
        vals=[r['rejection_rate'] for r in rows]; mean=st.mean(vals); sd=st.stdev(vals)
        means.append(dict(T=t,rho=rho,beta=beta,metric_type='size' if rho==1 else 'power',mean=mean,SD=sd,n_per_seed=N,model_seed_count=3))
        comparison.append(dict(T=t,rho=rho,beta=beta,metric_type='size' if rho==1 else 'power',
            seed_7_percent=100*vals[0],seed_17_percent=100*vals[1],seed_27_percent=100*vals[2],
            our_mean_percent=100*mean,our_SD_percentage_points=100*sd,paper_ANN_percent=published,
            difference_percentage_points=100*mean-published,absolute_difference_percentage_points=abs(100*mean-published)))
    summaries={}
    for t in SELECTED:
        sizes=[r for r in comparison if r['T']==t and r['metric_type']=='size']
        powers=[r for r in comparison if r['T']==t and r['metric_type']=='power']
        def describe(rows):
            ours=[r['our_mean_percent'] for r in rows]; theirs=[r['paper_ANN_percent'] for r in rows]
            return dict(our_mean_percent=st.mean(ours),our_range_percent=[min(ours),max(ours)],
                        paper_mean_percent=st.mean(theirs),paper_range_percent=[min(theirs),max(theirs)],
                        mean_absolute_difference_pp=st.mean(r['absolute_difference_percentage_points'] for r in rows))
        by_rho={}
        for rho in RHOS[1:]:
            d=describe([r for r in powers if r['rho']==rho]);d['difference_pp']=d['our_mean_percent']-d['paper_mean_percent'];d['absolute_difference_pp']=abs(d['difference_pp']);by_rho[str(rho)]=d
        summaries[str(t)]=dict(size=describe(sizes),size_cells=sizes,power=describe(powers),power_by_rho=by_rho)
    increasing=sum(all(next(r['our_mean_percent'] for r in comparison if (r['T'],r['rho'],r['beta'])==(a,rho,beta))<next(r['our_mean_percent'] for r in comparison if (r['T'],r['rho'],r['beta'])==(b,rho,beta)) for a,b in [(50,100),(100,250)]) for rho in RHOS[1:] for beta in BETAS)
    overall=dict(overall_MAD_pp=st.mean(r['absolute_difference_percentage_points'] for r in comparison),
        size_MAD_pp=st.mean(r['absolute_difference_percentage_points'] for r in comparison if r['metric_type']=='size'),
        power_MAD_pp=st.mean(r['absolute_difference_percentage_points'] for r in comparison if r['metric_type']=='power'),
        power_cells_increasing_with_T=increasing,power_cell_count=12,
        interpretation_limits='SD reflects model-seed variation on a shared test set, not independent Monte Carlo replication. No formal equivalence threshold or paper seed-level uncertainty is available.',
        assessment=None,assessment_basis=None)
    return means,comparison,summaries,overall


def render_report(protocol,comparison,summaries,overall):
    lines=['# ANN paper reproduction 최종 독립 Monte Carlo 평가','',
        '고정된 9개 checkpoint로 새로운 독립 test sample의 size와 power를 논문 ANN 결과와 비교했습니다. 추가 학습, calibration, validation 및 test 기반 선택은 없습니다.',
        '',f"원문: {protocol['source_url']} (Sections 2.2–3, Tables 4a/4b/5/6). 48개 ANN 수치는 요청문과 일치했습니다.",
        '', '## 설계 및 preflight','',
        'T=50/100/250, w2=0.375/0.4375/0.6375. Model seeds=7/17/27. Test seed=20261008. 각 T에서 16 cells × 1000 = 16000개, 총 48000개 고유 시계열을 생성하고 세 모델이 공유했습니다. rho=[1,.95,.9,.8], beta=[.9,.6,.3,0]. rho=.8은 학습 grid에 없었습니다.',
        '', 'DGP·전처리·ANN 구조·label/loss convention은 기존 calibration 코드와 일치하고, 200-epoch checkpoint provenance와 유한값을 검증했습니다. Class 0=unit root, class 1=stationary이며 class 1 예측이 기각입니다.',
        '', 'Adam/lr=.001/batch=256/epoch=200/dropout=0/weight decay=0 및 seed·초기화·중단 설정은 논문에 구체적으로 명시되지 않은 구현 선택입니다.',
        '', '차이: 논문의 polynomial weight 추정 대신 관측 training-grid 후보를 고정했고, weight 추정 후 재학습 대신 기존 checkpoint를 사용했습니다. 논문 w2=2.44/1.99/1.94는 참고값이며 적용하지 않았습니다. 인쇄된 WCE/class 정의와 Figure 2 방향 사이의 불일치를 유지·기록했습니다.',
        '', '표 1의 beta=0 누락은 표 3과 표본 총수로 보완했습니다. 표 2의 T250 size=0.47은 원문 내부 불일치로 기록했으며 수정값을 임의로 사용하지 않았습니다.',
        '', '## T별 요약 (%)','', '| T | 우리 size 평균 (범위) | 논문 size 평균 | Size MAD (pp) | 우리 power 평균 | 논문 power 평균 | Power MAD (pp) |', '|---|---|---|---|---|---|---|']
    for t,d in summaries.items():
        s=d['size'];p=d['power'];lines.append(f"| {t} | {s['our_mean_percent']:.3f} ({s['our_range_percent'][0]:.3f}–{s['our_range_percent'][1]:.3f}) | {s['paper_mean_percent']:.3f} | {s['mean_absolute_difference_pp']:.3f} | {p['our_mean_percent']:.3f} | {p['paper_mean_percent']:.3f} | {p['mean_absolute_difference_pp']:.3f} |")
    lines+=['','## Cell별 논문 비교','', 'Rate는 %, SD와 차이는 percentage points입니다. SD는 model seeds 3개의 sample SD(ddof=1)입니다.','']
    for metric in ['size','power']:
        lines += [f'### {metric}','', '| T | rho | beta | seed7 | seed17 | seed27 | 평균 | SD | 논문 | 차이 | 절대차이 |','|---|---|---|---|---|---|---|---|---|---|---|']
        for r in comparison:
            if r['metric_type']==metric:
                lines.append(f"| {r['T']} | {r['rho']} | {r['beta']} | "+' | '.join(f'{r[k]:.3f}' for k in ['seed_7_percent','seed_17_percent','seed_27_percent','our_mean_percent','our_SD_percentage_points','paper_ANN_percent','difference_percentage_points','absolute_difference_percentage_points'])+' |')
    lines+=['','## rho별 power와 T 패턴','', '| T | rho | 우리 mean power (%) | 논문 (%) | 차이 (pp) |','|---|---|---|---|---|']
    for t,d in summaries.items():
        for rho,p in d['power_by_rho'].items():lines.append(f"| {t} | {rho} | {p['our_mean_percent']:.3f} | {p['paper_mean_percent']:.3f} | {p['difference_pp']:.3f} |")
    lines+=['',f"T 증가에 따라 power가 증가한 동일 rho/beta 조합: {overall['power_cells_increasing_with_T']}/12. rho=.8 행은 학습에 없던 rho로의 일반화 결과입니다.",
        '',f"전체 48 cells MAD={overall['overall_MAD_pp']:.3f} pp; size MAD={overall['size_MAD_pp']:.3f} pp; power MAD={overall['power_MAD_pp']:.3f} pp.",
        '', '## 최종 판단','', str(overall['assessment'] or '수치 검토 후 판단 기록 예정'),'',str(overall['assessment_basis'] or ''),
        '', '세 모델은 같은 test sample을 공유하므로 SD는 model seed 변동만 나타냅니다. 논문의 seed별 결과·학습 상세가 없어 통계적 동등성을 입증하는 비교는 아닙니다. ADF/GtS는 이번 평가에서 제외했습니다.']
    return '\n'.join(lines)+'\n'


def run_evaluation():
    protocol,items=preflight()
    require(torch.backends.mps.is_available(),'MPS unavailable')
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',started_at=now(),test_seed=TEST_SEED,T_values=list(SELECTED),model_seeds=SEEDS,
        selected_w2=SELECTED,n_per_cell=N,cells_per_T=16,total_unique_test_series=48000,checkpoints=items,
        training_performed=False,test_only=True,test_based_selection=False,wandb_group=GROUP,evaluation_runs=[],
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment={'numpy':np.__version__,'torch':torch.__version__,'wandb':wandb.__version__},
        source_hashes={name:sha(name) for name in SCIENTIFIC+[str(REFERENCE),'scripts/run_ann_independent_test.py']})
    save(OUT/'test_manifest.json',manifest);save(OUT/'protocol_comparison.json',protocol)
    save(OUT/'paper_reference.json',read(REFERENCE)); fps={}; all_rows=[]; start=perf_counter()
    try:
        for t in SELECTED:
            raw,derived=make_test(t); fp=fingerprint(raw); selected=[r for r in items if r['T']==t]
            require(all(fp!=r['training_fingerprint'] for r in items),'Test/training fingerprint collision')
            fps[str(t)]=dict(test_seed=TEST_SEED,derived_seed=derived,fingerprint=fp,n=16000,
                training_fingerprint=selected[0]['training_fingerprint'],different_from_all_training=True)
            save(OUT/'test_dataset_fingerprints.json',fps)
            ds=TensorDataset(torch.from_numpy(raw['x']),torch.from_numpy(raw['y']))
            def evaluate_one(shared,item):
                require(shared is raw,'Dataset must be shared')
                require(sha(item['path'])==item['sha256'],'Checkpoint changed since preflight')
                model=ANNClassifier(t,item['hidden_size']).to('mps')
                model.load_state_dict(torch.load(item['path'],map_location='cpu',weights_only=True)['model'],strict=True)
                cfg=dict(model='ANN',stage='independent_test',T=t,selected_w2=SELECTED[t],model_seed=item['model_seed'],
                    test_seed=TEST_SEED,n_per_cell=N,test_only=True,training_performed=False,epochs=200,
                    test_fingerprint=fp,checkpoint_sha256=item['sha256'])
                with wandb.init(project='URT_wNN',group=GROUP,job_type='evaluation',name=f"ANN_test_T{t}_seed{item['model_seed']}",config=cfg,mode='online') as wr:
                    ev=evaluate_ann(model,ds,shared,partial(ann_weighted_cross_entropy,w1=1.,w2=SELECTED[t]),batch_size=256,device='mps')
                    require(len(ev['by_parameter'])==16 and all(c['n']==N for c in ev['by_parameter']),'Evaluation cell mismatch')
                    require(all(math.isfinite(v) for v in ev['overall'].values()),'Nonfinite evaluation metrics')
                    require(all(0 <= c['rejection_rate'] <= 1 for c in ev['by_parameter']),'Invalid rejection rates')
                    save(OUT/f'evaluation_T{t}_seed{item["model_seed"]}.json',dict(config=cfg,result=ev,wandb_id=wr.id,wandb_url=wr.url))
                    wr.log({'independent_size':ev['overall']['empirical_size'],'independent_power':ev['overall']['empirical_power'],
                        'cells':wandb.Table(columns=['rho','beta','n','rejection_rate'],data=[[c[k] for k in ['rho','beta','n','rejection_rate']] for c in ev['by_parameter']])})
                    manifest['evaluation_runs'].append(dict(T=t,seed=item['model_seed'],id=wr.id,url=wr.url,status='finished'))
                save(OUT/'test_manifest.json',manifest)
                return [dict(T=t,rho=c['rho'],beta=c['beta'],seed=item['model_seed'],metric_type='size' if c['rho']==1 else 'power',n=c['n'],rejection_rate=c['rejection_rate']) for c in ev['by_parameter']]
            for rows in evaluate_checkpoints(raw,selected,evaluate_one): all_rows.extend(rows)
            require(fingerprint(raw)==fp,'Dataset mutated during evaluation')
        means,comparison,summaries,overall=compare(all_rows)
        write_csv(OUT/'rejection_rates_by_seed.csv',all_rows);write_csv(OUT/'rejection_rates_mean.csv',means)
        write_csv(OUT/'paper_comparison.csv',comparison);save(OUT/'summary_by_T.json',summaries)
        save(OUT/'final_reproduction_summary.json',overall)
        (OUT/'final_reproduction_report.md').write_text(render_report(protocol,comparison,summaries,overall))
        require(all(sha(i['path'])==i['sha256'] for i in items),'Checkpoint modified')
        manifest.update(status='complete',finished_at=now(),evaluation_count=9,cell_seed_records=144,comparison_cells=48,runtime_seconds=perf_counter()-start)
        save(OUT/'test_manifest.json',manifest)
        print(json.dumps(overall,indent=2),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),failed_at=now());save(OUT/'test_manifest.json',manifest);raise


def finalize(assessment,basis):
    require(read(OUT/'test_manifest.json')['status']=='complete','Evaluation incomplete')
    overall=read(OUT/'final_reproduction_summary.json')
    overall.update(assessment=assessment,assessment_basis=basis,assessment_kind='descriptive expert judgment; no formal numerical equivalence cutoff')
    save(OUT/'final_reproduction_summary.json',overall)
    with (OUT/'paper_comparison.csv').open() as f:
        comparison=list(csv.DictReader(f))
    for r in comparison:
        for k in r:
            if k!='metric_type':r[k]=int(r[k]) if k=='T' else float(r[k])
    (OUT/'final_reproduction_report.md').write_text(render_report(read(OUT/'protocol_comparison.json'),comparison,read(OUT/'summary_by_T.json'),overall))


def upload():
    require(read(OUT/'test_manifest.json')['status']=='complete','Evaluation incomplete')
    require(read(OUT/'final_reproduction_summary.json')['assessment'] is not None,'Assessment not recorded')
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    receipt=dict(status='uploading',artifact_name=ARTIFACT,files=files)
    receipt_path=ROOT/'independent_test_artifact_receipt.json'
    try:
        with wandb.init(project='URT_wNN',group=GROUP,job_type='artifact-upload',name='ANN_final_results_archive',config={'test_only':True,'training_performed':False},mode='online') as wr:
            artifact=wandb.Artifact(ARTIFACT,type='reproduction-results',metadata={'evaluation_only':True,'cells':48,'model_checkpoints':9,'test_seed':TEST_SEED})
            artifact.add_dir(str(OUT)); logged=wr.log_artifact(artifact);logged.wait()
            qualified=f'{wr.entity}/{wr.project}/{logged.name}'
            remote=wandb.Api().artifact(qualified,type='reproduction-results')
            require(set(remote.manifest.entries)==set(files),'Remote artifact file list mismatch')
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(remote.download(root=tmp))
                require(all(sha(root/n)==digest for n,digest in files.items()),'Artifact downloaded SHA256 mismatch')
            receipt.update(status='verified',qualified_name=qualified,version=remote.version,url=remote.url,
                verified_at=now(),verification='Remote manifest file set and downloaded SHA256 for every file',upload_run_url=wr.url)
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));save(receipt_path,receipt);raise
    save(receipt_path,receipt);print(json.dumps(receipt,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--upload',action='store_true')
    parser.add_argument('--assessment',choices=['exact / close numerical reproduction','qualitative reproduction with numerical discrepancy','substantive reproduction failure']);parser.add_argument('--basis')
    args=parser.parse_args()
    if args.preflight:
        protocol,items=preflight();print(json.dumps({'preflight':'passed','checkpoints':len(items),'training_candidates':protocol['selected_training']},indent=2));return
    if args.assessment:
        require(bool(args.basis),'Assessment basis required');finalize(args.assessment,args.basis)
    if args.upload: upload()
    elif not args.assessment: run_evaluation()


if __name__=='__main__':main()
