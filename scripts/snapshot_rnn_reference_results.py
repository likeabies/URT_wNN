"""Read-only ANN/LSTM evidence capture for the separate RNN comparison report."""
import csv
from pathlib import Path
import statistics as st
import wandb
from scripts.run_rnn_paper2019_fixed200 import read, save, aggregate, require, sha, immutable, ANN, CONFIG


def main():
    base=Path(read(CONFIG)['output_dir'])
    base.mkdir(parents=True,exist_ok=True)
    target=base/'reference_comparison.json'
    if target.exists():
        print('Existing frozen reference snapshot reused',target)
        return
    rows=[]
    evidence={}
    official=read(ANN/'independent_test/test_dataset_fingerprints.json')
    validation=read(ANN/'fixed200_validation_calibration/new_test_dataset_fingerprints.json')
    for t in (50,100,250):
        require(validation[str(t)]['test_seed']==20261008 and
                validation[str(t)]['fingerprint']==official[str(t)]['fingerprint'],'ANN reference test mismatch')
    train_weights={50:.375,100:.4375,250:.6375}
    val_weights={s['T']:s['selected_w2'] for s in read(ANN/'fixed200_validation_calibration/validation_candidates.json')}
    for method,path,weights in [
        ('training_grid',ANN/'independent_test/rejection_rates_by_seed.csv',train_weights),
        ('validation_grid',ANN/'fixed200_validation_calibration/new_test_rejection_rates_by_seed.csv',val_weights)]:
        cells=list(csv.DictReader(path.open()))
        require(len(cells)==144,'Expected ANN 3T x 3seed x 16cells')
        evidence[str(path)]=sha(path)
        for t in (50,100,250):
            group=[]
            for seed in (7,17,27):
                subset=[r for r in cells if int(r['T'])==t and int(r['seed'])==seed]
                require(len(subset)==16 and all(int(r['n'])==1000 for r in subset),'Invalid reference cells')
                group.append(dict(model_seed=seed,w2=weights[t],
                    size=st.mean(float(r['rejection_rate']) for r in subset if float(r['rho'])==1),
                    power=st.mean(float(r['rejection_rate']) for r in subset if float(r['rho'])!=1),
                    **{f'power_rho_{rho}':st.mean(float(r['rejection_rate']) for r in subset if float(r['rho'])==rho)
                       for rho in (.95,.9,.8)}))
            rows.append(aggregate('ANN',method,t,group))
    api=wandb.Api(timeout=60)
    runs=api.runs('rdces-chung-ang-university/URT_wNN',filters={'group':'lstm_fixed200_official_independent_test_vs_ann_20261018'})
    records=[]
    raw=[]
    for run in runs:
        cfg=dict(run.config)
        summary=dict(run.summary)
        if cfg.get('model')!='LSTM' or 'model_seed' not in cfg:
            continue
        require(run.state=='finished' and cfg['test_seed']==20261008 and cfg['epochs']==200,'Incomplete/nonmatching LSTM reference')
        t=cfg['T']
        require(cfg['test_fingerprint']==official[str(t)]['fingerprint'],'LSTM reference fingerprint mismatch')
        raw.append(dict(id=run.id,url=run.url,config=cfg,summary=summary,state=run.state))
        records.append(dict(T=t,model_seed=cfg['model_seed'],w2=cfg['w2'],size=summary['size'],power=summary['power'],
            **{f'power_rho_{rho}':summary[f'power_by_rho.{rho}'] for rho in (.95,.9,.8)}))
    require(len(records)==9 and len({(r['T'],r['model_seed']) for r in records})==9,'Expected nine unique LSTM official tests')
    for t in (50,100,250):
        group=[r for r in records if r['T']==t]
        require(sorted(r['model_seed'] for r in group)==[7,17,27],'LSTM seed coverage')
        rows.append(aggregate('LSTM','training_grid',t,group))
    immutable(base/'lstm_wandb_reference.json',raw)
    immutable(target,dict(rows=rows,local_source_hashes=evidence,official_test_seed=20261008,
        limitations=['LSTM held-out validation-based 및 seedwise-linear 완료 결과는 로컬/W&B 검색에서 확인되지 않아 해당 비교에서 제외했다. LSTM 재학습은 수행하지 않았다.',
                     'ANN/RNN/LSTM training-grid와 validation-grid는 별도 행으로 비교한다. Linear 방식의 RNN 결과는 별도로 표시한다.']))
    print('Reference snapshot saved: ANN training/validation + LSTM training, 9 rows')


if __name__=='__main__':
    main()
