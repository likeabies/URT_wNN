import json, csv, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/Users/joganghyeon/URT_wNN')
manifest=root/'results/lstm_paper2019_extension/official_independent_test_vs_ann_20261018/test_manifest.json'
out=root/'analysis_outputs/lstm_official_checkpoint_convergence_20261019'
out.mkdir(exist_ok=False,parents=True)
m=json.loads(manifest.read_text())
expected={(t,w,s) for t,w in [(50,.55625),(100,.425),(250,.74375)] for s in [7,17,27]}
assert {(i['T'],i['w2'],i['model_seed']) for i in m['checkpoints']}==expected
series={}; provenance=[]; trajectory=[]; snapshots=[]; intervals=[]; trends=[]
for i in m['checkpoints']:
    p=root/Path(i['path']).parent/'history.json'
    h=json.loads(p.read_text())
    assert [r['epoch'] for r in h]==list(range(1,201))
    a=np.array([[r['train_loss'],r['train_acc']] for r in h])
    assert np.isfinite(a).all()
    series[i['T'],str(i['model_seed'])]=a
    provenance.append(dict(T=i['T'],w2=i['w2'],seed=i['model_seed'],checkpoint=i['path'],history=str(p),history_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for t in [50,100,250]:series[t,'mean']=np.mean([series[t,str(s)] for s in [7,17,27]],axis=0)
for (t,s),a in sorted(series.items()):
    for e,(loss,acc) in enumerate(a,1):trajectory.append(dict(T=t,seed=s,epoch=e,train_loss=loss,train_acc=acc))
    for e in [100,125,150,175,200]:snapshots.append(dict(T=t,seed=s,epoch=e,loss=a[e-1,0],accuracy_percent=100*a[e-1,1]))
    for start,end in [(100,150),(150,200),(175,200)]:
        drop=a[start-1,0]-a[end-1,0]
        intervals.append(dict(T=t,seed=s,start=start,end=end,loss_decrease=drop,relative_decrease_percent=100*drop/a[start-1,0],accuracy_change_pp=100*(a[end-1,1]-a[start-1,1])))
    for n in [20,50]:
        x=np.arange(201-n,201);y=a[-n:,0];slope,intercept=np.polyfit(x,y,1)
        residual=y-(slope*x+intercept);ss=np.sum((y-y.mean())**2)
        trends.append(dict(T=t,seed=s,last_n=n,first_epoch=int(x[0]),last_epoch=200,loss_slope_per_epoch=slope,
          fitted_loss_decrease=-slope*(n-1),fitted_relative_decrease_percent=-100*slope*(n-1)/y.mean(),
          r_squared=1-np.sum(residual**2)/ss if ss else 0.,residual_sd=float(np.sqrt(np.sum(residual**2)/(n-2))),
          first10_mean_loss=float(y[:10].mean()),last10_mean_loss=float(y[-10:].mean()),
          block_relative_decrease_percent=100*(y[:10].mean()-y[-10:].mean())/y[:10].mean(),
          block_accuracy_change_pp=100*(a[-10:,1].mean()-a[-n:-n+10,1].mean())))
def csvsave(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,rows in [('trajectory.csv',trajectory),('epoch_snapshots.csv',snapshots),('interval_changes.csv',intervals),('late_loss_trends.csv',trends)]:csvsave(name,rows)
(out/'provenance.json').write_text(json.dumps(dict(manifest=str(manifest),runs=provenance,training_performed=False,evaluation_performed=False),indent=2))
for start,name in [(1,'learning_curves'),(151,'late_learning_curves')]:
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for row,t in enumerate([50,100,250]):
        x=np.arange(start,201)
        for col,label in enumerate(['Training WCE loss','Training accuracy (%)']):
            ax=axes[row,col];scale=100 if col else 1
            for s in ['7','17','27']:ax.plot(x,scale*series[t,s][start-1:,col],lw=.9,alpha=.75,label='Seed '+s)
            ax.plot(x,scale*series[t,'mean'][start-1:,col],color='black',lw=1.8,label='3-seed mean')
            ax.set(title=f'T={t}',xlabel='Epoch',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def table(rows):
    keys=list(rows[0]);lines=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    for r in rows:lines.append('| '+' | '.join(f'{r[k]:.7g}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in keys)+' |')
    return '\n'.join(lines)
text=['# Official LSTM checkpoint 학습 history 분석','',
 '대상: T50/w2=.55625, T100/w2=.425, T250/w2=.74375; seeds 7/17/27. 기존 history 1–200만 분석. 새 학습·calibration·test 평가는 수행하지 않음.',
 '', '구간 loss 감소량 = loss(start) − loss(end); relative decrease = 감소량/loss(start) × 100. Accuracy 변화는 percentage points. 음의 감소량은 악화.',
 '', '마지막 50 epoch는 151–200, 마지막 20 epoch는 181–200. OLS slope는 loss/epoch 단위이며 fitted 감소량은 −slope × (n−1). R²와 residual SD는 기술 통계이며 serial correlation을 무시한 유의성 검정은 하지 않음. 첫/마지막 10-epoch 평균 비교로 단일 endpoint 변동을 보완.',
 '', '기록된 loss/accuracy는 epoch 동안 계속 갱신되는 모델의 shuffled training minibatch 평균이다. 고정 checkpoint를 전체 training set에서 재평가한 값이나 gradient norm이 아니다. WCE weight가 T마다 달라 T 간 loss 절대값 비교에는 의미를 부여하지 않음.',
 '', '## Seed별 및 평균 learning curves','', '![전체](learning_curves.png)','', '![후반](late_learning_curves.png)',
 '', '## 지정 epoch 값','',table(snapshots),'','## 구간별 변화','',table(intervals),
 '', '## 후반 loss 추세','',table(trends)]
(out/'report.md').write_text('\n'.join(text)+'\n')
print('OUTPUT',out)
print('MEAN SNAPSHOTS',json.dumps([r for r in snapshots if r['seed']=='mean']))
print('INTERVALS',json.dumps(intervals))
print('TRENDS',json.dumps(trends))
