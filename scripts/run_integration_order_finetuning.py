"""Independent A/B/C fine-tunes from frozen original_chained I3 baselines.

No old result is written. Calibration is the corrected lower-null-tail rule;
missing I3 calibration is reported, never filled using independent test data.
"""
import argparse
import hashlib
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.integration_order_lstm_I3 import IntegrationOrderLSTM, balanced_weights, loss_parts, fingerprint
from src.integration_order_finetuning import calibrate, boundary_margin, sample_indices, stage_metrics, stage1_weights, stage1_only_loss
from src.utils import set_seed

KEYS = ['level','difference','second_difference','d','rho','beta']
MODES = ['hard_only','hard_replay','hard_margin']
BASE = Path('results/integration_order_I3_4arch_20261110/original_chained')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(path):
    raw = {k:np.load(path/(k+'.npy'), mmap_mode='r') for k in KEYS}
    meta = json.loads((path/'dataset.json').read_text())
    if fingerprint(raw) != meta['fingerprint']:
        raise ValueError(f'Dataset fingerprint mismatch: {path}')
    return raw, meta


def forward_hidden(model, args):
    hidden = []
    handle = model.lstm.register_forward_hook(lambda module, inputs, outputs: hidden.append(outputs[1][0][-1]))
    try:
        logits = model(*args)
    finally:
        handle.remove()
    return logits, hidden[0]


def evaluate(model, raw, batch, device):
    model.eval()
    scores, hidden = [], []
    with torch.inference_mode():
        for start in range(0, len(raw['d']), batch):
            args = [torch.tensor(raw[k][start:start+batch],device=device) for k in KEYS[:3]]
            logits, z = forward_hidden(model, args)
            if not torch.isfinite(logits).all() or not torch.isfinite(z).all():
                raise FloatingPointError('Nonfinite evaluation')
            scores.append(logits.sigmoid().cpu().numpy()); hidden.append(z.cpu().numpy())
    return np.concatenate(scores), np.concatenate(hidden)


def atomic_checkpoint(path, payload):
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def check_finite(model, optimizer, stage):
    arrays = [p.grad if stage=='backward' else p for p in model.parameters()]
    if stage!='backward':
        arrays += [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    if any(a is not None and not bool(torch.isfinite(a).all()) for a in arrays):
        raise FloatingPointError('Nonfinite '+stage)


def fit(mode, t, root, raw, checkpoint, cfg, rounds, device):
    out = root/('finetune_'+mode)/f'T{t}'
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(cfg, fine_tuning=dict(mode=mode, rounds=rounds, samples_per_round=230400,
        updates_per_round=900, total_updates=900*rounds, lambda_margin=.1 if mode=='hard_margin' else 0.,
        margin=.5, pairing='nearest opposite-label per anchor; normalized L2 squared hinge',
        training_data='resample existing training cells with replacement; no new DGP draws',
        balance='unchanged N/(2*n_class) formula, recomputed for sampled cohort',
        positive_orders=[1,2,3] if mode=='hard_margin' else [1],
        trainable='all parameters' if mode=='hard_margin' else 'shared lstm + stage1 only; heads2/3 frozen',
        classification_loss='3-stage balanced BCE sum' if mode=='hard_margin' else 'Stage1 balanced BCE only',
        margin_candidates='d=0 or d=1 only', optimizer_start='restore baseline Adam state',
        seed_policy='set_seed(model_seed); resampling default_rng(SeedSequence([data_seed,T]).generate_state(1)[0])',
        baseline_checkpoint=str(checkpoint.resolve()), baseline_sha256=sha(checkpoint)))
    save(out/'config.json',protocol)
    state=dict(status='initializing',mode=mode,T=t,pid=os.getpid(),completed_round=0,updates=0,started_at=now())
    save(out/'run_manifest.json',state)
    set_seed(cfg['model_seed'])
    torch.mps.manual_seed(cfg['model_seed'])
    model=IntegrationOrderLSTM().to(device)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert cp['epoch']==200 and cp['T']==t
    model.load_state_dict(cp['model'])
    if mode != 'hard_margin':
        for head in (model.stage2,model.stage3):
            head.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(cp['optimizer'])
    assert optimizer.param_groups[0]['lr']==cfg['lr'] and cfg['batch_size']==256
    derived=int(np.random.SeedSequence([cfg['data_seed'],t]).generate_state(1)[0])
    rng=np.random.default_rng(derived)
    history=[]
    try:
        for epoch in range(1,rounds+1):
            tick=perf_counter();model.train()
            idx=sample_indices(raw,mode,cfg['rhos_I0'],cfg['betas'],rng)
            weights=(balanced_weights(raw['d'][idx]) if mode=='hard_margin' else stage1_weights(raw['d'][idx]))
            if epoch==1:
                save(out/'sampling.json',dict(n=len(idx),unique_n=len(np.unique(idx)),derived_seed=derived,
                    counts={str(d):int((raw['d'][idx]==d).sum()) for d in range(4)},
                    rho_counts={str(r):int((raw['rho'][idx]==r).sum()) for r in sorted(np.unique(raw['rho']))},
                    loss_weights=weights.tolist()))
            weight=torch.tensor(weights,device=device);totals=np.zeros(5)
            for b,start in enumerate(range(0,len(idx),cfg['batch_size']),1):
                selected=idx[start:start+cfg['batch_size']]
                args=[torch.tensor(raw[k][selected],device=device) for k in (KEYS[:3] if mode=='hard_margin' else KEYS[:1])]
                d=torch.tensor(raw['d'][selected],device=device)
                optimizer.zero_grad(set_to_none=True)
                if mode=='hard_margin':
                    logits,z=forward_hidden(model,args)
                    parts=loss_parts(logits,d,weight)
                    auxiliary=boundary_margin(z,d,.5)
                    loss=parts.sum()+.1*auxiliary
                else:
                    loss=stage1_only_loss(model,args[0],d,weight)
                    auxiliary=loss.detach()*0
                    parts=torch.stack((loss,auxiliary,auxiliary))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();check_finite(model,optimizer,'backward')
                optimizer.step();check_finite(model,optimizer,'optimizer_step')
                totals+=np.array([*parts.detach().cpu().tolist(),float(auxiliary.detach()),float(loss.detach())])
                state['updates']+=1
                if b%100==0:
                    state.update(status='training',round=epoch,batch=b)
                    save(out/'run_manifest.json',state)
            torch.mps.synchronize()
            row=dict(round=epoch,updates=state['updates'],seconds=perf_counter()-tick,
                     loss1=totals[0]/b,loss2=totals[1]/b,loss3=totals[2]/b,
                     margin_loss=totals[3]/b,total_loss=totals[4]/b)
            history.append(row);save(out/'history.json',history)
            path=out/f'round_{epoch:03d}.pt'
            atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                completed_round=epoch,updates=state['updates'],torch_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state(),sampling_rng=rng.bit_generator.state,protocol=protocol,T=t))
            state.update(status='training',completed_round=epoch,checkpoint=str(path),last_round_seconds=row['seconds'])
            save(out/'run_manifest.json',state);print(mode,t,json.dumps(row),flush=True)
        assert state['updates']==rounds*900
        if mode!='hard_margin':
            for name,value in model.state_dict().items():
                if name.startswith(('stage2.','stage3.')):
                    assert torch.equal(value.cpu(),cp['model'][name]), name
            state['frozen_heads_unchanged']=True
        state.update(status='trained');save(out/'run_manifest.json',state)
        return model,out,state
    except Exception as e:
        state.update(status='failed',error=repr(e),round=epoch,batch=locals().get('b'),finished_at=now())
        save(out/'run_manifest.json',state)
        if isinstance(e,FloatingPointError):save(out/'numeric_failure.json',state)
        raise


def latent_stats(hidden,raw):
    z=hidden.astype(np.float64);z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    null=z[raw['d']==1];center=null.mean(0)
    result=[]
    for rho in sorted(np.unique(raw['rho'][raw['d']==0])):
        a=z[(raw['d']==0)&(raw['rho']==rho)];centroid=a.mean(0)
        diff=centroid-center;dist=float(np.linalg.norm(diff));axis=diff/max(dist,1e-12)
        pa,pn=a@axis,null@axis;edges=np.linspace(min(pa.min(),pn.min()),max(pa.max(),pn.max())+1e-12,41)
        ha=np.histogram(pa,edges)[0]/len(pa);hn=np.histogram(pn,edges)[0]/len(pn)
        radius=np.sqrt((((a-centroid)**2).sum(1).mean()+((null-center)**2).sum(1).mean())/2)
        result.append(dict(rho=float(rho),centroid=centroid.tolist(),null_centroid=center.tolist(),
            normalized_centroid_distance=dist,separation_over_pooled_radius=float(dist/max(radius,1e-12)),
            centroid_axis_histogram_overlap=float(np.minimum(ha,hn).sum())))
    return result


def compare(root,t,test,entries):
    paths=[root/'baseline'/f'T{t}']+[root/('finetune_'+m)/f'T{t}' for m in MODES]
    mask=(test['d']==0)|(test['d']==1)
    allz=[np.load(p/'stage1_hidden.npy')[mask].astype(np.float64) for p in paths]
    # A single joint PCA basis, identical axes across the four saved models.
    pooled=np.concatenate(allz);mean=pooled.mean(0)
    _,_,v=np.linalg.svd(pooled-mean,full_matrices=False);basis=v[:2].T
    np.savez(root/f'pca_basis_T{t}.npz',mean=mean,basis=basis)
    fig,axes=plt.subplots(1,4,figsize=(20,5),sharex=True,sharey=True)
    ds=test['d'][mask];rs=test['rho'][mask];colors=plt.cm.viridis(np.linspace(0,1,6))
    for ax,z,name,p in zip(axes,allz,['baseline']+MODES,paths):
        xy=(z-mean)@basis;np.save(p/'pca_coordinates.npy',xy)
        for j,r in enumerate([.2,.5,.9,.95,.99,1.]):
            keep=np.flatnonzero((ds==1) if r==1 else ((ds==0)&(rs==r)))
            # Fixed sample indices across all models; plots use at most 800 per rho.
            take=keep[np.linspace(0,len(keep)-1,min(800,len(keep)),dtype=int)]
            ax.scatter(xy[take,0],xy[take,1],s=3,alpha=.25,color=colors[j],label=str(r))
        ax.set_title(name);ax.set_xlabel('Joint PC1');ax.set_ylabel('Joint PC2')
        stats=latent_stats(np.load(p/'stage1_hidden.npy'),test);save(p/'latent_centroids.json',stats)
    axes[-1].legend(markerscale=3);fig.tight_layout();fig.savefig(root/f'latent_comparison_T{t}.png',dpi=160);plt.close(fig)
    for p in paths:shutil.copyfile(root/f'latent_comparison_T{t}.png',p/'pca_comparison.png')
    save(root/f'comparison_T{t}.json',entries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=10);a=parser.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; CPU fallback forbidden')
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'source_snapshot';snapshot.mkdir()
    for source in [Path(__file__),Path('src/integration_order_finetuning.py'),Path('src/integration_order_lstm_I3.py'),Path('src/data.py')]:
        shutil.copyfile(source,snapshot/source.name)
    queue=dict(status='running',pid=os.getpid(),started_at=now(),runs=[],rounds=a.rounds)
    save(a.output/'queue_manifest.json',queue)
    expected={50:(.15881532,.99991512),100:(.10228259,.99998891),250:(.17177728,.99999976)}
    try:
        for t in (50,100,250):
            base=BASE/f'T{t}';manifest=json.loads((base/'run_manifest.json').read_text());cfg=manifest['protocol']
            assert manifest['status']=='complete' and manifest['completed_epoch']==200
            assert cfg['architecture']=='original_chained' and cfg['model_seed']==7 and cfg['alpha']==.05
            train,tm=cohort(base/'train');cal,cm=cohort(base/'calibration');test,em=cohort(base/'test')
            assert [tm['seed'],cm['seed'],em['seed']]==[20191007,20261009,20261008]
            checkpoint=base/'epoch_200.pt';original_sha=sha(checkpoint)
            out=a.output/'baseline'/f'T{t}';out.mkdir(parents=True)
            save(out/'provenance.json',dict(baseline=str(base.resolve()),checkpoint_sha256=original_sha,
                train=tm,calibration=cm,test=em,protocol=cfg))
            thresholds=calibrate(np.load(base/'calibration_probabilities.npy'),cal['d'],cfg['alpha'])
            assert np.allclose([thresholds['c1']['threshold'],thresholds['c2']['threshold']],expected[t],rtol=0,atol=5e-9)
            save(out/'corrected_thresholds.json',thresholds)
            model=IntegrationOrderLSTM().to('mps');cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(cp['model'])
            probs,z=evaluate(model,test,cfg['batch_size'],'mps')
            stored=np.load(base/'test_probabilities.npy')
            if not np.allclose(probs,stored,atol=2e-6,rtol=1e-5):raise ValueError('Baseline checkpoint/score mismatch')
            np.save(out/'stage1_hidden.npy',z);np.save(out/'test_probabilities.npy',stored)
            baseline=stage_metrics(stored,test,thresholds);save(out/'metrics.json',baseline)
            print('BASELINE VERIFIED',t,json.dumps({k:v for k,v in baseline.items() if k in ['stage1_size','stage1_I0_power','rho_power']}),flush=True)
            entries={'baseline':baseline}
            del model,cp,probs,z
            for mode in MODES:
                queue.update(current_T=t,current_mode=mode);save(a.output/'queue_manifest.json',queue)
                model,out,state=fit(mode,t,a.output,train,checkpoint,cfg,a.rounds,'mps')
                state['status']='calibration';save(out/'run_manifest.json',state)
                probs,_=evaluate(model,cal,cfg['batch_size'],'mps')
                thresholds=calibrate(probs,cal['d'],cfg['alpha'])
                thresholds.update(selected_at=now(),cohort_fingerprint=cm['fingerprint'],test_used_for_selection=False)
                save(out/'corrected_thresholds.json',thresholds);np.save(out/'calibration_probabilities.npy',probs)
                state['status']='evaluation';save(out/'run_manifest.json',state)
                probs,z=evaluate(model,test,cfg['batch_size'],'mps')
                np.save(out/'test_probabilities.npy',probs);np.save(out/'stage1_hidden.npy',z)
                result=stage_metrics(probs,test,thresholds)
                result.update(rho_power_difference={r:p-baseline['rho_power'][r] for r,p in result['rho_power'].items()},
                              test_fingerprint=em['fingerprint'],test_sample_order='unchanged original test array order')
                save(out/'metrics.json',result);entries[mode]=result
                state.update(status='complete',finished_at=now());save(out/'run_manifest.json',state)
                queue['runs'].append(dict(mode=mode,T=t,status='complete',output=str(out)))
                save(a.output/'queue_manifest.json',queue)
                del model,probs,z;torch.mps.empty_cache()
            compare(a.output,t,test,entries)
            assert sha(checkpoint)==original_sha
        queue.update(status='complete',finished_at=now());save(a.output/'queue_manifest.json',queue)
        print('ALL FINE-TUNES COMPLETE',flush=True)
    except Exception as e:
        queue.update(status='failed',error=repr(e),finished_at=now())
        save(a.output/'queue_manifest.json',queue)
        (a.output/'failure_traceback.log').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
