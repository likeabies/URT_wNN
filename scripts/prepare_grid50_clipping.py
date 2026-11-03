"""One training-only fixed-checkpoint replay and copied failure-state validation."""
import json,math,hashlib,time
from pathlib import Path
import numpy as np
import torch
from src.variable_length_lstm import VariableLengthLSTM,load_data,batches,fingerprint
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy, predict
from src.grid50_safe_adam import Grid50SafeAdam
from src.grid50_gradient_clipping import global_norm64,clip_global_norm64

def main():
    start=time.perf_counter()
    out=Path('analysis_outputs/grid50_clipping_20261026_v2');out.mkdir(exist_ok=False)
    root=Path('results/lstm_multilength_grid50_equalN_safeadam_20261020')
    oldrun=root/'runs/w2_0.2_20261023_070342_117339'
    manifest=json.loads((root/'run_manifest.json').read_text())
    cp_path=oldrun/'epoch_010.pt'
    digest=hashlib.sha256(cp_path.read_bytes()).hexdigest()
    assert digest==json.loads(cp_path.with_suffix('.json').read_text())['sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    raw=load_data(manifest['datasets']['train']['path'])
    assert fingerprint(raw)==manifest['datasets']['train']['fingerprint']
    model=VariableLengthLSTM().to('mps');model.load_state_dict(cp['model']);model.train()
    norms=[];by_t={t:[] for t in range(50,501,50)};seen=np.zeros(600000,dtype=np.int32)
    # Fixed epoch-11 sampler schedule, fixed epoch-10 parameters; no optimizer.
    for i,(x,y,lengths,ids) in enumerate(batches(raw,256,model_seed=7,epoch=11),1):
        np.add.at(seen,ids,1);model.zero_grad(set_to_none=True)
        loss=ann_weighted_cross_entropy(model(x.to('mps')),y.to('mps'),1.,.2)
        assert torch.isfinite(loss)
        loss.backward();n=global_norm64(model.parameters())
        norms.append(n);by_t[int(lengths[0])].append(n)
        if i%500==0:print('Replay batches',i,flush=True)
    assert i==2350 and (seen==1).all() and all(len(v)==235 for v in by_t.values())
    assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
    def summary(values):
        return dict(zip(['min','median','p90','p95','p99','p99.9','max'],
                        map(float,np.percentile(values,[0,50,90,95,99,99.9,100]))))
    threshold=2*max(norms)
    result=dict(checkpoint=str(cp_path),checkpoint_sha256=digest,
        training_fingerprint=manifest['datasets']['train']['fingerprint'],
        sampler_epoch=11,model_seed=7,w2=.2,device='mps',norm_accumulation='cpu_float64',
        batches=2350,all_finite=True,coverage=600000,overall=summary(norms),
        by_T={str(t):summary(v) for t,v in by_t.items()},max_norm=threshold,
        rule='2 * maximum finite global L2 norm; frozen before new training')
    with (out/'threshold.json').open('x') as f:json.dump(result,f,indent=2)
    print('THRESHOLD',json.dumps(result),flush=True)
    failure=torch.load(oldrun/'pre_failure_state.pt',map_location='cpu',weights_only=False)
    ctx=failure['last_good']['context'];ids=np.array(ctx['indices']);t=ctx['T']
    x=torch.from_numpy(np.array(raw['x'][raw['offsets'][ids,None]+np.arange(t)],copy=True)).unsqueeze(-1).to('mps')
    y=torch.from_numpy(np.array(raw['y'][ids],copy=True)).to('mps')
    model.load_state_dict(failure['last_good']['model']);opt=Grid50SafeAdam(model.parameters())
    opt.load_state_dict(failure['last_good']['optimizer']);model.zero_grad(set_to_none=True)
    loss=ann_weighted_cross_entropy(model(x),y,1.,.2);loss.backward()
    before=global_norm64(model.parameters())
    copies=[torch.nn.Parameter(torch.zeros_like(p)) for p in model.parameters()]
    for q,p in zip(copies,model.parameters()):q.grad=p.grad.detach().clone()
    standard=torch.nn.utils.clip_grad_norm_(copies,threshold)
    standard_norm=float(standard.cpu())
    info=clip_global_norm64(model.parameters(),threshold)
    opt.step()
    info.update(standard_clip_norm=str(standard_norm),
                standard_clip_norm_finite=math.isfinite(standard_norm),
                gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                second_moment_finite=all(bool(torch.isfinite(s['exp_avg_sq']).all()) for s in opt.state.values()),
                optimizer_finite=all(bool(torch.isfinite(v).all()) for s in opt.state.values() for v in s.values()),
                elapsed_seconds=time.perf_counter()-start)
    assert all(info[k] for k in ['gradients_finite','parameters_finite','second_moment_finite','optimizer_finite'])
    assert info['clipped'] and before>threshold
    info['status']='passed'
    with (out/'failure_validation.json').open('x') as f:json.dump(info,f,indent=2)
    print('VALIDATION',json.dumps(info),flush=True)

if __name__=='__main__':main()
