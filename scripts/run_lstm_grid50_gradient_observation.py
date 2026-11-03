"""Fixed w2, fresh 200-epoch observation; one scalar-only batch CSV."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from src.variable_length_lstm import VariableLengthLSTM, generate, batches
from src.grid50_safe_adam import Grid50SafeAdam
from src.grid50_gradient_clipping import clip_global_norm64
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy, predict
from src.utils import set_seed

ROOT=Path('results/lstm_grid50_gradient_observation_20261026')
FIELDS=['epoch','batch','T','loss','pre_clip_global_grad_norm',
        'post_clip_global_grad_norm','clip_coefficient','clipped','pre_clip_max_abs_grad']
STOP=False

def stop(signum,frame):
    global STOP
    STOP=True

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)

def telemetry(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except Exception as exc:print('W&B warning:',repr(exc),flush=True)

def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',type=int,choices=range(1,9),default=1)
    args=parser.parse_args()
    conditions=[(.2,256),(.3,256),(.2,128),(.4,256),(.3,128),(.2,512),(.4,128),(.3,512)]
    w2,batch_size=conditions[args.condition-1]
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert torch.backends.mps.is_available()
    c=json.loads(Path('configs/lstm_grid50_gradient_observation_20261026.json').read_text())
    if args.condition!=1:
        ROOT=Path(f'results/lstm_grid50_hourly_c{args.condition}_w2_{w2:.2f}_bs{batch_size}')
        c.update(experiment=ROOT.name,w2=w2,batch_size=batch_size)
    ROOT.mkdir(parents=True,exist_ok=False)
    state=dict(status='initializing',protocol=c,device='mps',completed_epoch=0,pid=os.getpid(),
               source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [__file__,'src/variable_length_lstm.py','src/grid50_safe_adam.py',
                 'src/grid50_gradient_clipping.py','src/train.py','src/model.py','src/lstm_binary.py','src/data.py']})
    save(ROOT/'run_manifest.json',state)
    wr=None;context={}
    def finite(condition,stage):
        if not condition:
            save(ROOT/'numeric_failure.json',dict(stage=stage,**context))
            raise FloatingPointError(stage)
    try:
        raw,meta=generate(ROOT/'datasets/train',c,'train')
        assert meta['n']==600000 and all(v['n']==60000 for v in meta['by_T'].values())
        state['training_data']=meta
        set_seed(c['model_seed'])
        model=VariableLengthLSTM().to('mps')
        opt=Grid50SafeAdam(model.parameters(),lr=c['lr'])
        params=list(model.parameters())
        wr=telemetry(wandb.init,project='URT_wNN',group=c['experiment'],name=c['experiment']+f'_w2_{w2:.2f}_ms7',
                     config=dict(c,training_fingerprint=meta['fingerprint']),mode='online')
        state.update(status='training',wandb_url=wr.url if wr else None,wandb_id=wr.id if wr else None)
        save(ROOT/'run_manifest.json',state)
        print(f'PID={os.getpid()} fresh model epoch=1 w2={w2} batch_size={batch_size} MPS; W&B={state["wandb_url"]}',flush=True)
        with (ROOT/'batch_gradients.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();stream.flush()
            for epoch in range(1,201):
                model.train();torch.mps.synchronize();start=perf_counter()
                seen=np.zeros(600000,dtype=np.int8);total_loss=0.;correct=0;total=0
                for batch,(x,y,lengths,ids) in enumerate(batches(raw,batch_size,model_seed=7,epoch=epoch),1):
                    np.add.at(seen,ids,1)
                    assert bool((lengths==x.shape[1]).all())
                    context=dict(epoch=epoch,batch=batch,T=int(x.shape[1]),w2=w2)
                    x=x.to('mps');y=y.to('mps');opt.zero_grad(set_to_none=True)
                    logits=model(x);loss=ann_weighted_cross_entropy(logits,y,1.,w2)
                    loss_value=loss.item()
                    context['loss']=loss_value if math.isfinite(loss_value) else str(loss_value)
                    finite(math.isfinite(loss_value),'nonfinite_loss')
                    loss.backward()
                    grads=torch.cat([p.grad.detach().flatten() for p in params])
                    finite(bool(torch.isfinite(grads).all()),'nonfinite_gradient_after_backward')
                    pre_max=grads.abs().max().item()
                    clip=clip_global_norm64(params,c['max_norm'])
                    finite(all(bool(torch.isfinite(p.grad).all()) for p in params),'nonfinite_gradient_after_clipping')
                    context.update(clip)
                    writer.writerow(dict(epoch=epoch,batch=batch,T=x.shape[1],loss=loss_value,
                        pre_clip_global_grad_norm=clip['pre_clip_global_norm'],
                        post_clip_global_grad_norm=clip['post_clip_global_norm'],
                        clip_coefficient=clip['clip_coefficient'],clipped=int(clip['clipped']),
                        pre_clip_max_abs_grad=pre_max))
                    stream.flush()
                    opt.step()
                    finite(all(bool(torch.isfinite(p).all()) for p in params),'nonfinite_parameter_after_step')
                    finite(all(bool(torch.isfinite(v).all()) for s in opt.state.values()
                               for v in s.values() if torch.is_tensor(v)),'nonfinite_adam_state_after_step')
                    total_loss+=loss_value*len(y);correct+=(predict(logits)==y).sum().item();total+=len(y)
                assert batch==10*math.ceil(60000/batch_size) and bool((seen==1).all()) and total==600000
                torch.mps.synchronize();seconds=perf_counter()-start
                checkpoint=ROOT/f'epoch_{epoch:03d}.pt';tmp=checkpoint.with_suffix('.tmp')
                assert not checkpoint.exists()
                with tmp.open('xb') as f:
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                        torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        shuffle=dict(model_seed=7,next_epoch=epoch+1,
                                     algorithm='numpy.default_rng(SeedSequence([model_seed,epoch,917]))'),
                        protocol=c),f)
                    f.flush();os.fsync(f.fileno())
                tmp.replace(checkpoint)
                state.update(completed_epoch=epoch,last_epoch_seconds=seconds,checkpoint=str(checkpoint),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                save(ROOT/'run_manifest.json',state)
                if wr:telemetry(wr.log,dict(epoch=epoch,train_loss=total_loss/total,train_acc=correct/total,
                                           train_seconds=seconds))
                print(f'Completed epoch={epoch} seconds={seconds:.3f}; samples={total}; checkpoint saved',flush=True)
                if STOP:
                    state['status']='paused_clean';save(ROOT/'run_manifest.json',state);return
        state['status']='complete';save(ROOT/'run_manifest.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save(ROOT/'run_manifest.json',state)
        raise
    finally:
        if wr:telemetry(wr.finish)

if __name__=='__main__':
    for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    main()
