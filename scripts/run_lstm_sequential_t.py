"""Sequential-T paired control; unchanged safe Adam and clipping training loop."""
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
from src.variable_length_lstm import VariableLengthLSTM
from src.sequential_t_lstm import generate_shared as generate, batches
from src.grid50_safe_adam import Grid50SafeAdam
from src.grid50_gradient_clipping import clip_global_norm64
from src.train import ann_weighted_cross_entropy
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
    parser.add_argument('--candidate',choices=['ASC-020','DESC-020','ASC-030','DESC-030','ASC-040','DESC-040','ASC-025','DESC-025','ASC-035','DESC-035'],required=True)
    parser.add_argument('--config',default='configs/lstm_sequential_t_grid25_20261027.json')
    args=parser.parse_args()
    w2=int(args.candidate.split('-')[1])/100
    batch_size=256
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert torch.backends.mps.is_available()
    c=json.loads(Path(args.config).read_text())
    c.update(w2=w2,ordering='ascending' if args.candidate.startswith('ASC') else 'descending',candidate=args.candidate)
    ROOT=Path('results')/c['experiment']/args.candidate
    ROOT.mkdir(parents=True,exist_ok=False)
    state=dict(status='initializing',protocol=c,device='mps',completed_epoch=0,pid=os.getpid(),
               source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [__file__,'src/sequential_t_lstm.py','src/variable_length_lstm.py','src/grid50_safe_adam.py',
                 'src/grid50_gradient_clipping.py','src/train.py','src/model.py','src/data.py']})
    save(ROOT/'run_manifest.json',state)
    wr=None;context={}
    def finite(condition,stage):
        if not condition:
            save(ROOT/'numeric_failure.json',dict(stage=stage,**context))
            raise FloatingPointError(stage)
    try:
        raw,meta=generate(Path(c.get('training_dataset',ROOT.parent/'datasets/train')),c,'train')
        assert meta['n']==600000 and all(v['n']==c['allocation'][t] and v['unit_root']==v['stationary']==v['n']//2 for t,v in meta['by_T'].items())
        state['training_data']=meta
        set_seed(c['model_seed'])
        model=VariableLengthLSTM().to('mps')
        opt=Grid50SafeAdam(model.parameters(),lr=c['lr'])
        params=list(model.parameters())
        wr=telemetry(wandb.init,project='URT_wNN',group=c['experiment'],name=args.candidate+'_ms7',
                     config=dict(c,training_fingerprint=meta['fingerprint']),mode='online')
        state.update(status='training',wandb_url=wr.url if wr else None,wandb_id=wr.id if wr else None)
        save(ROOT/'run_manifest.json',state)
        print(f'PID={os.getpid()} fresh model epoch=1 w2={w2} batch_size={batch_size} MPS; W&B={state["wandb_url"]}',flush=True)
        with (ROOT/'batch_gradients.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();stream.flush()
            for epoch in range(1,201):
                model.train();torch.mps.synchronize();start=perf_counter()
                seen=np.zeros(600000,dtype=np.int8);total_loss=0.;correct=0;total=0
                for batch,(x,y,lengths,ids) in enumerate(batches(raw,batch_size,model_seed=7,epoch=epoch,ordering=c['ordering']),1):
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
                    total_loss+=loss_value*len(y);correct+=(logits.argmax(1)==y).sum().item();total+=len(y)
                assert batch==sum(math.ceil(n/batch_size) for n in c['allocation'].values()) and bool((seen==1).all()) and total==600000
                torch.mps.synchronize();seconds=perf_counter()-start
                checkpoint=ROOT/f'epoch_{epoch:03d}.pt';tmp=checkpoint.with_suffix('.tmp')
                assert not checkpoint.exists()
                with tmp.open('xb') as f:
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                        torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        shuffle=dict(model_seed=7,next_epoch=epoch+1,
                                     algorithm='numpy.default_rng(SeedSequence([model_seed,epoch,917])); within-T shuffle only', ordering=c['ordering']),
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
