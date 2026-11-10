"""Bounded forensic runs only. Never calibrates, selects weights, or tests."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import diagnose_variable_length as diagnosis
from scripts import run_lstm_variable_length as production
from src.variable_length_lstm import VariableLengthLSTM, batches, load_data
from src.train import ann_weighted_cross_entropy
from src.utils import set_seed

ORIGINAL=Path('results/lstm_variable_length_feasibility_20261019')
RECOVERY=Path('results/lstm_variable_length_feasibility_20261019_recovery')


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return copy.deepcopy(value)


def digest(value):
    h=hashlib.sha256()
    if isinstance(value,dict):
        for k,v in value.items():h.update(str(k).encode());h.update(digest(v).encode())
    else:
        a=value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def stats(t):
    a=t.detach().cpu();finite=torch.isfinite(a);v=a[finite].double()
    return dict(dtype=str(t.dtype),device=str(t.device),shape=list(t.shape),finite=int(finite.sum()),
        nan=int(torch.isnan(a).sum()),posinf=int(torch.isposinf(a).sum()),neginf=int(torch.isneginf(a).sum()),
        finite_max_abs=float(v.abs().max()) if v.numel() else None,
        finite_norm=float(torch.linalg.vector_norm(v)) if v.numel() else None)


def optimizer_stats(model,opt):
    return {name:{key:stats(value) for key,value in opt.state[p].items() if torch.is_tensor(value)}
            for name,p in model.named_parameters()}


def rng():
    return dict(torch=digest(torch.get_rng_state()),mps=digest(torch.mps.get_rng_state()),
                numpy=repr(np.random.get_state()),python=repr(__import__('random').getstate()))


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


class StopTrace(Exception):pass


def short_trace(out,kind,limit):
    """Instrument the actual entry paths, not copies of their training loop."""
    out.mkdir();state=json.loads((ORIGINAL/'run_manifest.json').read_text());c=state['protocol']
    raw=load_data(state['datasets']['train']['path']);cal=load_data(state['datasets']['calibration']['path'])
    original_step=torch.optim.Adam.step;original_forward=VariableLengthLSTM.forward
    original_init=VariableLengthLSTM.__init__;active={};rows=[];init=[]
    def model_init(self,*a,**kw):
        original_init(self,*a,**kw);active['model']=self
        init.append(dict(parameter_checksum=digest(self.state_dict()),rng=rng()))
    def batch_iter(*a,**kw):
        for x,y,l,idx in batches(*a,**kw):
            active['input']=(x.clone(),y.clone(),l.clone(),idx.copy());yield x,y,l,idx
    def forward(self,x,l):
        result=original_forward(self,x,l);active['logits']=cpu(result);return result
    def step(opt,*a,**kw):
        model=active['model'];x,y,l,idx=active['input'];logits=active['logits']
        grads={n:cpu(p.grad) for n,p in model.named_parameters()}
        before=cpu(model.state_dict());ostate=cpu(opt.state_dict())
        result=original_step(opt,*a,**kw)
        row=dict(batch=len(rows)+1,indices_checksum=digest(idx),x_checksum=digest(x),lengths_checksum=digest(l),
            initial_parameter_checksum=init[0]['parameter_checksum'],before_checksum=digest(before),
            loss=active['loss'],logits_checksum=digest(logits),
            gradient_checksum=digest(grads),gradient_stats={n:stats(g) for n,g in grads.items()},
            after_checksum=digest(model.state_dict()),optimizer_stats=optimizer_stats(model,opt),
            optimizer_defaults=opt.defaults,parameters_are_model_objects=all(a is b for a,b in zip(opt.param_groups[0]['params'],model.parameters())))
        rows.append(row)
        torch.save(dict(before=before,optimizer_before=ostate,after=cpu(model.state_dict()),
                        logits=logits,gradients=grads,optimizer_after=cpu(opt.state_dict())),out/f'batch_{len(rows):03d}.pt')
        write(out/'trace.json',dict(initializations=init,rows=rows))
        if len(rows)>=limit:raise StopTrace()
        return result
    def no_online(*a,**kw):return None
    module=diagnosis if kind=='diagnosis' else production
    original_loss=module.ann_weighted_cross_entropy
    def loss_call(*a,**kw):
        value=original_loss(*a,**kw);active['loss']=value.item();return value
    def redirected_path(value):
        return out/'actual_diagnosis_output' if str(value)=='results/lstm_variable_length_diagnosis_20261019' else Path(value)
    with patch.object(VariableLengthLSTM,'__init__',model_init),patch.object(VariableLengthLSTM,'forward',forward),\
         patch.object(torch.optim.Adam,'step',step),patch.object(module,'batches',batch_iter),\
         patch.object(module,'ann_weighted_cross_entropy',loss_call),\
         patch.object(diagnosis,'Path',redirected_path),patch.object(production.wandb,'init',no_online):
        try:
            if kind=='diagnosis':diagnosis.main()
            else:
                root=out/'actual_production_output';root.mkdir()
                production.train_candidate(root,c,dict(datasets=state['datasets'],runs=[]),raw,cal,.2,'coarse','mps')
        except StopTrace:pass


def find_failure(out,max_batches):
    out.mkdir();state=json.loads((RECOVERY/'run_manifest.json').read_text());c=state['protocol']
    raw=load_data(state['datasets']['train']['path']);set_seed(7)
    model=VariableLengthLSTM().to('mps');opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=0.)
    history=production.restore_epoch(state['progress']['checkpoint'],model,opt,c,'mps')
    write(out/'source.json',dict(checkpoint=state['progress']['checkpoint'],protocol=c))
    previous=None;count=0
    for epoch in range(len(history)+1,36):
        for batch,(x,y,l,idx) in enumerate(batches(raw,256,25,7,epoch),1):
            count+=1
            before=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()))
            x=x.to('mps');y=y.to('mps');l=l.to('mps');opt.zero_grad(set_to_none=True)
            logits=model(x,l);loss=ann_weighted_cross_entropy(logits,y,1.,.2);loss.backward()
            grads={n:cpu(p.grad) for n,p in model.named_parameters()}
            finite_grad=all(torch.isfinite(g).all().item() for g in grads.values())
            if finite_grad and torch.isfinite(loss).item():opt.step()
            states=optimizer_stats(model,opt)
            bad=[dict(parameter=n,state_key=k,**v) for n,ss in states.items() for k,v in ss.items() if v['nan']+v['posinf']+v['neginf']]
            record=dict(epoch=epoch,batch=batch,loss=str(loss.item()),gradient_finite=finite_grad,
                gradients={n:stats(g) for n,g in grads.items()},parameters={n:stats(p) for n,p in model.named_parameters()},
                optimizer=states,indices_checksum=digest(idx))
            snapshot=dict(before=before,x=cpu(x),y=cpu(y),lengths=cpu(l),indices=idx.copy(),logits=cpu(logits),
                gradients=grads,after=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict())),record=record)
            if bad or not finite_grad or not torch.isfinite(loss).item():
                torch.save(snapshot,out/'failure_batch.pt')
                if previous:torch.save(previous,out/'previous_batch.pt')
                write(out/'failure.json',dict(record,bad_optimizer_tensors=bad))
                print(json.dumps(dict(epoch=epoch,batch=batch,loss=str(loss.item()),bad=bad)),flush=True)
                return
            previous=snapshot
            if count%100==0:print(json.dumps(dict(epoch=epoch,batch=batch,max_gradient=max(v.abs().max().item() for v in grads.values()))),flush=True)
            if count>=max_batches:
                write(out/'bounded_no_failure.json',dict(epoch=epoch,batch=batch,batches=count));return


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compare','failure']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-batches',type=int,default=2353);args=p.parse_args();args.output.mkdir(exist_ok=False)
    if args.mode=='compare':
        short_trace(args.output/'diagnosis_a','diagnosis',20)
        short_trace(args.output/'production','production',20)
        short_trace(args.output/'diagnosis_b','diagnosis',20)
    else:find_failure(args.output/'recovery_epoch6',args.max_batches)


if __name__=='__main__':main()
