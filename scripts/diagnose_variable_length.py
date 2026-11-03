"""Replay the original training protocol, stopping at the first numeric failure."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from src.utils import set_seed
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy, predict
from src.variable_length_lstm import VariableLengthLSTM, batches, load_data


def main():
    root=Path('results/lstm_variable_length_diagnosis_20261019')
    root.mkdir(exist_ok=False)
    old=json.loads(Path('results/lstm_variable_length_feasibility_20261019/run_manifest.json').read_text())
    c=old['protocol'];raw=load_data(old['datasets']['train']['path'])
    (root/'provenance.json').write_text(json.dumps(old,indent=2))
    set_seed(7);model=VariableLengthLSTM().to('mps')
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=0.)
    history=[]
    def checkpoint(name,epoch,**extra):
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                   torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),
                   sampler=dict(model_seed=7,next_epoch=epoch+1,salt=917),history=history,**extra)
        tmp=root/(name+'.tmp');torch.save(state,tmp);tmp.replace(root/name)
    for epoch in range(1,36):
        seen=np.zeros(len(raw['y']),dtype=np.int32);total=0.;correct=0;start=perf_counter()
        for batch,(x,y,l,idx) in enumerate(batches(raw,256,25,7,epoch),1):
            np.add.at(seen,idx,1)
            x=x.to('mps');y=y.to('mps');l=l.to('mps')
            opt.zero_grad(set_to_none=True);logits=model(x,l)
            loss=ann_weighted_cross_entropy(logits,y,1.,.2);loss.backward()
            grad=torch.cat([p.grad.flatten() for p in model.parameters()])
            if not torch.isfinite(grad).all().item() or not torch.isfinite(loss).item():
                checkpoint('failure.pt',epoch-1,failure_epoch=epoch,batch=batch,indices=idx.tolist(),
                           x=x.cpu(),y=y.cpu(),lengths=l.cpu())
                cpu=VariableLengthLSTM();cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()})
                cl=ann_weighted_cross_entropy(cpu(x.cpu(),l.cpu()),y.cpu(),1.,.2);cl.backward()
                report=dict(epoch=epoch,batch=batch,loss=str(loss.item()),cpu_loss=str(cl.item()),
                            processed_n=int(seen.sum()),unique_n=int((seen>0).sum()),
                            parameters_finite=all(torch.isfinite(p).all().item() for p in model.parameters()),
                            gradients={k:dict(mps_finite=bool(torch.isfinite(p.grad).all().item()),
                                cpu_finite=bool(torch.isfinite(q.grad).all().item()),
                                mps_max=str(p.grad.abs().max().item()),cpu_max=str(q.grad.abs().max().item()))
                                for (k,p),q in zip(model.named_parameters(),cpu.parameters())})
                (root/'failure.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
                return
            opt.step();total+=loss.item()*len(y);correct+=(predict(logits)==y).sum().item()
        row=dict(epoch=epoch,loss=total/len(seen),accuracy=correct/len(seen),processed_n=int(seen.sum()),
                 unique_n=int((seen>0).sum()),missing_n=int((seen==0).sum()),duplicate_n=int((seen>1).sum()),
                 batch_count=batch,seconds=perf_counter()-start)
        assert (seen==1).all() and np.isfinite(row['loss'])
        history.append(row);checkpoint(f'epoch_{epoch:03d}.pt',epoch)
        (root/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
