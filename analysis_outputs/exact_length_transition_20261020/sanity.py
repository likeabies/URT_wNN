import json,copy
from pathlib import Path
import numpy as np
import torch
from src.variable_length_lstm import VariableLengthLSTM
from src.data import make_ann_dataset
from src.train import ann_weighted_cross_entropy
torch.set_num_threads(4)
assert torch.backends.mps.is_available()
rows=[]
for t in [50,250,500]:
    data=make_ann_dataset(t,[1.,.95],[.6],384,384,20191007+t)
    torch.manual_seed(7);cpu=VariableLengthLSTM();mps=VariableLengthLSTM().to('mps')
    mps.load_state_dict(cpu.state_dict())
    a=torch.optim.Adam(cpu.parameters(),lr=.001);b=torch.optim.Adam(mps.parameters(),lr=.001)
    ids=np.random.default_rng(7).permutation(768)
    for j in range(3):
        ix=ids[j*256:(j+1)*256]
        x=torch.from_numpy(data['x'][ix]).unsqueeze(-1);y=torch.from_numpy(data['y'][ix])
        a.zero_grad();b.zero_grad()
        ca=cpu(x);mb=mps(x.to('mps'))
        torch.testing.assert_close(ca,mb.cpu(),atol=2e-6,rtol=2e-4)
        ann_weighted_cross_entropy(ca,y,1.,.2).backward()
        ann_weighted_cross_entropy(mb,y.to('mps'),1.,.2).backward()
        allerr=0
        for (name,p),(_,q) in zip(cpu.named_parameters(),mps.named_parameters()):
            ga=p.grad.detach();gb=q.grad.cpu()
            assert torch.isfinite(gb).all()
            torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-4)
            allerr=max(allerr,(ga-gb).abs().max().item())
            if name=='lstm.weight_hh_l0':
                mask=(ga.abs()>1e-8)&(gb.abs()>1e-8)
                signs=int(((ga.sign()!=gb.sign())&mask).sum())
                cosine=torch.nn.functional.cosine_similarity(ga.flatten().double(),gb.flatten().double(),dim=0).item()
                assert signs==0 and cosine>.99999
                rec=dict(T=t,batch=j,hh_max_abs_error=(ga-gb).abs().max().item(),sign_mismatch=signs,cosine=cosine)
        a.step();b.step()
        assert all(torch.isfinite(v).all() for st in b.state.values() for v in st.values() if torch.is_tensor(v))
        assert all(torch.isfinite(p).all() for p in mps.parameters())
        rows.append(dict(rec,max_gradient_error=allerr,finite_adam=True))
        # Each comparison starts from identical parameters/optimizer, avoiding drift.
        mps.load_state_dict(cpu.state_dict());b.load_state_dict(copy.deepcopy(a.state_dict()))
report=dict(status='passed',path='h_n[-1] only',rows=rows)
Path('analysis_outputs/exact_length_transition_20261020/sanity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
