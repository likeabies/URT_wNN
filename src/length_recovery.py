"""Fixed-length recovery models and paired copies of the official DGP."""
import numpy as np
import torch
from torch import nn
from src.data import normalize_ann_series
from scripts.run_ann_paper2019 import dataset, fingerprint


def paired_data(c, test=False, smoke=False):
    seed=int(np.random.SeedSequence([c['test_seed'] if test else c['data_seed'],250]).generate_state(1)[0])
    rng=np.random.default_rng(seed)
    rhos=sorted(set(c['rhos']+([.8] if test else [])),reverse=True)
    n0,n1=((2,2) if test else (10,2)) if smoke else ((1000,1000) if test else (5000,1000))
    n=sum(n0 if r==1 else n1 for r in rhos)*len(c['betas'])
    raw=np.empty((n,250),np.float64);x=np.empty((n,250),np.float32)
    y=np.empty(n,np.int64);rho=np.empty(n,np.float64);beta=np.empty(n,np.float64);pos=0
    for r in rhos:
        for b in c['betas']:
            count=n0 if r==1 else n1
            errors=rng.normal(size=(count,251));innov=b*errors[:,:-1]+errors[:,1:]
            values=np.empty((count,250),np.float64);previous=np.zeros(count)
            for t in range(250):
                previous=r*previous+innov[:,t];values[:,t]=previous
            sl=slice(pos,pos+count);raw[sl]=values;x[sl]=normalize_ann_series(values)
            y[sl]=int(r<1);rho[sl]=r;beta[sl]=b;pos+=count
    order=rng.permutation(n)
    pack=dict(x=x[order],y=y[order],rho=rho[order],beta=beta[order])
    reference,derived=dataset(c,250,test=test,smoke=smoke)
    assert derived==seed and fingerprint(pack)==fingerprint(reference)
    pack['raw']=raw[order]
    return pack,seed


def short_input(raw,t):
    return normalize_ann_series(raw[:,:t]).astype(np.float32)


class Encoder(nn.Module):
    def __init__(self, state=None):
        super().__init__();self.lstm=nn.LSTM(1,30,1,batch_first=True)
        if state is not None:self.lstm.load_state_dict(state)
    def forward(self,x):
        return self.lstm(x.unsqueeze(-1))[1][0][-1]


class Completion(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.decoder=nn.LSTM(1,30,1,batch_first=True);self.output=nn.Linear(30,1)
    def forward(self,prefix,decoder_input):
        state=self.encoder.lstm(prefix.unsqueeze(-1))[1]
        out,_=self.decoder(decoder_input.unsqueeze(-1),state)
        return self.output(out).squeeze(-1)
    def generate(self,prefix,steps):
        state=self.encoder.lstm(prefix.unsqueeze(-1))[1];last=prefix[:,-1:];parts=[]
        for _ in range(steps):
            out,state=self.decoder(last.unsqueeze(-1),state)
            last=self.output(out).squeeze(-1);parts.append(last)
        return torch.cat(parts,1)


class LogitStudent(nn.Module):
    def __init__(self,state):
        super().__init__();self.encoder=Encoder(state);self.fc=nn.Linear(30,2)
    def forward(self,x):return self.fc(self.encoder(x))
