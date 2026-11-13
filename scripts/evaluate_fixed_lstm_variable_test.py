"""Frozen official fixed-T seed-7 LSTMs on saved, actual-length test inputs."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from src.model import Paper2019LSTMClassifier
from src.variable_length_lstm import load_data,fingerprint,batches
from src.lstm_binary import binary_metrics, predict, from_legacy

OUT=Path('results/fixed_lstm_variable_test_seed7_20261029')
TEST=Path('results/shared_lstm_independent_evaluation_20261029/variable_test')
OFFICIAL=Path('results/lstm_paper2019_extension/official_independent_test_vs_ann_20261018/test_manifest.json')
FP='fbbbb1d22afb89cc6cbdc6a3f6b33ce9172ceb6b85a82d49cc9e4ad483040953'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def metrics(y,p):
 d=binary_metrics(y,p)
 return dict(H0_N=int((y==1).sum()),H1_N=int((y==0).sum()),size=d['empirical_size'],power=d['empirical_power'])

def main():
 m=json.loads(OFFICIAL.read_text());assert m['status']=='complete'
 items=sorted([i for i in m['checkpoints'] if i['model_seed']==7],key=lambda i:i['T'])
 assert [i['T'] for i in items]==[50,100,250]
 raw=load_data(TEST);assert fingerprint(raw)==FP
 assert len(raw['y'])==24000 and (raw['y']==1).sum()==12000 and (raw['y']==0).sum()==12000
 assert np.array_equal(np.unique(raw['lengths']),np.arange(50,501))
 for i in items:assert sha(i['path'])==i['sha256']
 old_manifest_hash=sha(OFFICIAL)
 OUT.mkdir(exist_ok=False)
 save(OUT/'protocol.json',dict(checkpoints=items,test_path=str(TEST),fingerprint=FP,test_seed=20261008,
     model_seed=7,selection='Same model seed as preceding shared-model evaluation; no test-based selection',
     input='Actual sequence length, exact-length batches; no padding/packing/truncation/resampling',
     classifier='Original Paper2019LSTMClassifier; h_n[-1]',training_performed=False,calibration_performed=False))
 overall=[];exact=[];bins=[]
 for item in items:
  cp=torch.load(item['path'],map_location='cpu',weights_only=True)
  assert cp['epoch']==200 and all(torch.isfinite(v).all() for v in cp['model'].values())
  model=Paper2019LSTMClassifier().to('mps');model.load_state_dict(cp['model'],strict=True);model.eval()
  pred=np.full(24000,-1,dtype=np.int64);seen=np.zeros(24000,dtype=np.int8)
  with torch.inference_mode():
   for x,y,lengths,ids in batches(raw,256):
    assert (lengths==x.shape[1]).all()
    logits=model(x.squeeze(-1).to('mps'));assert torch.isfinite(logits).all()
    pred[ids]=predict(logits, softmax=True).cpu().numpy();seen[ids]+=1
  assert (seen==1).all() and (pred>=0).all()
  row=dict(training_T=item['T'],w2=item['w2'],seed=7,**metrics(raw['y'],pred));overall.append(row)
  for t in range(50,501):
   mask=raw['lengths']==t
   exact.append(dict(training_T=item['T'],test_T=t,**metrics(raw['y'][mask],pred[mask])))
  for lo in range(50,500,50):
   hi=lo+49 if lo<450 else 500;mask=(raw['lengths']>=lo)&(raw['lengths']<=hi)
   bins.append(dict(training_T=item['T'],test_T=f'{lo}-{hi}',**metrics(raw['y'][mask],pred[mask])))
  np.save(OUT/f'T{item["T"]}_predictions.npy',pred)
  print(json.dumps(row),flush=True)
  del model,cp;torch.mps.empty_cache()
 assert fingerprint(raw)==FP and sha(OFFICIAL)==old_manifest_hash
 assert all(sha(i['path'])==i['sha256'] for i in items)
 save(OUT/'overall.json',overall);save(OUT/'by_exact_test_T.json',exact);save(OUT/'by_length_bin.json',bins)
 save(OUT/'completion.json',dict(status='complete',model_count=3,test_fingerprint_unchanged=True,checkpoint_hashes_unchanged=True))
 print('BINS '+json.dumps(bins),flush=True)
if __name__=='__main__':main()
