"""Existing ADF benchmark on immutable saved independent-test cohorts only."""
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import statsmodels
from src.adf_utils import adf_binary_predict
from src.variable_length_lstm import load_data, fingerprint
from scripts.run_ann_paper2019 import fingerprint as fixed_fingerprint

SOURCE=Path('results/shared_lstm_independent_evaluation_20261029')
OUT=Path('results/adf_saved_independent_tests_20261029')
EXPECTED={50:'d1513a67d48f6f689ceabe1f84c383b7e7518a235529f07c9cc5d119f36230e0',100:'04aaa1699cc91799f9c3ae32421f58a5d679584d24d1d5f3e62d4071046a64aa',250:'e95d95a0efc66894dcde26bd9d9cda1ca2cff288389605f189cf8ac78b1cd7ce'}
VARIABLE='fbbbb1d22afb89cc6cbdc6a3f6b33ce9172ceb6b85a82d49cc9e4ad483040953'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def predict(x):return adf_binary_predict(x,alpha=.05,regression='n')

def main():
 raw=load_data(SOURCE/'variable_test');assert fingerprint(raw)==VARIABLE
 cohorts=[('Variable T50-500',raw,VARIABLE)]
 for t,fp in EXPECTED.items():
  with np.load(SOURCE/f'fixed_test_arrays/T{t}.npz') as z:d={k:z[k] for k in ['x','y','rho','beta']}
  assert fixed_fingerprint(d)==fp
  cohorts.append((f'Fixed T{t}',d,fp))
 for name,d,fp in cohorts:
  counts=(int((d['y']==0).sum()),int((d['y']==1).sum()))
  assert counts==((12000,12000) if 'Variable' in name else (4000,12000))
 inputs=[p for p in SOURCE.rglob('*') if p.is_file()]
 before={str(p):sha(p) for p in inputs}
 OUT.mkdir(exist_ok=False)
 save(OUT/'protocol.json',dict(regression='n',deterministic_terms='none',autolag='AIC',maxlag=None,
  maxlag_default='min(ceil(12*(T/100)**0.25), T//2-1)',alpha=.05,rejection='statistic < critical_values[5%]',
  implementation='src.adf_utils.adf_binary_predict',implementation_sha256=sha(Path('src/adf_utils.py')),
  statsmodels=statsmodels.__version__,test_seed=20261008,new_data_generated=False,input_hashes=before,
  fingerprints={name:fp for name,_,fp in cohorts}))
 print('All four fingerprints and class counts verified.',flush=True)
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for name,d,fp in cohorts:
   if 'offsets' in d:
    series=(np.asarray(d['x'][d['offsets'][i]:d['offsets'][i+1]]) for i in range(len(d['y'])))
   else:series=iter(d['x'])
   pred=np.empty(len(d['y']),dtype=np.int8)
   for i,v in enumerate(pool.map(predict,series,chunksize=128)):
    pred[i]=v
    if (i+1)%8000==0:print(f'{name}: {i+1}/{len(pred)}',flush=True)
   np.save(OUT/(name.replace(' ','_')+'_rejections.npy'),pred)
   h0=d['y']==0;h1=d['y']==1
   row=dict(test_set=name,H0_N=int(h0.sum()),H1_N=int(h1.sum()),size=float(pred[h0].mean()),power=float(pred[h1].mean()))
   rows.append(row);print(json.dumps(row),flush=True)
   save(OUT/(name.replace(' ','_')+'_metrics.json'),row)
 assert all(sha(Path(p))==v for p,v in before.items()),'Existing input/result changed'
 save(OUT/'results.json',dict(status='complete',input_hashes_unchanged=True,results=rows))
if __name__=='__main__':main()
