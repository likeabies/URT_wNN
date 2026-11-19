"""Standalone second-moment arithmetic only; no training or optimizer changes."""
import json, math, time, runpy
from pathlib import Path
import torch

start=time.perf_counter()
runpy.run_path('results/lstm_mps_correctness_review_20261019/adam_reproducer.py')
beta2=.999
rows=[]
for name,grad_values,state_values in [
    ('original_large',[7.243555335899919e19],[1e12]),
    ('normal',[0.,1e-6,-.01,1.,-10.],[0.,1e-8,.1,2.,100.])
]:
    outputs={}
    for device in ['cpu','mps']:
        g=torch.tensor(grad_values,dtype=torch.float32,device=device)
        v=torch.tensor(state_values,dtype=torch.float32,device=device)
        original=v.clone().mul_(beta2).addcmul_(g,g,value=1-beta2)
        scaled=g*math.sqrt(1-beta2)
        workaround=v.clone().mul_(beta2).addcmul_(scaled,scaled)
        outputs[device]=(original.cpu(),workaround.cpu())
    ref=outputs['cpu'][0].double()
    actual=outputs['mps'][1].double()
    abs_error=(actual-ref).abs()
    rel_error=abs_error/torch.clamp(ref.abs(),min=1e-300)
    row=dict(case=name,gradient=grad_values,initial_state=state_values,
        results={d:{method:dict(values=[str(x) for x in value.tolist()],finite=bool(torch.isfinite(value).all()))
            for method,value in zip(['original','scaled_grad'],values)} for d,values in outputs.items()},
        cpu_original_vs_mps_scaled_max_abs_error=abs_error.max().item(),
        cpu_original_vs_mps_scaled_max_relative_error=rel_error.max().item(),
        cpu_original_vs_cpu_scaled_max_abs_error=(outputs['cpu'][0].double()-outputs['cpu'][1].double()).abs().max().item(),
        numerically_matches=bool(torch.allclose(actual,ref,rtol=1e-6,atol=1e-12)))
    rows.append(row)
report=dict(beta2=beta2,dtype='float32',torch_version=torch.__version__,cases=rows,
            elapsed_seconds=time.perf_counter()-start)
report['outcome']='A' if (not rows[0]['results']['mps']['original']['finite']
    and all(r['results']['mps']['scaled_grad']['finite'] and r['numerically_matches'] for r in rows)) else 'B'
out=Path(__file__).with_name('report.json')
with out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
