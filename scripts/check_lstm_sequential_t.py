"""One locked status/launch pass for six sequential-T candidates, without analysis."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT/'results/lstm_sequential_t_grid25_20261027'
STATE=BASE/'queue.json'
CANDIDATES=['ASC-020','DESC-020','ASC-030','DESC-030','ASC-040','DESC-040']
MODULE='scripts.run_lstm_sequential_t'
NUMERIC={'nonfinite_loss','nonfinite_gradient_after_backward','nonfinite_gradient_after_clipping',
         'nonfinite_parameter_after_step','nonfinite_adam_state_after_step'}


def read(p): return json.loads(p.read_text()) if p.exists() else {}

def save(q):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(STATE)

def processes():
    found={}
    for line in subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True).splitlines():
        cols=line.strip().split(None,2)
        if len(cols)==3 and 'python' in Path(cols[2].split()[0]).name.lower() and '-m '+MODULE in cols[2]:
            found[int(cols[0])]=cols[2]
    return found


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=read(STATE) or dict(attempts=[])
        alive=processes()
        current=q['attempts'][-1] if q['attempts'] else None
        result=dict(current_candidate=current['candidate'] if current else None,new_process_started='no')
        if alive:
            result.update(status='running',pid=next(iter(alive)))
            print(json.dumps(result));return
        if current:
            root=BASE/current['candidate'];m=read(root/'run_manifest.json');f=read(root/'numeric_failure.json')
            completed=m.get('status')=='complete' and m.get('completed_epoch')==200
            failed=m.get('status')=='failed' and f.get('stage') in NUMERIC
            result['status']='completed' if completed else 'failed'
            if not (completed or failed):
                result['note']='No confirmed completion/numerical failure; no launch'
                print(json.dumps(result));return
        index=len(q['attempts'])
        if index==len(CANDIDATES):
            result['terminal']=True;print(json.dumps(result));return
        candidate=CANDIDATES[index]
        assert candidate not in [a['candidate'] for a in q['attempts']]
        log=PROJECT/f'logs/lstm_sequential_t_grid25_20261027_{candidate}.log'
        if (BASE/candidate).exists() or log.exists():
            result.update(status='failed',note='Existing output/log; duplicate launch refused')
            print(json.dumps(result));return
        entry=dict(candidate=candidate,log=str(log),state='launch_reserved')
        q['attempts'].append(entry);save(q)
        with log.open('xb') as f:
            child=subprocess.Popen(['/usr/bin/nohup','/usr/bin/caffeinate','-is',
                str(PROJECT/'.venv/bin/python'),'-u','-m',MODULE,'--candidate',candidate],
                cwd=PROJECT,env=dict(os.environ,WANDB_MODE='online',PYTORCH_ENABLE_MPS_FALLBACK='0'),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        entry.update(pid=child.pid,state='launched');save(q)
        time.sleep(2)
        started=child.poll() is None and bool(processes())
        result.update(new_candidate=candidate,new_process_started='yes' if started else 'no',pid=child.pid)
        if current is None:result.update(current_candidate=candidate,status='running' if started else 'failed')
        print(json.dumps(result))

if __name__=='__main__':main()
