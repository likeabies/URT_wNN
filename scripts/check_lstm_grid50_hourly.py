"""One bounded status/conditional-launch pass. No analysis, retry, or scheduler."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
STATE=PROJECT/'results/lstm_grid50_hourly_queue.json'
CONDITIONS=[(.2,256),(.3,256),(.2,128),(.4,256),(.3,128),(.2,512),(.4,128),(.3,512)]
MODULE='scripts.run_lstm_grid50_gradient_observation'
NUMERIC={'nonfinite_loss','nonfinite_gradient_after_backward','nonfinite_gradient_after_clipping',
         'nonfinite_parameter_after_step','nonfinite_adam_state_after_step'}

def read(path):
    return json.loads(path.read_text())

def save(value):
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(STATE)

def processes():
    found={}
    for line in subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True).splitlines():
        cols=line.strip().split(None,2)
        if len(cols)==3 and 'python' in Path(cols[2].split()[0]).name.lower() and '-m '+MODULE in cols[2]:
            found[int(cols[0])]=cols[2]
    return found

def latest_epoch(root,manifest):
    csv=root/'batch_gradients.csv'
    if csv.exists() and csv.stat().st_size:
        with csv.open('rb') as f:
            f.seek(max(0,csv.stat().st_size-2048));lines=f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            fields=line.split(',')
            if len(fields)==9 and fields[0].isdigit():return int(fields[0])
    return manifest.get('completed_epoch',0)

def main():
    os.chdir(PROJECT)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=read(STATE) if STATE.exists() else dict(attempts=[dict(condition=1,
            output='results/lstm_grid50_gradient_observation_20261026')])
        current=q['attempts'][-1];n=current['condition'];w,b=CONDITIONS[n-1]
        root=PROJECT/current['output']
        manifest=read(root/'run_manifest.json') if (root/'run_manifest.json').exists() else {}
        alive=processes()
        result=dict(status='running' if alive else 'failed',w2=w,batch_size=b,
                    current_epoch=latest_epoch(root,manifest),new_process_started='no')
        if alive:
            result['pid']=next(iter(alive))
            print(json.dumps(result));return
        if manifest.get('status')=='complete' and manifest.get('completed_epoch')==200:
            result.update(status='completed',terminal=True)
            print(json.dumps(result));return
        f=root/'numeric_failure.json'
        failure=read(f) if f.exists() else {}
        if manifest.get('status')!='failed' or failure.get('stage') not in NUMERIC:
            result.update(note='Stopped without confirmed numerical failure; no launch',terminal=False)
            print(json.dumps(result));return
        result['failure']={k:failure.get(k) for k in ['epoch','batch','T']}
        if n==8:
            result.update(terminal=True,note='All eight conditions exhausted')
            print(json.dumps(result));return
        next_n=n+1
        assert next_n not in [a['condition'] for a in q['attempts']]
        nw,nb=CONDITIONS[next_n-1]
        output=f'results/lstm_grid50_hourly_c{next_n}_w2_{nw:.2f}_bs{nb}'
        if (PROJECT/output).exists():
            result['note']='Next output already exists; refusing duplicate launch'
            print(json.dumps(result));return
        log=PROJECT/f'logs/lstm_grid50_hourly_c{next_n}.log'
        if log.exists():
            result['note']='Next log already exists; refusing duplicate launch'
            print(json.dumps(result));return
        # Reserve the condition before spawning: a crash must never repeat it.
        entry=dict(condition=next_n,output=output,log=str(log),state='launch_reserved')
        q['attempts'].append(entry);save(q)
        with log.open('xb') as f:
            child=subprocess.Popen(['/usr/bin/nohup','/usr/bin/caffeinate','-is',
                str(PROJECT/'.venv/bin/python'),'-u','-m',MODULE,'--condition',str(next_n)],
                cwd=PROJECT,env=dict(os.environ,WANDB_MODE='online',PYTORCH_ENABLE_MPS_FALLBACK='0'),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        entry.update(pid=child.pid,state='launched');save(q)
        time.sleep(1)
        started=child.poll() is None and child.pid in processes()
        result.update(new_w2=nw,new_batch_size=nb,new_process_started='yes' if started else 'no',pid=child.pid)
        print(json.dumps(result))

if __name__=='__main__':main()
