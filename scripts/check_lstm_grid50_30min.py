"""Bounded hourly-check wrapper: immediate run counts as execution one."""
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'results/lstm_grid50_30min_schedule.json'

def main():
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(STATE.read_text())
        if state['executions']>=state['limit']:
            print(json.dumps(dict(schedule_finished=True,note='12 executions exhausted; no action')))
            return
        state['executions']+=1
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(STATE)
        result=subprocess.run([sys.executable,'-m','scripts.check_lstm_grid50_hourly'],
                              cwd=ROOT,capture_output=True,text=True)
        if result.returncode:
            data=dict(status='failed',note='Status command failed; no retry',
                      error=result.stderr[-400:])
        else:
            data=json.loads(result.stdout)
        data.update(execution=state['executions'],schedule_finished=state['executions']>=state['limit'])
        print(json.dumps(data))

if __name__=='__main__':main()
