"""Authorized gate continuation, with SIGUSR1/STOP_AFTER_EPOCH clean pause."""
import fcntl
import os
import signal
from pathlib import Path
import torch
from scripts.run_lstm_variable_length import read, save, sha, require, run, request_clean_stop, stamp

ROOT=Path('results/lstm_multilength_grid50_equalN_safeadam_20261020')
TRANSITION=Path('analysis_outputs/grid50_resume_20261023/source_transition.json')


def main():
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0','MPS fallback must be disabled')
    require(torch.backends.mps.is_available(),'MPS required')
    require(not (ROOT/'STOP_AFTER_EPOCH').exists(),'Remove acknowledged stop request before an explicitly authorized resume')
    with (ROOT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read(ROOT/'run_manifest.json');c=read(ROOT/'protocol.json')
        require(m['status']!='failed','Failed run requires diagnosis, never automatic restart')
        transition=read(TRANSITION)
        require(all(sha(p)==h for p,h in transition['after'].items()),'Approved source changed')
        if m['source_hashes']==transition['before']:
            require(m['status']=='awaiting_user_approval','Unexpected migration state')
            save(ROOT/('manifest_before_continuation_'+stamp()+'.json'),m)
            m.update(source_hashes=transition['after'],
                     continuation_authorization='User authorized epoch-1 resume and fixed-200 calibration on 2026-10-23',
                     source_transition=str(TRANSITION))
            save(ROOT/'run_manifest.json',m)
        require(m['source_hashes']==transition['after'],'Unknown source provenance')
        require(c['optimizer_implementation']=='grid50_scaled_second_moment_v1' and c['epochs']==200,
                'Unexpected training protocol')
        signal.signal(signal.SIGUSR1,request_clean_stop)
        signal.signal(signal.SIGTERM,request_clean_stop)
        signal.signal(signal.SIGINT,request_clean_stop)
        print(f'PID={os.getpid()} safe Adam MPS production; SIGUSR1 requests stop after current epoch.',flush=True)
        run(c,ROOT,production_continuation=True)


if __name__=='__main__':
    main()
