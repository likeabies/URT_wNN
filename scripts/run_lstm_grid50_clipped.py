"""Fresh grid50 clipped experiment; frozen training-only threshold."""
import argparse
import fcntl
import os
import signal
from pathlib import Path
import torch
from scripts.run_lstm_variable_length import read,require,sha,save,run,request_clean_stop

CONFIG=Path('configs/lstm_grid50_safeadam_clipped_20261026.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    c=read(CONFIG);clip=c['gradient_clipping']
    evidence=Path(clip['threshold_file']);threshold=read(evidence)
    validation=read(evidence.with_name('failure_validation.json'))
    require(validation['status']=='passed','Failure-batch clipping validation required')
    require(sha(evidence)==clip['threshold_sha256'],'Frozen threshold evidence changed')
    require(clip['max_norm']==threshold['max_norm']==2*threshold['overall']['max'],'Threshold rule changed')
    require(threshold['batches']==2350 and threshold['all_finite'],'Incomplete training-only norm replay')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0','MPS fallback prohibited')
    require(torch.backends.mps.is_available(),'MPS required')
    root=Path(c['output_dir'])
    require(not (root/'STOP_AFTER_EPOCH').exists(),'Pending clean-stop request')
    root.mkdir(parents=True,exist_ok=args.resume)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.resume:
            save(root/'clipping_provenance.json',dict(threshold=threshold,failure_validation=validation,
                  fresh_model=True,old_optimizer_reused=False))
        for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:
            signal.signal(sig,request_clean_stop)
        print(f'PID={os.getpid()} fresh clipped safe Adam; fixed max_norm={clip["max_norm"]}',flush=True)
        run(c,root,production_continuation=True)

if __name__=='__main__':main()
