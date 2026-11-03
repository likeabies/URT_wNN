"""Pooled held-out w2 calibration, then one independent variable-length test."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb

from scripts.run_ann_paper2019 import save
from scripts.run_ann_fixed200_calibration import local_grid, write_csv
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy, predict
from src.utils import set_seed, select_device
from src.variable_length_lstm import VariableLengthLSTM, batches, fingerprint, generate, load_data

STOP_REQUESTED=False


def request_clean_stop(signum=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def should_stop(root):
    return STOP_REQUESTED or (Path(root)/'STOP_AFTER_EPOCH').exists()


CONFIG=Path('configs/lstm_variable_length_randomT_exact_batch.json')
SOURCES=[__file__,'src/variable_length_lstm.py','src/data.py','src/model.py','src/lstm_binary.py','src/train.py','src/utils.py',
         'scripts/run_ann_paper2019.py','scripts/run_ann_fixed200_calibration.py','scripts/run_lstm_multilength_grid50.py']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sync(device):
    if device=='mps':torch.mps.synchronize()
    elif device.startswith('cuda'):torch.cuda.synchronize()


def validate_epoch(row,seen,expected_n,expected_epoch,parameters_finite,gradients_finite):
    diagnostics=dict(expected_n=expected_n,processed_n=int(seen.sum()),unique_n=int((seen>0).sum()),
        missing_n=int((seen==0).sum()),duplicate_n=int((seen>1).sum()),
        batch_count=row.get('batch_count'),epoch=row['epoch'],expected_epoch=expected_epoch,
        loss=str(row['train_loss']),accuracy=str(row['train_acc']),
        loss_finite=math.isfinite(row['train_loss']),accuracy_finite=math.isfinite(row['train_acc']),
        accuracy_in_range=0<=row['train_acc']<=1,seconds_finite=math.isfinite(row['train_seconds']),
        parameters_finite=parameters_finite,gradients_finite=gradients_finite)
    checks=dict(sample_count=diagnostics['processed_n']==expected_n,
        unique_coverage=len(seen)==expected_n and bool((seen==1).all()),
        epoch=row['epoch']==expected_epoch,batches=bool(row.get('batch_count',0)>0),
        **{k:diagnostics[k] for k in ['loss_finite','accuracy_finite','accuracy_in_range',
                                      'seconds_finite','parameters_finite','gradients_finite']})
    failed=[k for k,v in checks.items() if not v]
    require(not failed,'Invalid training epoch: '+json.dumps(dict(diagnostics,failed_predicates=failed)))
    return diagnostics


def atomic_checkpoint(path,model,optimizer,epoch,history,c,device):
    """Publish immutable epochs, then a checksum receipt; partial files are ignored."""
    path=Path(path);require(not path.exists(),'Refusing to overwrite checkpoint '+str(path))
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,
        torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        numpy_rng=dict(bit_generator=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),
            pos=np.random.get_state()[2],has_gauss=np.random.get_state()[3],cached_gaussian=np.random.get_state()[4]),
        shuffle=dict(algorithm='numpy.default_rng(SeedSequence([model_seed, epoch, 917]))',
            model_seed=c['model_seed'],next_epoch=epoch+1,
            next_generator_state=np.random.default_rng(np.random.SeedSequence([c['model_seed'],epoch+1,917])).bit_generator.state),
        mps_rng=torch.mps.get_rng_state() if device=='mps' else None,
        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
    tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    receipt=dict(checkpoint=str(path),sha256=sha(path),epoch=epoch)
    save(path.with_suffix('.json'),receipt)
    return receipt


def restore_epoch(receipt,model,optimizer,c,device):
    require(sha(receipt['checkpoint'])==receipt['sha256'],'Resume checkpoint checksum mismatch')
    cp=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True)
    epoch=cp['epoch'];require(epoch==receipt['epoch'] and 0<epoch<=c['epochs'],'Resume epoch mismatch')
    require([r['epoch'] for r in cp['history']]==list(range(1,epoch+1)),'Resume history incomplete')
    require(all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Nonfinite checkpoint model')
    require(cp['shuffle']['model_seed']==c['model_seed'] and cp['shuffle']['next_epoch']==epoch+1,'Shuffle state mismatch')
    for r in cp['history']:
        require(r['processed_n']==r['expected_n']==r['unique_n'] and r['missing_n']==r['duplicate_n']==0,
                'Checkpoint coverage invalid')
        require(math.isfinite(r['train_loss']) and math.isfinite(r['train_acc']) and 0<=r['train_acc']<=1,
                'Checkpoint metrics invalid')
    for value in cp['optimizer']['state'].values():
        require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in value.values()),'Nonfinite optimizer')
    model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
    torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng'])
    n=cp['numpy_rng'];np.random.set_state((n['bit_generator'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    if device=='mps':torch.mps.set_rng_state(cp['mps_rng'])
    if device.startswith('cuda'):torch.cuda.set_rng_state_all(cp['cuda_rng'])
    return cp['history']


def wb_call(path,operation,*args,**kwargs):
    """Telemetry failures never invalidate an already saved scientific state."""
    try:return operation(*args,**kwargs)
    except Exception as exc:
        message=dict(time=stamp(),operation=getattr(operation,'__name__',str(operation)),error=repr(exc))
        print('W&B warning: '+json.dumps(message),flush=True)
        with (Path(path)/'wandb_errors.jsonl').open('a') as f:f.write(json.dumps(message)+'\n')
        return None


def evaluate(model,raw,c,w2,device,calibration=False):
    model.eval();predicted=np.empty(len(raw['y']),dtype=np.int64);loss_sum=0.
    with torch.no_grad():
        for x,y,lengths,indices in batches(raw,c['batch_size']):
            logits=model(x.to(device))
            loss=ann_weighted_cross_entropy(logits,y.to(device),c['w1'],w2)
            loss_sum+=loss.item()*len(y)
            predicted[indices]=predict(logits, softmax=True).cpu().numpy()
    ur=raw['y']==1;stationary=~ur
    result=dict(n=len(predicted),loss=loss_sum/len(predicted),accuracy=float((predicted==raw['y']).mean()),
                pooled_size=float((predicted[ur]==0).mean()))
    require(all(math.isfinite(v) for v in result.values()),'Nonfinite evaluation')
    if calibration:
        require(ur.all(),'Held-out calibration must contain H0 only')
    else:
        result['pooled_power']=float((predicted[stationary]==0).mean())
        result['power_by_rho']={str(r):float((predicted[raw['rho']==r]==0).mean()) for r in c['rhos'] if r<1}
    result['by_T']={}
    for length in np.unique(raw['lengths']):
        mask=raw['lengths']==length;h0=mask&ur;h1=mask&stationary
        result['by_T'][str(length)]=dict(n=int(mask.sum()),n_unit_root=int(h0.sum()),n_stationary=int(h1.sum()),
            empirical_size=float((predicted[h0]==0).mean()) if h0.any() else None,
            empirical_power=float((predicted[h1]==0).mean()) if h1.any() else None)
    return result


def convergence(history):
    lookup={r['epoch']:r for r in history}
    result={'snapshots':[lookup[e] for e in [100,150,175,200] if e in lookup], 'intervals':[]}
    for a,b in [(100,150),(150,200),(175,200)]:
        if a in lookup and b in lookup:
            decrease=lookup[a]['train_loss']-lookup[b]['train_loss']
            result['intervals'].append(dict(start=a,end=b,loss_decrease=decrease,
                relative_decrease=decrease/lookup[a]['train_loss'],accuracy_change=lookup[b]['train_acc']-lookup[a]['train_acc']))
    if len(history)>=50:
        result['last50_loss_slope']=float(np.polyfit([r['epoch'] for r in history[-50:]],[r['train_loss'] for r in history[-50:]],1)[0])
    return result


def search_action(records,c):
    """Only pooled H0 size is visible to selection. Never receives test/T metrics."""
    done={round(r['w2'],10):r for r in records}
    for w in c['coarse_w2']:
        if w not in done:return 'coarse',[w],None
    coarse=[r for r in records if r['stage'] in ['coarse','extension']]
    points=[dict(w2=r['w2'],mean_type_i_error=r['calibration']['pooled_size']) for r in coarse]
    exact=[r for r in points if r['mean_type_i_error']==c['target_size']]
    if exact:return 'select',[],dict(reason='An observed coarse pooled size equals target exactly',bracketed=True)
    plan=local_grid(points,c['target_size'])
    if not plan['bracketed']:
        below=all(r['mean_type_i_error']<c['target_size'] for r in points)
        w=round(max(done)+c['extension_step'],10) if below else round(min(done)-c['extension_step'],10)
        if w<0:return 'select',[],dict(reason='No crossing down to w2=0; nonnegative WCE boundary reached',bracketed=False)
        return 'extension',[w],dict(reason='All observed pooled sizes below target' if below else 'All observed pooled sizes above target')
    lo,hi=plan['interval'];count=round((hi-lo)/c['fine_step'])
    grid=[round(lo+i*c['fine_step'],10) for i in range(count+1)]
    require(abs(grid[-1]-hi)<1e-9,'Fine interval not divisible by .05')
    plan['w2_values']=grid
    plan['crossings']=[[a['w2'],b['w2']] for a,b in zip(sorted(points,key=lambda r:r['w2']),sorted(points,key=lambda r:r['w2'])[1:])
                       if (a['mean_type_i_error']-c['target_size'])*(b['mean_type_i_error']-c['target_size'])<=0]
    plan['selection_rule']='Historical local_grid: nearest coarse point, crossing preference, endpoint error tie-break; select final w2 by pooled absolute error over ALL completed candidates, lower w2 on ties.'
    pending=[w for w in grid if w not in done]
    return ('fine' if pending else 'select'),pending,plan


def get_dataset(root,c,state,split):
    # Called for test only after a saved, locked selection exists.
    if split=='test':require((root/'selection.json').is_file(),'Test forbidden before final selection')
    if split in state['datasets']:
        meta=state['datasets'][split];raw=load_data(meta['path'])
        require(fingerprint(raw)==meta['fingerprint'],'Cached dataset checksum mismatch')
        expected_seed=c[{'train':'data_seed','calibration':'calibration_seed','test':'test_seed'}[split]]
        require(meta['seed']==expected_seed,'Cached dataset seed mismatch')
        return raw
    folder=root/'datasets';folder.mkdir(exist_ok=True)
    candidates=sorted(folder.glob(split+'_*/dataset.json'))
    for p in candidates:
        meta=read(p);raw=load_data(p.parent)
        require(fingerprint(raw)==meta['fingerprint'],'Cached dataset checksum mismatch')
        state['datasets'][split]=dict(meta,path=str(p.parent));save(root/'run_manifest.json',state)
        return raw
    raw,meta=generate(folder/(split+'_'+stamp()),c,split)
    state['datasets'][split]=dict(meta,path=str(Path(raw['x'].filename).parent))
    save(root/'run_manifest.json',state)
    print(f'{split} dataset: {meta}',flush=True)
    return raw


def train_candidate(root,c,state,raw,cal,w2,stage,device):
    attempt=stamp();path=root/'runs'/f'w2_{w2:.10g}_{attempt}';path.mkdir(parents=True)
    cfg=dict(c,w2=w2,stage=stage,device=device,training_fingerprint=state['datasets']['train']['fingerprint'],
             calibration_fingerprint=state['datasets']['calibration']['fingerprint'],
             architecture='existing LSTM(1,30,1) + Linear(30,2); h_n[-1]; exact-length batches',test_accessed=False)
    save(path/'config_snapshot.json',cfg)
    set_seed(c['model_seed']);model=VariableLengthLSTM().to(device)
    safe = c.get('optimizer_implementation') == 'grid50_scaled_second_moment_v1'
    if safe:
        from src.grid50_safe_adam import Grid50SafeAdam
        from src.grid50_step_audit import StepAudit
        optimizer=Grid50SafeAdam(model.parameters(),lr=c['lr'])
        audit=StepAudit(path,model,optimizer,device)
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=0.)
        audit=None
    clipping=c.get('gradient_clipping')
    history=[]
    receipt=None
    # Every restart is a new attempt; previous histories/checkpoints remain immutable.
    candidates=[]
    for config_path in (root/'runs').glob('*/config_snapshot.json'):
        if config_path.parent==path:continue
        old=read(config_path)
        if old==cfg:
            for receipt_path in config_path.parent.glob('epoch_[0-9][0-9][0-9].json'):
                receipt=read(receipt_path);candidates.append(receipt)
    if candidates:
        invalid=[]
        for candidate in sorted(candidates,key=lambda r:(r['epoch'],r['checkpoint']),reverse=True):
            try:
                history=restore_epoch(candidate,model,optimizer,c,device)
                receipt=candidate;break
            except (RuntimeError,ValueError,KeyError,OSError) as exc:
                invalid.append(dict(checkpoint=candidate['checkpoint'],error=repr(exc)))
        if invalid:save(path/'invalid_checkpoints.json',invalid)
        require(history and receipt is not None,'No valid checkpoint remains; refusing an implicit restart')
        save(path/'resumed_from.json',receipt)
        save(path/'history.json',history)
        print(f'Restored epoch {len(history)} from {receipt["checkpoint"]}',flush=True)
    wr=wb_call(path,wandb.init,project='URT_wNN',group=c['experiment'],job_type='calibration-training',
        name=f"{c['experiment']}_{stage}_w2_{w2:.10g}_ms7_{attempt}",config=cfg,mode='online')
    identity=dict(wandb_id=wr.id if wr else None,wandb_url=wr.url if wr else None)
    save(path/'wandb.json',identity)
    try:
        for epoch in range(len(history)+1,c['epochs']+1):
            model.train();total=0;total_loss=0.;correct=0;sync(device);start=perf_counter()
            seen=np.zeros(len(raw['y']),dtype=np.int32)
            epoch_gradient_max=0.;epoch_second_max=0.
            clip_counts={};epoch_preclip_max=0.;epoch_clipped=0
            for batch_count,(x,y,lengths,indices) in enumerate(batches(raw,c['batch_size'],model_seed=c['model_seed'],epoch=epoch),1):
                np.add.at(seen,indices,1)
                require(x.ndim==3 and x.shape[2]==1 and bool((lengths==x.shape[1]).all()),'Non-exact-length batch')
                if audit:
                    audit.begin(dict(epoch=epoch,batch=batch_count,T=x.shape[1],w2=w2,
                                     seed=c['model_seed']),indices)
                x=x.to(device);y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x);loss=ann_weighted_cross_entropy(logits,y,c['w1'],w2)
                loss.backward()
                if audit:audit.before_step(loss)
                gradients_finite=torch.isfinite(torch.cat([p.grad.flatten() for p in model.parameters()])).all().item()
                if not gradients_finite or not math.isfinite(loss.item()):
                    failure=dict(epoch=epoch,batch=batch_count,expected_n=len(seen),processed_n=int(seen.sum()),
                        unique_n=int((seen>0).sum()),loss=str(loss.item()),gradients_finite=gradients_finite,
                        parameters_finite=all(torch.isfinite(p).all().item() for p in model.parameters()))
                    save(path/'numeric_failure.json',failure)
                    raise RuntimeError('Invalid training batch: '+json.dumps(failure))
                if clipping:
                    from src.grid50_gradient_clipping import clip_global_norm64
                    clip_info=clip_global_norm64(model.parameters(),clipping['max_norm'])
                    key=str(x.shape[1])
                    count=clip_counts.setdefault(key,dict(batches=0,clipped=0))
                    count['batches']+=1;count['clipped']+=int(clip_info['clipped'])
                    epoch_clipped+=int(clip_info['clipped'])
                    epoch_preclip_max=max(epoch_preclip_max,clip_info['pre_clip_global_norm'])
                    if audit:audit.last.update(clip_info)
                optimizer.step()
                if audit:
                    audit.after_step()
                    epoch_gradient_max=max(epoch_gradient_max,audit.last['gradient_max'])
                    epoch_second_max=max(epoch_second_max,audit.last['exp_avg_sq_max'])
                bad_params=[name for name,p in model.named_parameters() if not torch.isfinite(p).all().item()]
                bad_state=[f'{name}.{key}' for name,p in model.named_parameters()
                           for key,value in optimizer.state[p].items()
                           if torch.is_tensor(value) and not torch.isfinite(value).all().item()]
                if bad_params or bad_state:
                    failure=dict(epoch=epoch,batch=batch_count,nonfinite_parameters=bad_params,
                                 nonfinite_optimizer_state=bad_state,processed_n=int(seen.sum()))
                    save(path/'numeric_failure.json',failure)
                    raise RuntimeError('Invalid optimizer update: '+json.dumps(failure))
                total_loss+=loss.item()*len(y);correct+=(predict(logits)==y).sum().item();total+=len(y)
            sync(device)
            row=dict(epoch=epoch,train_loss=total_loss/total,train_acc=correct/total,train_seconds=perf_counter()-start,
                     batch_count=batch_count)
            diagnostics=validate_epoch(row,seen,len(raw['y']),epoch,
                all(torch.isfinite(p).all().item() for p in model.parameters()),True)
            for value in optimizer.state.values():
                require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in value.values()),'Nonfinite optimizer state')
            row.update({k:diagnostics[k] for k in ['expected_n','processed_n','unique_n','missing_n','duplicate_n']})
            if audit:
                row.update(epoch_max_gradient=epoch_gradient_max,epoch_max_exp_avg_sq=epoch_second_max,
                           final_batch_gradient_max=audit.last['gradient_max'],
                           final_exp_avg_sq_max=audit.last['exp_avg_sq_max'])
            if clipping:
                row.update(clip_fraction=epoch_clipped/batch_count,
                           clip_fraction_by_T={k:v['clipped']/v['batches'] for k,v in clip_counts.items()},
                           clipping_counts_by_T=clip_counts,maximum_pre_clip_norm=epoch_preclip_max)
            history.append(row)
            receipt=atomic_checkpoint(path/f'epoch_{epoch:03d}.pt',model,optimizer,epoch,history,c,device)
            save(path/'history.json',history)
            state['progress']=dict(w2=w2,stage=stage,epoch=epoch,path=str(path),checkpoint=receipt)
            if w2==.2 and epoch==35:
                state['verification_35']=dict(status='passed',w2=w2,epoch=epoch,coverage_verified=True,checkpoint=receipt)
            save(root/'run_manifest.json',state)
            if wr:wb_call(path,wr.log,row)
            print(f'w2={w2:g} {row}',flush=True)
            if clipping and (row['clip_fraction']>clipping['stop_fraction'] or
                             any(v>clipping['stop_fraction'] for v in row['clip_fraction_by_T'].values())):
                save(path/'excessive_clipping.json',dict(epoch=epoch,metrics=row,threshold=clipping,
                     reason='Majority of batches clipped in full epoch or one T; threshold not changed'))
                raise RuntimeError('Excessive clipping; stopped without threshold adjustment')
            if state.get('production_continuation') and should_stop(root):
                state.update(status='paused_clean',pause_reason='User stop requested at completed epoch',
                             paused_at=stamp())
                save(root/'run_manifest.json',state)
                print('CLEAN STOP: completed epoch checkpoint saved.',flush=True)
                return dict(status='paused_clean',checkpoint=receipt)
            if c.get('pause_after_epoch') == epoch and not state.get('production_continuation'):
                state.update(status='awaiting_user_approval',gate=dict(w2=w2,epoch=epoch,
                             checkpoint=receipt,metrics=row,wandb=identity,
                             reason='First epoch gate; no further training/calibration/test authorized'))
                save(root/'run_manifest.json',state)
                save(path/'epoch_gate.json',state['gate'])
                return dict(status='awaiting_user_approval',checkpoint=receipt)
        checkpoint=Path(receipt['checkpoint'])
        metrics=evaluate(model,cal,c,w2,device,calibration=True)
        summary=convergence(history);save(path/'convergence.json',summary);save(path/'calibration.json',metrics)
        record=dict(status='complete',stage=stage,w2=w2,model_seed=c['model_seed'],epochs=c['epochs'],
                    path=str(path),history=str(path/'history.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                    calibration=metrics,convergence=summary,**identity,
                    training_fingerprint=cfg['training_fingerprint'],calibration_fingerprint=cfg['calibration_fingerprint'])
        if wr:wb_call(path,wr.log,{'calibration':metrics})
    except Exception as exc:
        if audit and audit.good is not None and not (path/'numeric_failure.json').exists():
            audit.fail(repr(exc))
        raise
    finally:
        if wr:wb_call(path,wr.finish)

    save(path/'completed.json',record)
    state['runs'].append(record);save(root/'run_manifest.json',state)
    return record


def restore_records(root,c,state):
    records=[]
    for p in sorted((root/'runs').glob('*/completed.json')):
        r=read(p);cfg=read(p.parent/'config_snapshot.json');hist=read(p.parent/'history.json')
        require(r['status']=='complete' and r['epochs']==c['epochs'],'Invalid completion record')
        require(all(cfg.get(k)==v for k,v in c.items()),'Completed run config differs')
        require(cfg['w2']==r['w2'] and r['model_seed']==c['model_seed'],'Completed run identity differs')
        require([h['epoch'] for h in hist]==list(range(1,c['epochs']+1)),'Incomplete epoch history')
        require(sha(r['checkpoint'])==r['checkpoint_sha256'],'Checkpoint changed')
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
        require(cp['epoch']==c['epochs'] and all(torch.isfinite(v).all().item() for v in cp['model'].values()),'Invalid checkpoint')
        VariableLengthLSTM().load_state_dict(cp['model'])
        require(read(p.parent/'calibration.json')==r['calibration'],'Calibration result mismatch')
        for split,field in [('train','training_fingerprint'),('calibration','calibration_fingerprint')]:
            require(r[field]==state['datasets'][split]['fingerprint'],'Completed data fingerprint differs')
        records.append(r)
    require(len({r['w2'] for r in records})==len(records),'Duplicate completed weight')
    state['runs']=records;save(root/'run_manifest.json',state)


def finish(root,c,state,test,device):
    selection=read(root/'selection.json')
    best=next(r for r in state['runs'] if r['w2']==selection['w2'])
    require(sha(best['checkpoint'])==best['checkpoint_sha256'],'Selected checkpoint changed')
    if not (root/'independent_test.json').exists():
        model=VariableLengthLSTM().to(device)
        model.load_state_dict(torch.load(best['checkpoint'],map_location='cpu',weights_only=True)['model'])
        result=evaluate(model,test,c,best['w2'],device)
        save(root/'independent_test.json',dict(selected_w2=best['w2'],test_fingerprint=state['datasets']['test']['fingerprint'],metrics=result))
    result=read(root/'independent_test.json')
    require(result['selected_w2']==best['w2'] and result['test_fingerprint']==state['datasets']['test']['fingerprint'],
            'Saved independent test provenance mismatch')
    if not (root/'final_wandb.json').exists():
        wr=wb_call(root,wandb.init,project='URT_wNN',group=c['experiment'],job_type='independent-test',
                        name=c['experiment']+'_final_'+stamp(),config=dict(c,selected_w2=best['w2'],device=device,
                        test_fingerprint=state['datasets']['test']['fingerprint']),mode='online')
        if wr:
            wb_call(root,wr.log,{'independent_test':result['metrics'],'calibration_size':best['calibration']['pooled_size']})
            receipt=dict(id=wr.id,url=wr.url)
            wb_call(root,wr.finish)
            save(root/'final_wandb.json',receipt)
    rows=[dict(stage=r['stage'],w2=r['w2'],pooled_calibration_size=r['calibration']['pooled_size'],
               absolute_error=abs(r['calibration']['pooled_size']-c['target_size']),checkpoint=r['checkpoint'],wandb_url=r['wandb_url']) for r in sorted(state['runs'],key=lambda r:r['w2'])]
    write_csv(root/'calibration_table.csv',rows)
    save(root/'convergence_summary.json',[dict(w2=r['w2'],**r['convergence']) for r in state['runs']])
    save(root/'final_summary.json',dict(selection=selection,independent_test=result,convergence=best['convergence'],
         interpretation='Pooled feasibility only; no length-specific calibration or model selection. Fixed model seed 7 is not a multi-seed stability assessment.'))
    state.update(status='complete',finished_at=stamp(),selected_w2=best['w2'],independent_test=result)
    save(root/'run_manifest.json',state)
    print('COMPLETE '+json.dumps(result),flush=True)


def run(c,root,smoke=False,recover_from=None,production_continuation=False):
    sources=SOURCES+(['src/grid50_safe_adam.py','src/grid50_step_audit.py',
                      'scripts/run_lstm_grid50_safeadam_gate.py'] if c.get('optimizer_implementation') else [])
    if production_continuation:sources+=['scripts/run_lstm_grid50_safeadam_production.py']
    if c.get('gradient_clipping'):
        sources+=['src/grid50_gradient_clipping.py','scripts/run_lstm_grid50_clipped.py']
    hashes={str(Path(s).relative_to(Path.cwd())) if Path(s).is_absolute() else s:sha(s) for s in sources}
    if (root/'protocol.json').exists():
        require(read(root/'protocol.json')==c,'Resume protocol mismatch')
        state=read(root/'run_manifest.json');require(state['source_hashes']==hashes,'Resume source changed')
        require(not c.get('optimizer_implementation') or state['status']!='failed' and
                (state['status']!='awaiting_user_approval' or production_continuation),
                'Safe Adam gate requires user review; automatic restart/resume prohibited')
        if state['status']=='complete':print('Already complete; no training or test repeated.');return
    else:
        save(root/'protocol.json',c)
        state=dict(status='initializing',source_hashes=hashes,device=c.get('device',select_device()),protocol=c,datasets={},runs=[],smoke=smoke)
        if recover_from:
            previous=read(Path(recover_from)/'run_manifest.json')
            require(all(previous['protocol'].get(k)==v for k,v in c.items() if k not in ['experiment','output_dir']),
                    'Recovery scientific protocol mismatch')
            require(not previous['runs'],'Recovery import is only for the failed zero-completion experiment')
            state['datasets']={k:v for k,v in previous['datasets'].items() if k in ['train','calibration']}
            state['recovery_source']=str(recover_from)
        save(root/'run_manifest.json',state)
    if production_continuation:state['production_continuation']=True
    device=c.get('device',select_device());require(device==state['device'],'Device changed on resume')
    require(device!='mps' or torch.backends.mps.is_available(),'MPS required but unavailable')
    try:
        raw=get_dataset(root,c,state,'train');cal=get_dataset(root,c,state,'calibration')
        restore_records(root,c,state)
        state.update(status='calibrating',error=None);save(root/'run_manifest.json',state)
        if not (root/'selection.json').exists():
            if smoke:
                if not state['runs']:train_candidate(root,c,state,raw,cal,.6,'smoke',device)
                plan=dict(reason='Smoke pipeline only; fixed diagnostic weight, independent smoke RNG seeds')
            else:
                while True:
                    if production_continuation and should_stop(root):
                        state.update(status='paused_clean',pause_reason='User stop requested between fits')
                        save(root/'run_manifest.json',state)
                        return
                    stage,pending,plan=search_action(state['runs'],c)
                    state['search_plan']=dict(stage=stage,pending=pending,detail=plan);save(root/'run_manifest.json',state)
                    if stage=='select':break
                    train_candidate(root,c,state,raw,cal,pending[0],stage,device)
                    if state['status'] in ['awaiting_user_approval','paused_clean']:return
                    write_csv(root/'calibration_progress.csv',[dict(stage=r['stage'],w2=r['w2'],pooled_size=r['calibration']['pooled_size']) for r in state['runs']])
            best=min(state['runs'],key=lambda r:(abs(r['calibration']['pooled_size']-c['target_size']),r['w2']))
            save(root/'selection.json',dict(w2=best['w2'],pooled_calibration_size=best['calibration']['pooled_size'],
                 criterion='min abs(pooled held-out H0 size - .05), lower w2 on ties',search_plan=plan,
                 selected_at=stamp(),checkpoint=best['checkpoint'],test_generated=False))
        state['status']='independent_test';save(root/'run_manifest.json',state)
        test=get_dataset(root,c,state,'test');finish(root,c,state,test,device)
        if smoke:
            h=read(Path(state['runs'][0]['path'])/'history.json')
            seconds=statistics.mean(r['train_seconds'] for r in h[1:])
            full_batches=len(c['length_grid'])*math.ceil(60000/256) if c.get('length_grid') else 451*math.ceil(600000/451/256)
            smoke_batches=sum(1 for _ in batches(raw,c['batch_size']))
            estimate=dict(measured_postwarmup_epoch_seconds=seconds,smoke_batches=smoke_batches,
                approximate_full_batches=full_batches,hours_per_200_epoch_fit=seconds*full_batches/smoke_batches*200/3600,
                note='Approximate MPS batch-time extrapolation. Five mandatory coarse fits; each added weight repeats 200 epochs. No accuracy-based shortening.')
            save(root/'runtime_estimate.json',estimate);print(json.dumps(estimate),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save(root/'run_manifest.json',state);raise


def main(default_config=CONFIG):
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--recover-from',type=Path,help='Preserve failed experiment, reuse only immutable train/calibration data')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();c=read(default_config)
    require(len({c['data_seed'],c['calibration_seed'],c['test_seed']})==3,'Dataset seeds must differ')
    require(c['epochs']==200 and c['model_seed']==7 and c['w1']==1.,'Production protocol mismatch')
    root=Path(c['output_dir'])
    if args.output:
        root=args.output;c=dict(c,output_dir=str(root),experiment=root.name)
    require(not args.recover_from or (args.output and root.resolve()!=args.recover_from.resolve()),'Recovery needs a separate output')
    if args.smoke:
        c=dict(c,epochs=3,train_counts=[5,1],calibration_counts=[2,0],test_counts=[2,1])
        for name in ['data_seed','calibration_seed','test_seed']:c[name]+=100000000
        c['experiment']+='_smoke_'+stamp();root=root.with_name(c['experiment']);c['output_dir']=str(root)
    root.mkdir(parents=True,exist_ok=args.resume)
    with (root/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another runner holds this experiment lock')
        run(c,root,args.smoke,args.recover_from)


if __name__=='__main__':main()
