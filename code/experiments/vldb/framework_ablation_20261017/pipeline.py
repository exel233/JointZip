"""Resume immutable SZ2016 stages after the delivered five-backend run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from . import common as c

MODULE='experiments.vldb.framework_ablation_20261017'
FOUNDATION_PYTHON='/opt/jointzip-home/.venvs/nclab_foundation_20260922/bin/python'
def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=pid,start_ticks=fields[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:return None
def stages(workers):
    yield 'smoke_codec',[sys.executable,'-u','-m',MODULE+'.smoke_codec']
    yield 'compression',[sys.executable,'-u','-c',f'from {MODULE}.compress import run; run({workers})']
    for split in ('val','test'):
        for model in ('ridge','dlinear','patchtst','chronos2','chronos_bolt_small','timesfm3'):
            for seed in (c.SEEDS if model=='patchtst' else [0]):
                python=FOUNDATION_PYTHON if model in ('chronos2','chronos_bolt_small','timesfm3') else sys.executable
                yield f'forecast_{split}_{model}_{seed}',[python,'-u','-m',MODULE+'.consumers','forecast','--model',model,'--seed',str(seed),'--split',split]
    # Each UCR neural/1NN instance writes classification and retrieval together.
    for consumer in ('one_nn','logistic','resnet_v1','resnet_v2'):
        for seed in (c.SEEDS if 'resnet' in consumer else [0]):
            yield f'ucr_{consumer}_{seed}',[sys.executable,'-u','-m',MODULE+'.consumers','ucr','--consumer',consumer,'--seed',str(seed)]
    for model in ('ridge_svd64','knn','lstm'):
        for seed in (c.SEEDS if model=='lstm' else [0]):
            yield f'anomaly_{model}_{seed}',[sys.executable,'-u','-m',MODULE+'.consumers','anomaly','--model',model,'--seed',str(seed)]
    yield 'closeout',[sys.executable,'-u','-m',MODULE+'.closeout']
    yield 'report',[sys.executable,'-u','-m',MODULE+'.report']

def run(resume,workers):
    import fcntl
    from .prepare import run as prepare, verify_dependency
    verify_dependency()
    c.RUN.mkdir(parents=True,exist_ok=True)
    with (c.RUN/'RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.RUN/'freeze.json').exists() and not resume: raise RuntimeError('Existing run requires explicit --resume')
        prepare()
        failure=c.RUN/'execution/FAILED.json'
        if failure.exists():
            history=c.RUN/'execution/resume_history'/str(__import__('time').time_ns());history.mkdir(parents=True)
            failure.replace(history/'FAILED.json')
        for label,command in stages(workers):
            folder=c.RUN/'execution/stages'/label;done=folder/'COMPLETE.json'
            active=folder/'ACTIVE.json'
            if active.exists():
                prior=c.read(active)
                if prior.get('child') and process_identity(prior['child']['pid'])==prior['child']:
                    raise RuntimeError('Previous stage process is still active; refuse duplicate: '+label)
            if done.exists():
                previous=c.read(done)
                if previous['command']!=command or not previous['complete']: raise RuntimeError('Stage identity mismatch')
                continue
            folder.mkdir(parents=True,exist_ok=True)
            c.write(c.RUN/'execution/STATUS.json',dict(phase=label,pid=os.getpid(),observed_utc=c.now()))
            print(json.dumps(dict(event='stage_start',stage=label,utc=c.now())),flush=True)
            with (folder/'stdout.log').open('a') as stream:
                process=subprocess.Popen(command,cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                c.write(active,dict(command=command,child=process_identity(process.pid),started_utc=c.now()))
                try: process.wait()
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            if process.returncode: raise RuntimeError(f'Stage {label} exited {process.returncode}; inspect {folder}/stdout.log')
            c.write(done,dict(complete=True,command=command,completed_utc=c.now()),exclusive=True)
        closed=c.read(c.RUN/'audit/closeout/CLOSEOUT.json')
        if closed.get('passed') is not True: raise RuntimeError('Final exact coverage closeout did not pass')
        result=dict(complete=True,compression=27900,consumer_records=220462,completed_utc=c.now())
        c.write(c.RUN/'execution/COMPLETE.json',result);c.write(c.RUN/'execution/STATUS.json',dict(phase='complete',**result))

def main():
    def interrupted(signum,frame): raise KeyboardInterrupt('pipeline terminated by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not args.execute: raise SystemExit('Explicit --execute required')
    try:run(args.resume,args.workers)
    except BaseException as error:
        c.write(c.RUN/'execution/FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),observed_utc=c.now()));raise
if __name__=='__main__':main()
