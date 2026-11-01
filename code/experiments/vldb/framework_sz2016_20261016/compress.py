"""Fixed-backend complete streams, with verified historical and window reuse."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from . import common as c, codec, container, reuse
from experiments.vldb.prior_comparison.common import audit

def configs():
    # VAL is fully compressed before TEST consumers; no selection uses TEST scores.
    for entry in c.entries():
        for split in c.splits(entry):
            for point in c.points(entry):
                for backend in c.BACKENDS:
                    for framework in (False,True): yield entry,split,point,backend,framework

def name(entry,split,point,backend,framework):
    return '__'.join((entry['task'],entry['sid'],split,backend,'jointzip' if framework else 'native',point['label']))

def save_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_bytes(data);tmp.replace(path)

def one(args):
    entry,split,point,backend,framework=args
    method=backend+('_jointzip' if framework else '_native')
    output=c.row_path(entry,split,method,point['label'])
    ident=name(*args)
    if output.exists():
        row=c.read(output)
        if row['status']!='complete' or row['input_sha256']!=entry['input_sha256'] or row['point']!=point:
            raise RuntimeError('Committed compression identity differs: '+ident)
        for field in ('artifact','reconstruction','metadata'):
            if not (c.RUN/row[field]).is_file(): raise RuntimeError('Missing committed '+field+': '+ident)
        return dict(condition=ident,origin='already_complete',cr=row['cr'])
    data=c.arrays(entry);source=np.ascontiguousarray(data['val_source' if split=='val' else 'source'],dtype=np.float32)
    if entry['task']=='anomaly' and entry['true_length']!=len(data['original'])-int(data['train_end']):
        raise RuntimeError('NASA true suffix length differs from frozen input')
    work=c.RUN/'work'/ident;work.mkdir(parents=True,exist_ok=True)
    progress=c.RUN/'execution/conditions'/(ident+'.json')
    cache=codec.WindowCache(); lineage={}; imported=None
    started=time.perf_counter()
    if not framework:
        imported=reuse.load_native(entry,split,backend,point,source)
        if imported is not None:
            cache.add_records(backend,source,point['epsilon'],point['tau'],point['rho'],16,False,
                imported['records'],None,imported['lineage'],workdir=work/'historical_decode')
            lineage=dict(imported['lineage'],historical_method=imported['lineage']['original_method'])
    if imported is None and framework and backend=='sz3':
        # The prelaunch consumer check computed real complete searches on the
        # original full inputs. Reuse those bytes too, not its smoke label/scores.
        smoke=Path('/opt/jointzip-home/.local/cth_expansion_runs/vldb_framework_full_consumer_smoke_20261015_v1')
        prior_path=smoke/'compression'/entry['task']/entry['sid']/split/(method+'__'+point['label']+'.json')
        if prior_path.exists():
            prior=c.read(prior_path)
            if prior['status']!='complete' or prior['point']!=point or prior['input_sha256']!=entry['input_sha256']:
                raise RuntimeError('Completed prelaunch search binding differs')
            with np.load(smoke/entry['input'],allow_pickle=False) as z:
                key='val_source' if split=='val' else 'source'
                if z[key].tobytes()!=source.tobytes() or z[key].shape!=source.shape: raise RuntimeError('Prelaunch source differs')
            for field in ('artifact','metadata','reconstruction'):
                if c.sha(smoke/prior[field])!=prior[field+'_sha256']: raise RuntimeError('Completed prelaunch artifact changed')
            meta,inner=container.unframe((smoke/prior['artifact']).read_bytes())
            lineage=dict(kind='completed_real_prelaunch_search',original_row=str(prior_path),
                original_row_sha256=c.sha(prior_path),windows=len(source),independent_current_decode_required=True,
                timing='Reuse only; prelaunch wall time is not fresh production encoder speed.')
            cache.add(source,c.read(smoke/prior['metadata']),inner,lineage,workdir=work/'prelaunch_decode')
            imported=True
    if imported is None:
        pilot=reuse.load_pilot(entry,split,backend,point,source,framework=framework)
        if pilot is not None:
            ix=pilot['indices']
            cache.add_records(backend,source[ix],point['epsilon'],point['tau'],point['rho'],16,framework,
                pilot['records'],pilot['evidence'],pilot['lineage'],workdir=work/'pilot_decode')
            lineage=pilot['lineage']
    checkpoints=c.RUN/'window_checkpoints'/ident
    options=codec.search_options(backend,framework)
    source_digest=lambda i:hashlib.sha256(source[i].tobytes()).hexdigest()
    resumed=0
    if checkpoints.exists():
        for path in sorted(checkpoints.glob('*.json')):
            record=c.read(path);i=record['index']
            if record['source_sha256']!=source_digest(i) or record['identity']!=codec.identity(backend) or record['options']!=options:
                raise RuntimeError('Window resume binding differs: '+str(path))
            payload=path.with_suffix('.jzp').read_bytes()
            if hashlib.sha256(payload).hexdigest()!=record['payload_sha256']: raise RuntimeError('Window checkpoint changed')
            header,records=codec.unframe(payload)
            cache.add_records(backend,source[i:i+1],point['epsilon'],point['tau'],point['rho'],16,framework,
                records,[record['evidence']],{'kind':'interrupted_current_run_window','window':i},workdir=work/'resume_decode')
            resumed+=1
    import_seconds=time.perf_counter()-started
    def checkpoint(index,record,evidence,opts):
        path=checkpoints/(str(index).zfill(6)+'.json')
        if not path.exists() and not evidence.get('reused'):
            payload,_=codec._frame(backend,(1,1,source.shape[2]),point['epsilon'],point['tau'],point['rho'],16,[record])
            save_bytes(path.with_suffix('.jzp'),payload)
            c.write(path,dict(index=index,source_sha256=source_digest(index),identity=codec.identity(backend),
                options=opts,payload_sha256=hashlib.sha256(payload).hexdigest(),evidence=evidence),exclusive=True)
        c.write(progress,dict(condition=ident,phase='encoding',completed_windows=index+1,total_windows=len(source),observed_utc=c.now()))
    encoded=codec.encode(backend,source,point['epsilon'],point['tau'],point['rho'],16,framework=framework,
                         workdir=work/'encode',warm_cache=cache,window_callback=checkpoint)
    payload,meta=container.frame(encoded.payload,entry,split,source.shape)
    artifact=c.RUN/'artifacts'/(ident+'.jzf');save_bytes(artifact,payload)
    begin=time.perf_counter();rebuilt,decoded_meta=container.decode(payload,work/'independent_decode');decode_seconds=time.perf_counter()-begin
    if rebuilt.dtype!=np.float32 or rebuilt.shape!=source.shape or not np.isfinite(rebuilt).all() or rebuilt.tobytes()!=encoded.reconstruction.tobytes():
        raise RuntimeError('Independent final stream decode differs: '+ident)
    strict=audit(source,rebuilt,point['epsilon'],point['rho'],point['tau'])
    tol=float(64*np.finfo(np.float64).eps)
    numerical=audit(source,rebuilt,point['epsilon']+tol,point['rho']-tol,point['tau']+tol);numerical['tolerance']=tol
    if framework and not numerical['joint_pass']: raise RuntimeError('Final E/R/A contract failed: '+ident)
    reconstruction=c.RUN/'reconstructions'/(ident+'.npy');reconstruction.parent.mkdir(parents=True,exist_ok=True)
    with reconstruction.with_suffix('.tmp').open('wb') as f: np.save(f,rebuilt,allow_pickle=False)
    reconstruction.with_suffix('.tmp').replace(reconstruction)
    metadata=c.RUN/'metadata'/(ident+'.json')
    if metadata.exists():
        history=c.RUN/'execution/resume_history'/('uncommitted_metadata_'+str(time.time_ns()))
        history.mkdir(parents=True,exist_ok=True);metadata.replace(history/metadata.name)
    c.write(metadata,encoded.metadata,exclusive=True)
    raw_bytes=meta['true_count']*4
    row=dict(status='complete',task=entry['task'],sid=entry['sid'],dataset=entry['dataset'],split=split,method=method,
        backend=backend,framework=framework,point=point,input_sha256=entry['input_sha256'],shape=list(source.shape),
        raw_bytes=raw_bytes,artifact_bytes=len(payload),cr=raw_bytes/len(payload),
        artifact=str(artifact.relative_to(c.RUN)),artifact_sha256=c.sha(artifact),
        reconstruction=str(reconstruction.relative_to(c.RUN)),reconstruction_sha256=c.sha(reconstruction),
        metadata=str(metadata.relative_to(c.RUN)),metadata_sha256=c.sha(metadata),
        strict_audit=strict,numerical_audit=numerical,enabled_contract='ERA' if framework else 'E',
        enabled_pass_strict=bool(strict['joint_pass'] if framework else strict['point_pass_count']==strict['series_windows']),
        enabled_pass_numerical=bool(numerical['joint_pass'] if framework else numerical['point_pass_count']==numerical['series_windows']),
        independent_decode_bitwise=True,container_metadata=decoded_meta,reuse=lineage,resumed_windows=resumed,
        reuse_import_seconds=import_seconds,encoding_diagnostic=encoded.metadata['encode_seconds'],
        fresh_speed_valid=False,decode_diagnostic_seconds=decode_seconds,completed_utc=c.now())
    c.write(output,row,exclusive=True)
    c.write(progress,dict(condition=ident,phase='complete',completed_windows=len(source),total_windows=len(source),observed_utc=c.now()))
    return dict(condition=ident,origin=lineage.get('kind','fresh'),cr=row['cr'])

def run(workers=4):
    jobs=list(configs());done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,args):name(*args) for args in jobs}
        for future in as_completed(futures):
            try: result=future.result()
            except Exception:
                for other in futures: other.cancel()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
            done+=1
            c.write(c.RUN/'execution/compression_status.json',dict(completed=done,total=len(jobs),last=result,observed_utc=c.now()))
            print(__import__('json').dumps(dict(event='compression_complete',completed=done,total=len(jobs),**result)),flush=True)
    c.write(c.RUN/'execution/COMPRESSION_COMPLETE.json',dict(complete=True,conditions=done,completed_utc=c.now()))
