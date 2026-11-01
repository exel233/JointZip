"""New ablation arms; completed ERA is imported verbatim and never re-encoded."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from . import common as c, codec, container
from .native_cache import backend_fingerprint
from .replay import trajectory_sha256
from experiments.vldb.prior_comparison.common import audit


def configs():
    for entry in c.entries():
        for split in c.splits(entry):
            for point in c.points(entry):
                for backend in c.BACKENDS:
                    for arm in c.ARMS:
                        yield entry,split,point,backend,arm


def name(entry,split,point,backend,arm):
    return '__'.join((entry['task'],entry['sid'],split,backend,arm,point['label']))


def save_bytes(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp'); temp.write_bytes(data); temp.replace(path)


def copy_existing(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if c.sha(target)!=c.sha(source): raise RuntimeError('Reuse destination differs: '+str(target))
        return
    try: os.link(source,target)
    except OSError: shutil.copyfile(source,target)


def historical(entry,split,point,backend,policy='jointzip'):
    root=c.history_run(backend)
    path=root/'compression'/entry['task']/entry['sid']/split/(backend+'_'+policy+'__'+point['label']+'.json')
    row=c.read(path)
    if row['status']!='complete' or row['point']!=point or row['input_sha256']!=entry['input_sha256'] or row['backend']!=backend:
        raise RuntimeError('Historical condition differs: '+str(path))
    for field in ('artifact','reconstruction','metadata'):
        if c.sha(root/row[field])!=row[field+'_sha256']: raise RuntimeError('Historical '+field+' changed')
    meta=c.read(root/row['metadata'])
    if backend_fingerprint(meta['identity'])!=backend_fingerprint(codec.identity(backend)):
        raise RuntimeError('Historical backend identity differs')
    return root,path,row,meta


def enabled_audit(source,rebuilt,point,arm,tolerance=0.):
    # Pearson is bounded below by -1 and ACF gap above by 2. These finite
    # bounds disable tests in the independent audit without JSON infinities.
    result=audit(source,rebuilt,point['epsilon']+tolerance,
                 point['rho']-tolerance if 'R' in arm else -1.,
                 point['tau']+tolerance if 'A' in arm else 2.)
    result['tolerance']=tolerance; result['enabled_contract']=c.contract(arm)
    return result


def import_era(entry,split,point,backend,output):
    root,path,old,meta=historical(entry,split,point,backend)
    if not old['enabled_pass_numerical'] or not old['independent_decode_bitwise']:
        raise RuntimeError('Historical ERA was not complete/verified')
    row=dict(old,method=backend+'_ERA',arm='ERA',framework=True,
             reuse=dict(kind='completed_current_ERA',original_row=str(path),original_row_sha256=c.sha(path),
                        identical_full_stream=True,independent_decode='verified in completed source run; stream unchanged'),
             enabled_audit_strict=old['strict_audit'],enabled_audit_numerical=old['numerical_audit'],
             fresh_speed_valid=False,completed_utc=c.now())
    ident=name(entry,split,point,backend,'ERA')
    for field,folder,suffix in (('artifact','artifacts','.jzf'),('reconstruction','reconstructions','.npy'),('metadata','metadata','.json')):
        target=c.RUN/folder/(ident+suffix);copy_existing(root/old[field],target)
        row[field]=str(target.relative_to(c.RUN));row[field+'_sha256']=old[field+'_sha256']
    c.write(output,row,exclusive=True)
    return dict(condition=ident,origin='completed_current_ERA',cr=row['cr'])


def one(args):
    entry,split,point,backend,arm=args
    method=backend+'_'+arm;output=c.row_path(entry,split,method,point['label']);ident=name(*args)
    if output.exists():
        row=c.read(output)
        if row['status']!='complete' or row['input_sha256']!=entry['input_sha256'] or row['point']!=point or row['arm']!=arm:
            raise RuntimeError('Committed compression identity differs: '+ident)
        for field in ('artifact','reconstruction','metadata'):
            if not (c.RUN/row[field]).is_file(): raise RuntimeError('Missing committed '+field)
        return dict(condition=ident,origin='already_complete',cr=row['cr'])
    if arm=='ERA': return import_era(entry,split,point,backend,output)
    data=c.arrays(entry);source=np.ascontiguousarray(data['val_source' if split=='val' else 'source'],dtype=np.float32)
    if entry['task']=='anomaly' and entry['true_length']!=len(data['original'])-int(data['train_end']):
        raise RuntimeError('NASA true suffix length differs')
    limits=c.arm_constraints(point,arm);work=c.RUN/'work'/ident;work.mkdir(parents=True,exist_ok=True)
    progress=c.RUN/'execution/conditions'/(ident+'.json')
    cache=codec.NativeCandidateCache(c.RUN/'native_candidate_cache'/backend)
    start=time.perf_counter();trace=None;trace_path=None;imported=0
    for policy in ('jointzip','native'):
        root,path,old,metadata=historical(entry,split,point,backend,policy)
        outer,payload=container.unframe((root/old['artifact']).read_bytes())
        if old['shape']!=list(source.shape): raise RuntimeError('Historical source shape differs')
        imported+=cache.import_stream(backend,source,metadata,payload,work/'import_base',
                                     dict(row=str(path),input_sha256=entry['input_sha256']))
        if policy=='jointzip':trace=metadata;trace_path=path
    import_seconds=time.perf_counter()-start
    checkpoints=c.RUN/'window_checkpoints'/ident
    warm=codec.WindowCache();resume_records={};resumed=0
    if arm in ('E','ER','EA'):
        proof=c.read(c.ROOT/'tmp/runtime_framework_ablation_20261017/CODEC_READY.json')
        match=proof['real_source']
        if (entry['sid']==match['entry'] and entry['input_sha256']==match['input_sha256']
                and split==match['split'] and point==proof['threshold']):
            folder=c.RUN/'audit/preflight_codec'/backend/(arm+'_cached')
            warm.add(source[:1],c.read(folder/'metadata.json'),(folder/'stream.jzp').read_bytes(),
                     dict(kind='completed_matching_preflight_window',source=str(folder)),workdir=work/'preflight_decode')
    options=codec.search_options(backend,True)
    if arm=='ERA_no_correction':
        options=dict(trace['search_options'],policy='fixed_era_trace_no_correction',
                     trace_sha256=trajectory_sha256(trace),adaptive_refinement=False)
    def source_digest(i):return hashlib.sha256(source[i].tobytes()).hexdigest()
    if checkpoints.exists():
        for path in sorted(checkpoints.glob('*.json')):
            record=c.read(path);i=record['index']
            if record['source_sha256']!=source_digest(i) or record['identity']!=codec.identity(backend) or record['options']!=options or record['limits']!=limits:
                raise RuntimeError('Window resume binding differs: '+str(path))
            payload=path.with_suffix('.jzp').read_bytes()
            if hashlib.sha256(payload).hexdigest()!=record['payload_sha256']:raise RuntimeError('Window checkpoint changed')
            header,records=codec.unframe(payload)
            if arm=='ERA_no_correction':resume_records[i]=dict(record=records[0],evidence=record['evidence'])
            else:
                warm.add_records(backend,source[i:i+1],limits['epsilon'],limits['tau'],limits['rho'],16,True,
                                 records,[record['evidence']],dict(kind='completed_window_checkpoint'),workdir=work/'resume_decode')
            resumed+=1
    def checkpoint(index,record,evidence,opts):
        path=checkpoints/(str(index).zfill(6)+'.json')
        if not path.exists():
            payload,_=codec._frame(backend,(1,1,source.shape[2]),limits['epsilon'],limits['tau'],limits['rho'],16,[record])
            save_bytes(path.with_suffix('.jzp'),payload)
            c.write(path,dict(index=index,source_sha256=source_digest(index),identity=codec.identity(backend),
                    options=opts,limits=limits,payload_sha256=hashlib.sha256(payload).hexdigest(),evidence=evidence),exclusive=True)
        c.write(progress,dict(condition=ident,phase='encoding',completed_windows=index+1,total_windows=len(source),observed_utc=c.now()))
    if arm=='ERA_no_correction':
        encoded=codec.replay_without_correction(backend,source,point['epsilon'],point['tau'],point['rho'],16,trace,
                    workdir=work/'encode',native_cache=cache,window_callback=checkpoint,resume_records=resume_records)
    else:
        encoded=codec.encode(backend,source,limits['epsilon'],limits['tau'],limits['rho'],16,framework=True,
                    workdir=work/'encode',native_cache=cache,warm_cache=warm,window_callback=checkpoint)
    cache.close()
    payload,meta=container.frame(encoded.payload,entry,split,source.shape)
    artifact=c.RUN/'artifacts'/(ident+'.jzf');save_bytes(artifact,payload)
    begin=time.perf_counter();rebuilt,decoded_meta=container.decode(payload,work/'independent_decode');decode_seconds=time.perf_counter()-begin
    if rebuilt.dtype!=np.float32 or rebuilt.shape!=source.shape or not np.isfinite(rebuilt).all() or rebuilt.tobytes()!=encoded.reconstruction.tobytes():
        raise RuntimeError('Independent final stream decode differs')
    tol=float(64*np.finfo(np.float64).eps)
    strict=audit(source,rebuilt,point['epsilon'],point['rho'],point['tau'])
    numerical=audit(source,rebuilt,point['epsilon']+tol,point['rho']-tol,point['tau']+tol);numerical['tolerance']=tol
    en_strict=enabled_audit(source,rebuilt,point,arm);en_numerical=enabled_audit(source,rebuilt,point,arm,tol)
    if not en_numerical['joint_pass']:raise RuntimeError('Enabled contract failed: '+ident)
    reconstruction=c.RUN/'reconstructions'/(ident+'.npy');reconstruction.parent.mkdir(parents=True,exist_ok=True)
    with reconstruction.with_suffix('.tmp').open('wb') as f:np.save(f,rebuilt,allow_pickle=False)
    reconstruction.with_suffix('.tmp').replace(reconstruction)
    metadata=c.RUN/'metadata'/(ident+'.json')
    if metadata.exists():
        history=c.RUN/'execution/resume_history'/('uncommitted_metadata_'+str(time.time_ns()))
        history.mkdir(parents=True);metadata.replace(history/metadata.name)
    c.write(metadata,encoded.metadata,exclusive=True)
    row=dict(status='complete',task=entry['task'],sid=entry['sid'],dataset=entry['dataset'],split=split,
        method=method,backend=backend,framework=True,arm=arm,point=point,encoder_constraints=limits,
        input_sha256=entry['input_sha256'],shape=list(source.shape),raw_bytes=meta['true_count']*4,
        artifact_bytes=len(payload),cr=meta['true_count']*4/len(payload),
        artifact=str(artifact.relative_to(c.RUN)),artifact_sha256=c.sha(artifact),
        reconstruction=str(reconstruction.relative_to(c.RUN)),reconstruction_sha256=c.sha(reconstruction),
        metadata=str(metadata.relative_to(c.RUN)),metadata_sha256=c.sha(metadata),
        strict_audit=strict,numerical_audit=numerical,enabled_audit_strict=en_strict,enabled_audit_numerical=en_numerical,
        enabled_contract=c.contract(arm),enabled_pass_strict=en_strict['joint_pass'],enabled_pass_numerical=en_numerical['joint_pass'],
        independent_decode_bitwise=True,container_metadata=decoded_meta,reuse=dict(kind='missing_ablation_configuration',imported_native_members=imported),
        resumed_windows=resumed,reuse_import_seconds=import_seconds,encoding_diagnostic=encoded.metadata['encode_seconds'],
        fresh_speed_valid=False,decode_diagnostic_seconds=decode_seconds,completed_utc=c.now())
    if arm=='ERA_no_correction':
        row['correction_ablation']=dict(trace_row=str(trace_path),trace_row_sha256=c.sha(trace_path),
            trace_sha256=trajectory_sha256(trace),fixed_logical_caps=True,repair_disabled=True)
    c.write(output,row,exclusive=True)
    c.write(progress,dict(condition=ident,phase='complete',completed_windows=len(source),total_windows=len(source),observed_utc=c.now()))
    return dict(condition=ident,origin='new_ablation',cr=row['cr'])


def group(args):
    # Same input/backend/point is processed serially, maximizing cross-arm reuse.
    return [one((*args,arm)) for arm in ('ERA','E','ER','EA','ERA_no_correction')]


def run(workers=4):
    jobs=[(e,s,p,b) for e in c.entries() for s in c.splits(e) for p in c.points(e) for b in c.BACKENDS]
    done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(group,args):args for args in jobs}
        for future in as_completed(futures):
            try:results=future.result()
            except Exception:
                for other in futures:other.cancel()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
            for result in results:
                done+=1
                c.write(c.RUN/'execution/compression_status.json',dict(completed=done,total=len(jobs)*5,last=result,observed_utc=c.now()))
                print(__import__('json').dumps(dict(event='compression_complete',completed=done,total=len(jobs)*5,**result)),flush=True)
    c.write(c.RUN/'execution/COMPRESSION_COMPLETE.json',dict(complete=True,conditions=done,completed_utc=c.now()))
