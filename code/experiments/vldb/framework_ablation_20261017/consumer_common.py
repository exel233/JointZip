"""Small, explicit persistence/reuse helpers for the fixed-backend consumers.

No training, codec search, or changes to the historical experiment modules.
"""
from __future__ import annotations
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import common as c


def rel(path):
    return str(Path(path).relative_to(c.ROOT)).replace('\\', '/')


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_npz(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def key(row):
    return 'raw' if row is None else row['method']+'__'+row['point']['label']


def binding(path):
    return {'path': rel(path), 'sha256': c.sha(path)}


def checked(binding_value):
    path = c.ROOT / binding_value['path']
    if c.sha(path) != binding_value['sha256']:
        raise RuntimeError('Historical consumer dependency changed: '+str(path))
    return path


def freeze(task, parent_manifest, extra_files=()):
    """Bind the selected dependencies once; c.sha has a stat-keyed cache."""
    path = c.RUN/'evaluation'/task/'consumer_manifest.json'
    files = list(Path(__file__).parent.glob('consumer*.py'))
    files += [Path(c.__file__), c.RUN/'freeze.json', Path(parent_manifest), *map(Path, extra_files)]
    identity = dict(schema='jointzip.fixed-backend.consumers.v1', task=task,
                    files={rel(p): c.sha(p) for p in sorted(set(files))},
                    parent_manifest=binding(parent_manifest), seeds=list(c.SEEDS),
                    policy='Frozen trained consumers; fixed batches/device. Reuse requires exact source/reconstruction and a successful raw replay. No TEST parameter selection.')
    c.write(path, identity, exclusive=True)
    return path, c.sha(path)


def base(entry, row, manifest_sha, task, model, seed, split='test'):
    return dict(task=task, model=model, consumer=model, seed=seed, sid=entry['sid'],
        dataset=entry['dataset'], split=split, input_sha256=entry['input_sha256'],
        consumer_manifest_sha256=manifest_sha, method='raw' if row is None else row['method'],
        point='raw' if row is None else row['point']['label'],
        compression_row=None if row is None else row['row_path'],
        compression_row_sha256=None if row is None else c.sha(c.RUN/row['row_path']),
        artifact_bytes=None if row is None else row.get('artifact_bytes'),
        cr=1. if row is None else row.get('cr'),
        enabled_contract=None if row is None else row.get('enabled_contract'),
        enabled_pass_numerical=True if row is None else row.get('enabled_pass_numerical'),
        enabled_pass_strict=True if row is None else row.get('enabled_pass_strict'))


def finished(path, expected):
    if not path.exists():
        return False
    old = c.read(path)
    for field in ('task','model','seed','sid','split','input_sha256','consumer_manifest_sha256',
                  'method','point','compression_row_sha256'):
        if old.get(field) != expected.get(field):
            raise RuntimeError('Consumer resume identity changed: '+str(path)+':'+field)
    if old['status'] != 'complete':
        raise RuntimeError('Incomplete committed consumer row: '+str(path))
    if c.sha(c.ROOT/old['prediction']) != old['prediction_sha256']:
        raise RuntimeError('Saved consumer arrays changed: '+str(path))
    return True


def persist(path, result, values, metrics, origin, lineage=None):
    arr = path.with_suffix('.npz')
    save_npz(arr, values)
    result.update(status='complete', metrics=metrics, prediction=rel(arr),
                  prediction_sha256=c.sha(arr), prediction_origin=origin,
                  lineage={} if lineage is None else lineage)
    # Keep one common prediction field for all consumers and legacy-compatible aliases.
    if result['task'] in ('classification','retrieval'):
        result.update(per_query_file=str(arr.relative_to(c.RUN)), per_query_sha256=c.sha(arr))
    if result['task'] == 'anomaly':
        result.update(scores=str(arr.relative_to(c.RUN)), scores_sha256=c.sha(arr))
    c.write(path, result, exclusive=True)


def historical_key(row):
    if row is None:
        return 'raw'
    # A method name only identifies a lookup candidate. Exact input equality below
    # is mandatory, including for JointZip outputs which happen to match a baseline.
    method = row.get('reuse', {}).get('historical_method')
    if method is None:
        method = row.get('backend') or row['method'].removesuffix('_native').removesuffix('_jointzip')
    if method == 'serf_xor':
        method = 'serf'
    if method not in ('sz2016','sz3','zfp','sperr','serf','serf_qt','zstd'):
        return None
    return method+'__'+row['point']['label']


def parent_reconstruction(entry, historical_row):
    cp = historical_row.get('compression_row')
    if not cp:
        return None
    path = c.PARENT_RUN/cp
    old = c.read(path)
    if old['status'] != 'complete' or old['sid'] != entry['sid']:
        raise RuntimeError('Historical compression row unavailable')
    rp = c.PARENT_RUN/old['reconstruction']
    if c.sha(rp) != old['reconstruction_sha256']:
        raise RuntimeError('Historical reconstruction changed')
    return np.load(rp, allow_pickle=False), binding(path)


def parent_values(path, field, hash_field):
    row = c.read(path)
    if row.get('status') != 'complete':
        raise RuntimeError('Historical consumer not complete: '+str(path))
    value_path = (c.ROOT if field == 'prediction' else c.PARENT_RUN)/row[field]
    if c.sha(value_path) != row[hash_field]:
        raise RuntimeError('Historical consumer arrays changed')
    return row, npz(value_path), {'row': binding(path), 'arrays': binding(value_path)}


def ensure_parent_sources(parent, key='files'):
    """Verify saved model/preprocessing bindings, not a recursive directory hash."""
    for path, sha in parent.get(key, {}).items():
        if c.sha(c.ROOT/path) != sha:
            raise RuntimeError('Frozen consumer/model binding changed: '+path)


# Current fixed-backend ERA results are a completed experiment, not a speculative
# cache. Import their model outputs without launching those models again. All
# checks below read the small bound manifests or the specific arrays being used.
@lru_cache(maxsize=8)
def history_manifest(root, family):
    root=Path(root)
    path=root/'evaluation'/family/'consumer_manifest.json'
    value=c.read(path)
    parent_name='protocol.json' if family=='anomaly' else 'consumer_manifest.json'
    parent=c.PARENT_RUN/'evaluation'/family/parent_name
    if value['parent_manifest']!=binding(parent) or value['seeds']!=list(c.SEEDS):
        raise RuntimeError('Completed framework parent consumer binding differs: '+str(path))
    # These include the actual frozen preprocessing/model consumer code, the
    # parent model manifest and the historical experiment freeze, not new code.
    ensure_parent_sources(value)
    closeout=c.read(root/'audit/closeout/CLOSEOUT.json')
    if not closeout.get('passed'):
        raise RuntimeError('Cannot reuse an incomplete framework experiment')
    return binding(path)


@lru_cache(maxsize=2)
def history_sources(task, sid, split):
    """Small per-input index: original arrays and the 12 completed methods.

    Exact arrays, rather than method labels or statistics, determine reuse.
    Cache only two inputs so the full UCR query matrices do not accumulate.
    """
    out=[]
    for root in (Path(c.HISTORY_FULL),Path(c.HISTORY_SZ)):
        frozen=c.read(root/'freeze.json')
        entries=[e for e in frozen['entries'] if e['task']==task and e['sid']==sid]
        if len(entries)!=1: raise RuntimeError('Historical input roster differs: '+sid)
        entry=entries[0]
        ip=root/entry['input']
        if c.sha(ip)!=entry['input_sha256']: raise RuntimeError('Historical input changed')
        data=npz(ip)
        if task=='forecast': raw=data['val_source' if split=='val' else 'source'][:,0,-96:]
        elif task=='ucr': raw=data['raw']
        else: raw=data['original'][int(data['train_end']):]
        out.append(dict(root=root,entry=entry,method='raw',point='raw',input=raw,
                        digest=digest(raw),compression=None))
        folder=root/'compression'/task/sid/split
        for path in sorted(folder.glob('*.json')):
            row=c.read(path)
            if row['status']!='complete' or row['input_sha256']!=entry['input_sha256']:
                raise RuntimeError('Historical compression identity differs: '+str(path))
            rp=root/row['reconstruction']
            if c.sha(rp)!=row['reconstruction_sha256']: raise RuntimeError('Historical reconstruction changed')
            rec=np.load(rp,allow_pickle=False)
            if task=='forecast': query=rec[:,0,-96:]
            elif task=='ucr': query=rec[:,0,:]
            else: query=rec.ravel()[:len(raw)]
            out.append(dict(root=root,entry=entry,method=row['method'],point=row['point']['label'],
                            input=query,digest=digest(query),compression=path))
    return out


def _history_path(root,task,model,seed,entry,split,key):
    if task=='forecast':
        return root/'evaluation/forecast/rows'/model/str(seed)/entry['sid']/split/(key+'.json')
    if task in ('classification','retrieval'):
        return root/'evaluation/ucr'/task/model/entry['sid']/str(seed)/(key+'.json')
    return root/'evaluation/anomaly/rows'/entry['sid']/(model+'_'+str(seed))/(key+'.json')


def _history_replay(root,task,model,seed,entry,split):
    if task=='forecast':
        path=root/'evaluation/forecast/raw_replays'/model/str(seed)/entry['sid']/(split+'.json')
        runtime_path=root/'evaluation/forecast/runtime'/(model+'__'+str(seed)+'.json')
        current=c.RUN/'evaluation/forecast/runtime'/(model+'__'+str(seed)+'.json')
        if c.read(runtime_path)!=c.read(current): raise RuntimeError('Historical forecast runtime differs')
    elif task in ('classification','retrieval'):
        replay_model='one_nn' if model=='shape' else model
        path=root/'evaluation/ucr/raw_replays'/replay_model/entry['sid']/(str(seed)+'.json')
        runtime_path=root/'evaluation/ucr/runtime.json'
        if c.read(runtime_path)!=c.read(c.RUN/'evaluation/ucr/runtime.json'):
            raise RuntimeError('Historical UCR runtime differs')
    else:
        path=root/'evaluation/anomaly/raw_replays'/entry['sid']/(model+'_'+str(seed)+'.json')
    value=c.read(path)
    if not value.get('passed') or (task=='anomaly' and not value.get('cache_reuse_allowed')):
        raise RuntimeError('Historical raw replay does not authorize exact reuse: '+str(path))
    return binding(path)


def completed_history(entry,row,task,model,seed,split,query):
    """Return matching completed output plus provenance; never infer a score.

    ERA must match its own backend/point exactly. Other arms may reuse any
    already-computed identical consumer input under the same frozen consumer.
    """
    family='ucr' if task in ('classification','retrieval') else task
    candidates=history_sources(entry['task'],entry['sid'],split)
    is_era=row is not None and (row.get('arm')=='ERA' or row['method'].endswith('_ERA'))
    if row is None:
        candidates=[x for x in candidates if x['method']=='raw']
    elif is_era:
        expected_root=Path(c.history_run(row['backend']))
        candidates=[x for x in candidates if x['root']==expected_root and
                    x['method']==row['backend']+'_jointzip' and x['point']==row['point']['label']]
        if len(candidates)!=1: raise RuntimeError('Required completed ERA source missing')
    else:
        # Prefer the same backend and threshold when several arrays coincide.
        candidates=sorted(candidates,key=lambda x:(x['method']!=row['backend']+'_jointzip',
                                                   x['point']!=row['point']['label']))
    sha=digest(query)
    for item in candidates:
        if item['entry']['input_sha256']!=entry['input_sha256']:
            raise RuntimeError('Historical frozen source differs')
        if item['digest']!=sha or not bits(item['input'],query): continue
        root=item['root']; manifest=history_manifest(str(root),family)
        replay=_history_replay(root,task,model,seed,entry,split)
        key='raw' if item['method']=='raw' else item['method']+'__'+item['point']
        path=_history_path(root,task,model,seed,entry,split,key)
        record=c.read(path)
        identity=dict(task=task,model=model,seed=seed,sid=entry['sid'],split=split,
                      method=item['method'],point=item['point'],input_sha256=entry['input_sha256'])
        if any(record.get(k)!=v for k,v in identity.items()) or record.get('status')!='complete':
            raise RuntimeError('Historical consumer identity mismatch: '+str(path))
        if record['consumer_manifest_sha256']!=manifest['sha256'] or record['consumer_input_sha256']!=sha:
            raise RuntimeError('Historical consumer input/manifest binding differs')
        if item['compression'] is not None and record['compression_row_sha256']!=c.sha(item['compression']):
            raise RuntimeError('Historical consumer compression binding differs')
        prediction=c.ROOT/record['prediction']
        if c.sha(prediction)!=record['prediction_sha256']: raise RuntimeError('Historical prediction changed')
        lineage=dict(row=binding(path),arrays=binding(prediction),consumer_manifest=manifest,
                     completed_raw_replay=replay,match_basis='bitwise_effective_consumer_input',
                     no_model_call=True,era_direct=is_era)
        if item['compression'] is not None: lineage['compression']=binding(item['compression'])
        return record,npz(prediction),lineage
    if is_era or row is None:
        raise RuntimeError('Required completed ERA/Raw input does not match; refuse rerun')
    # Resume-safe deduplication also covers outputs already committed by an
    # earlier arm in this run, rather than only the current in-memory batch.
    folder=_history_path(c.RUN,task,model,seed,entry,split,'unused').parent
    current_manifest=binding(c.RUN/'evaluation'/family/'consumer_manifest.json')
    for path in sorted(folder.glob('*.json')):
        record=c.read(path)
        if (record.get('status')!='complete' or record.get('consumer_input_sha256')!=sha
                or record.get('input_sha256')!=entry['input_sha256']): continue
        if record['consumer_manifest_sha256']!=current_manifest['sha256']:
            raise RuntimeError('Current consumer manifest differs')
        if any(record.get(k)!=v for k,v in dict(task=task,model=model,seed=seed,sid=entry['sid'],split=split).items()):
            raise RuntimeError('Current cached consumer identity differs')
        if record['method']=='raw':
            data=c.arrays(entry)
            if task=='forecast': previous=data['val_source' if split=='val' else 'source'][:,0,-96:]
            elif family=='ucr': previous=data['raw']
            else: previous=data['original'][int(data['train_end']):]
            compression=None
        else:
            compression=c.RUN/record['compression_row']
            if c.sha(compression)!=record['compression_row_sha256']:
                raise RuntimeError('Current cached compression changed')
            rec=c.decoded(c.read(compression))
            previous=rec[:,0,-96:] if task=='forecast' else rec[:,0,:] if family=='ucr' else rec.ravel()[:len(query)]
        if not bits(previous,query): raise RuntimeError('Current consumer input digest collision')
        prediction=c.ROOT/record['prediction']
        if c.sha(prediction)!=record['prediction_sha256']: raise RuntimeError('Current cached prediction changed')
        lineage=dict(row=binding(path),arrays=binding(prediction),consumer_manifest=current_manifest,
                     match_basis='bitwise_current_committed_consumer_input',no_model_call=True,
                     era_direct=False,current_run_reuse=True)
        if compression is not None: lineage['compression']=binding(compression)
        if 'raw_replay' in record: lineage['completed_raw_replay']=record['raw_replay']
        return record,npz(prediction),lineage
    return None


def import_completed(entry,row,result,path,task,model,seed,split,query):
    found=completed_history(entry,row,task,model,seed,split,query)
    if found is None: return False
    record,values,lineage=found
    for field in ('consumer','consumer_input_sha256','queries','threshold','vus_window','test_points',
                  'decoded_true_suffix_sha256','labels_sha256','score_array_sha256'):
        if field in record: result[field]=record[field]
    if task=='anomaly': result['raw_replay']=lineage['completed_raw_replay']
    origin=('current_committed_identical_consumer_input' if lineage.get('current_run_reuse') else
            'completed_framework_ERA_reuse' if lineage['era_direct'] else
            'completed_framework_raw_reuse' if row is None else 'completed_framework_identical_input_reuse')
    persist(path,result,values,record['metrics'],origin,lineage)
    return True
