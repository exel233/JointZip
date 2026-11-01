"""Six frozen forecast families, fixed batch128, exact-input prediction reuse."""
from __future__ import annotations
from contextlib import nullcontext
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import numpy as np
from . import common as c
from . import consumer_common as h
from experiments.vldb.paper_budget_guided_20261013_v1 import forecast_consumers as old

FAMILIES = old.FAMILIES
INSTANCES = [(m,s) for m in FAMILIES for s in (c.SEEDS if m == 'patchtst' else [0])]
OUT = c.RUN/'evaluation/forecast'


def output(model, seed, entry, split, key):
    return OUT/'rows'/model/str(seed)/entry['sid']/split/(key+'.json')


def runtime(model, seed):
    import torch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1); torch.manual_seed(seed or 20260922)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    packages = {p: importlib.metadata.version(p) for p in ('numpy','torch')}
    if model in old.FOUNDATIONS:
        packages.update({p: importlib.metadata.version(p) for p in
                        ('chronos-forecasting','timesfm','transformers','safetensors','accelerate','einops')})
    value = dict(model=model, seed=seed, python=platform.python_version(), packages=packages,
                 device='cpu' if model == 'ridge' else 'cuda',
                 gpu=None if model == 'ridge' else torch.cuda.get_device_name(0),
                 batch=128, tf32=False, deterministic=True)
    parent = c.read(c.PARENT_RUN/'evaluation/forecast/runtime'/(model+'__'+str(seed)+'.json'))
    if parent != value:
        raise RuntimeError('Frozen forecast runtime differs: '+json.dumps({'current':value,'parent':parent}))
    c.write(OUT/'runtime'/(model+'__'+str(seed)+'.json'), value, exclusive=True)


def parent_prediction(model, seed, entry, split, row, reconstruction=None):
    key = h.historical_key(row)
    if key is None:
        return None
    path = c.PARENT_RUN/'evaluation/forecast/rows'/model/str(seed)/entry['sid']/split/(key+'.json')
    if not path.exists():
        return None
    previous, values, lineage = h.parent_values(path, 'prediction', 'prediction_sha256')
    if previous['input_sha256'] != entry['input_sha256']:
        raise RuntimeError('Historical forecast input binding differs')
    if previous['consumer_manifest_sha256'] != c.sha(c.PARENT_RUN/'evaluation/forecast/consumer_manifest.json'):
        raise RuntimeError('Historical forecast manifest binding differs')
    if row is not None:
        prior, cp = h.parent_reconstruction(entry, previous)
        if not h.bits(prior, reconstruction):
            return None
        lineage['compression'] = cp
        lineage['match_basis'] = 'bitwise_complete_reconstruction'
    return values, lineage


def evaluate_entry(model, seed, net, entry, split, manifest_sha):
    data = c.arrays(entry)
    source = data['val_source' if split == 'val' else 'source']
    future = data['val_future' if split == 'val' else 'future']
    rows = [None, *c.rows(entry, split)]
    pending = [(r, h.base(entry,r,manifest_sha,'forecast',model,seed,split)) for r in rows]
    pending = [(r,b) for r,b in pending if not h.finished(output(model,seed,entry,split,h.key(r)),b)]
    if not pending:
        return {'written':0, 'existing':len(rows)}
    raw_parent = parent_prediction(model, seed, entry, split, None)
    if raw_parent is None:
        raise RuntimeError('Historical RAW prediction missing')
    cached, arrays = {}, {}
    for row, _ in pending:
        if row is not None and row['status'] != 'complete':
            raise RuntimeError('Cannot score incomplete compression: '+row['row_path'])
        rec = source if row is None else c.decoded(row)
        if rec.shape != source.shape or not np.isfinite(rec).all():
            raise RuntimeError('Invalid forecast reconstruction')
        arrays[h.key(row)] = rec[:,0,-96:]
        if row is not None:
            item = parent_prediction(model,seed,entry,split,row,rec)
            if item is not None:
                cached[h.key(row)] = item
    # Preserve the historical batch packing: RAW first, then unique pending panels.
    unique = [source[:,0,-96:]]; indexes = {'raw':0}; by_digest = {h.digest(unique[0]):0}
    for row, _ in pending:
        key = h.key(row)
        if key in cached or key == 'raw':
            continue
        x = arrays[key]; sha = h.digest(x)
        if sha not in by_digest:
            by_digest[sha] = len(unique); unique.append(x)
        elif not h.bits(x, unique[by_digest[sha]]):
            raise RuntimeError('Input digest collision')
        indexes[key] = by_digest[sha]
    merged = old.predict(model,net,np.concatenate(unique)); n = len(source)
    predictions = {key:{k:v[j*n:(j+1)*n] for k,v in merged.items()} for key,j in indexes.items()}
    raw = predictions['raw']; replay = old.predict(model,net,source[:,0,-96:])
    if set(raw) != set(replay) or set(raw) != set(raw_parent[0]):
        raise RuntimeError('RAW output schema changed')
    if not all(h.bits(raw[k],replay[k]) and h.bits(raw[k],raw_parent[0][k]) for k in raw):
        raise RuntimeError('RAW forecast does not exactly replay historical fixed consumer')
    replay_path = OUT/'raw_replays'/model/str(seed)/entry['sid']/(split+'.json')
    c.write(replay_path, dict(passed=True, parent=raw_parent[1],
                            rule='bitwise RAW alone, RAW with current companions, and historical RAW'), exclusive=True)
    predictions.update({k:v[0] for k,v in cached.items()})
    naive = np.repeat(source[:,0,-1:],24,axis=1)
    for row, result in pending:
        key = h.key(row)
        if key in cached:
            origin, lineage = 'historical_identical_input_after_raw_replay', cached[key][1]
        elif key == 'raw':
            origin, lineage = 'historical_raw_verified_by_fresh_replay', raw_parent[1]
        elif indexes[key] == 0:
            origin, lineage = 'current_identical_raw_input', {'raw_replay':h.binding(replay_path)}
        else:
            origin, lineage = 'fresh_fixed_consumer', {}
        result['consumer_input_sha256'] = h.digest(arrays[key])
        h.persist(output(model,seed,entry,split,key), result, predictions[key],
                  old.metric(future,predictions[key],raw,naive), origin, lineage)
    return dict(written=len(pending), historical_reuse=len(cached), unique_fresh_panels=len(unique),
                origins=n, raw_validation_calls=2)


def run(model, seed=0, split='val', dataset=None):
    if (model,seed) not in INSTANCES:
        raise ValueError('Unknown forecast model/seed')
    parent_manifest = c.PARENT_RUN/'evaluation/forecast/consumer_manifest.json'
    parent = c.read(parent_manifest); h.ensure_parent_sources(parent)
    _, manifest_sha = h.freeze('forecast',parent_manifest,[Path(old.__file__)])
    if split == 'test':
        for m,s in INSTANCES:
            status = OUT/'status'/f'{m}__{s}__val.json'
            if not status.exists() or not c.read(status).get('complete'):
                raise RuntimeError('All frozen forecast VAL instances must finish before TEST')
    lock = nullcontext() if model == 'ridge' else c.gpu_lock()
    with lock:
        runtime(model,seed); net = None; previous = None
        for entry in c.entries('forecast'):
            if dataset and entry['dataset'] != dataset:
                continue
            group = entry['sid'] if model == 'ridge' else entry['dataset'] if model in ('dlinear','patchtst') else 'global'
            if group != previous:
                net = None
                if model != 'ridge':
                    import torch
                    torch.cuda.empty_cache()
                net = old.load(model,seed,entry['dataset'],entry['sid']); previous = group
            report = evaluate_entry(model,seed,net,entry,split,manifest_sha)
            print(json.dumps(dict(model=model,seed=seed,sid=entry['sid'],split=split,**report)),flush=True)
    if dataset is None:
        c.write(OUT/'status'/f'{model}__{seed}__{split}.json',
                dict(complete=True,model=model,seed=seed,split=split,consumer_manifest_sha256=manifest_sha))
