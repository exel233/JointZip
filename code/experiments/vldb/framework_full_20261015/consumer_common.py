"""Small, explicit persistence/reuse helpers for the fixed-backend consumers.

No training, codec search, or changes to the historical experiment modules.
"""
from __future__ import annotations
import hashlib
import os
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
        method = row['method'].removesuffix('_native').removesuffix('_jointzip')
    if method == 'serf_xor':
        method = 'serf'
    if method not in ('sz3','zfp','sperr','serf','serf_qt','zstd'):
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
