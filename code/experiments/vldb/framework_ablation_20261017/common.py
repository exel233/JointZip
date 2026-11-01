"""Frozen fixed-backend experiment shared data and persistence."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.vldb.simple_correction_20260929 import common as old

ROOT = Path(__file__).resolve().parents[3]
NAME = os.environ.get('JOINTZIP_ABLATION_RUN_NAME', 'vldb_framework_ablation_20261017_v1')
RUN = ROOT/'experiments/results'/NAME
PARENT_RUN = old.PARENT_RUN
OLD_SIMPLE = ROOT/'experiments/results/vldb_simple_correction_20260929_v1'
MAIN, PATCH, FOUNDATION, UCR, NASA = old.MAIN, old.PATCH, old.FOUNDATION, old.UCR, old.NASA
SEEDS, POINTS, STRESS = deepcopy(old.SEEDS), deepcopy(old.POINTS), deepcopy(old.STRESS)
BACKENDS = ('sz3','zfp','sperr','serf_xor','serf_qt','sz2016')
ARMS = ('E','ER','EA','ERA','ERA_no_correction')
METHODS = tuple(b+'_'+a for b in BACKENDS for a in ARMS)
HISTORY_FULL = ROOT/'experiments/results/vldb_framework_full_20261015_v1'
HISTORY_SZ = ROOT/'experiments/results/vldb_framework_sz2016_20261016_v1'
def history_run(backend): return HISTORY_SZ if backend=='sz2016' else HISTORY_FULL
def arm_constraints(point,arm):
    if arm not in ARMS: raise ValueError('Unknown ablation arm')
    return dict(epsilon=point['epsilon'], rho=point['rho'] if 'R' in arm else -1.,
                tau=point['tau'] if 'A' in arm else -1.)
def contract(arm): return 'ERA' if arm=='ERA_no_correction' else arm
PROTOCOL = {
 'schema':'jointzip.framework-ablation.v1',
 'scope':'Same 172 inputs, six fixed backends, five arms; 27900 compression rows including 5580 reused ERA rows.',
 'backends':list(BACKENDS),'arms':list(ARMS),'methods':list(METHODS),'points':POINTS,'ucr_point':STRESS,
 'constraints':'E=max absolute error, R=source/reconstructed Pearson, A=max slice-Pearson ACF gap lags1..16. Disabled R/A is -1 for anchor, repair, final enabled audit; all ERA metrics still recorded.',
 'policy':'Default formula scalar proposals 22/12/divisor512 for SZ3,ZFP,SPERR1D,SZ2016; 8/5/64 for Serf. No grid, density changes, TEST selection or complex coding.',
 'correction':'Original posthoc greedy, legacy optimizer, original plus 7/15/31/63/127 targets, formats0-6.',
 'correction_ablation':'Freeze existing ERA per-window ordered logical cap trace, including duplicate effective ZFP caps. Disable correction only, audit base E/R/A plus original cap eligibility, retain same exact fallback and byte/cap tie rule. Conditional on the full framework candidate pool, not an independently adapted no-repair search.',
 'container':'Unchanged JZF1/JZP1, complete bytes including normalizer/channel/shape/true count and repairs. Unpadded Float32 numerator.',
 'reuse':'Existing ERA full streams/reconstructions/metadata and matching downstream arrays reused. Native winner base payloads and matching candidate bases reused; unselected unstored bases computed once. Cross-arm native cache excludes repair thresholds, includes exact source/backend/effective native parameter.',
 'consumers':'Same frozen Raw-trained models, preprocessing, batches, seeds and runtime; no training; all forecast VAL before TEST. Exact-input historical and current-panel reuse.',
 'timing':'No new end-to-end speed ranking from parallel or cached work. Correction-specific fixed-pool replay costs are diagnostics only, not autonomous algorithm encoding throughput.',
 'selection':'All original thresholds, datasets, seeds and tasks retained; report improvements, ties, declines, lossless fallback and repaired bytes. Current data were seen during method development.',
 'expected':dict(inputs=172,compression=27900,consumers=220462,reused_era=5580),
}

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value): return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def write(path,value,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); data=canonical(value)
    if exclusive and path.exists():
        if path.read_bytes()!=data: raise RuntimeError('Immutable result differs: '+str(path))
        return
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp'); temp.write_bytes(data); temp.replace(path)
@lru_cache(maxsize=65536)
def _sha(path,size,mtime):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def sha(path):
    p=Path(path); s=p.stat(); return _sha(str(p.resolve()),s.st_size,s.st_mtime_ns)
@lru_cache(maxsize=1)
def frozen():
    result=read(RUN/'freeze.json')
    if result['protocol']!=PROTOCOL: raise RuntimeError('Frozen protocol differs')
    return result
def entries(task=None): return [x for x in frozen()['entries'] if task is None or x['task']==task]
def arrays(entry):
    p=RUN/entry['input']
    if sha(p)!=entry['input_sha256']: raise RuntimeError('Frozen input changed: '+entry['sid'])
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def points(entry,method=None): return [STRESS] if entry['task']=='ucr' else POINTS
def splits(entry): return ['val','test'] if entry['task']=='forecast' else ['test']
def row_path(entry,split,method,point): return RUN/'compression'/entry['task']/entry['sid']/split/(method+'__'+point+'.json')
def rows(entry,split,require_complete=True):
    out=[]
    for method in METHODS:
        for point in points(entry,method):
            p=row_path(entry,split,method,point['label'])
            if not p.exists():
                if require_complete: raise RuntimeError('Missing compression row: '+str(p))
                continue
            r=read(p)
            if r['status']!='complete': raise RuntimeError('Incomplete committed compression row')
            r['row_path']=str(p.relative_to(RUN)); out.append(r)
    return out
def decoded(row):
    p=RUN/row['reconstruction']
    if sha(p)!=row['reconstruction_sha256']: raise RuntimeError('Saved reconstruction changed')
    return np.load(p,allow_pickle=False)
@contextmanager
def gpu_lock():
    import fcntl
    with open('/tmp/nclab_downstream_gpu_20260915.lock','a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
