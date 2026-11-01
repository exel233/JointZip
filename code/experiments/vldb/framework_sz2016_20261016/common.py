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
NAME = os.environ.get('JOINTZIP_SZ2016_RUN_NAME', 'vldb_framework_sz2016_20261016_v1')
RUN = ROOT/'experiments/results'/NAME
PARENT_RUN = old.PARENT_RUN
OLD_SIMPLE = ROOT/'experiments/results/vldb_simple_correction_20260929_v1'
MAIN, PATCH, FOUNDATION, UCR, NASA = old.MAIN, old.PATCH, old.FOUNDATION, old.UCR, old.NASA
SEEDS, POINTS, STRESS = deepcopy(old.SEEDS), deepcopy(old.POINTS), deepcopy(old.STRESS)
BACKENDS = ('sz2016',)
METHODS = tuple(b+'_'+p for b in BACKENDS for p in ('native','jointzip'))
PROTOCOL = {
 'schema':'jointzip.fixed-backend-sz2016.v1',
 'scope':'All original 172 inputs; SZ2016 native versus JointZip; 930 conditions per method, 1860 total.',
 'backends':list(BACKENDS), 'methods':list(METHODS), 'points':POINTS, 'ucr_point':STRESS,
 'constraints':'E=max absolute error, R=original/reconstructed window Pearson, A=max slice-Pearson ACF gap at lags 1..16; report strict and 64*float64-eps audits.',
 'policy':'Frozen Oct13/14 formula scalar proposals; SZ2016 22 attempts with 12 initial; byte-guided refinement; no old grid fallback.',
 'formula_role':'Source-dependent proposal anchors; permitted relaxation to epsilon; final decode/simple repair/validation provides the contract. No deterministic nonlinear ACF guarantee from proposals.',
 'correction':'Original posthoc greedy, legacy optimizer, original values plus 7/15/31/63/127 targets; formats0-6; no later complex coding.',
 'native':'Same official SZ2016 backend/mode/precision and epsilon as the historical standalone baseline, one native encode per window, no statistical repair.',
 'container':'Identical JZF1 complete wrapper for both policies: channel, normalizer, shape, true flattened count, JZP1 independently decoded records; all bytes billed. NASA padded suffix excluded from raw numerator.',
 'reuse':'Historical SZ2016 native payloads reused and independently decoded after reframing. Exact-input/runtime checked downstream reuse. Never transplant old CR or encoding speed.',
 'consumers':'Frozen Raw-trained Ridge,DLinear,PatchTST3seeds,Chronos2,ChronosBoltSmall,TimesFM3; UCR1NN,Logistic,ResNetv1/v2,shape/deep retrieval; NASA Ridge/KNN/LSTM3seeds.',
 'selection':'Fixed original thresholds; all forecast VAL before TEST. No policy or sample selection from new TEST. Report all planned inputs, thresholds and consumers, including declines.',
 'density':'No new density sweep. The completed five-backend parent has a separate SZ3/ZFP VAL-only formula/noformula22/44/88 diagnostic; it is not evidence specific to SZ2016.',
 'timing':'Six forecast sources, first four VAL windows, strict/loose; 12 native/framework pairs, one warmup plus three measured calls per cell, 96 calls. Concurrent compression and reused rows are not fresh codec speed.',
 'dependency':'Start only after vldb_framework_full_20261015_v1 scientific, density, timing, closeout and Windows export gates all pass.',
 'development_status':'Exploratory extension with a backend chosen after inspecting earlier experiments, not a wholly unseen confirmatory benchmark. All outcomes retained.',
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
