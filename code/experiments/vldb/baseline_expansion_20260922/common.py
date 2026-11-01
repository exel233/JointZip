from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/results/vldb_baseline_expansion_20260922_v1'
MAIN = ROOT / 'experiments/results/vldb_edbt_main_20260907_v1'
PATCH = ROOT / 'experiments/results/vldb_patchtst_full_20260915_v1'
FOUNDATION = ROOT / 'experiments/results/vldb_foundation_20260922_v1'
UCR = ROOT / 'experiments/results/vldb_ucr_classification_expansion_20260915_v1'
NASA = ROOT / 'experiments/results/vldb_tsb_nasa_downstream_20260915_v1'
SEEDS = [20260915,20260916,20260917]
POINTS = [dict(label='strict',epsilon=.1,rho=.999,tau=.01),dict(label='medium',epsilon=.25,rho=.995,tau=.02),dict(label='loose',epsilon=.5,rho=.99,tau=.05)]
STRESS = dict(label='stress',epsilon=1.,rho=.99,tau=.05)
LOSSLESS = ('zstd','fpzip','chimp128')
OLD_METHODS = ['qpet_point','qpet_pearson','qpet_acf','qpet_joint','sz2016','sz3','sperr','zfp','cameo','zstd']
EXTRA_METHODS = ['serf','lfzip','qoz','fpzip','chimp128']
PRIOR_METHODS = ['speqc_er','speqc_era','optz_er','optz_era']
METHODS = OLD_METHODS + EXTRA_METHODS + PRIOR_METHODS
UCR_OLD_MAP = {'qpet_point':'point_only','qpet_pearson':'pearson','qpet_acf':'acf','qpet_joint':'joint','sperr':'sperr','sz2016':'sz2016'}
PROTOCOL = dict(
 schema='nclab.baseline-expansion.v1',methods=METHODS,points=POINTS,ucr_point=STRESS,
 selection='Reuse all previously declared six-source 142-series, six UCR sets, and 24 NASA Eval files; no result-based sample selection.',
 input='TRAIN-standardized float32. Forecast protection128/consumer96/horizon24; UCR whole query96-512; NASA independent128 blocks with true tail length billed.',
 metrics='Independent float64 max absolute error, per-window original-reconstructed Pearson, max slice-Pearson ACF gap lags1-16. Undefined and constant cases explicitly counted.',
 contracts='Report zero-tolerance and numerical-tolerance outcomes separately. CAMEO original mean-ACF contract is not the Joint max-lag contract. Lossless verified bitwise.',
 payload='Complete independently decodable SCS2 with channel, normalizer and provenance. NASA adds actual 16-byte TSL1 true-length header; raw numerator excludes padded samples.',
 consumers='Reuse frozen raw-trained Ridge,DLinear,PatchTST(3seeds),3foundation models,classification1NN/Logistic/ResNetv1,v2(3seeds),shape/deep retrieval,NASA Ridge/KNN/LSTM(3seeds). No model fitting on reconstructed TEST.',
 quality_CR=dict(scope='Forecasting only',budgets=[.05,.10,.20],primary=.10,candidates='fixed three configurations per named lossy arm, plus common Zstd fallback; lossless arm plus Zstd',
    unit='series x consumer; average per-seed MSE before selecting one shared codec configuration',
    criterion='VAL MSE <= (1+budget)*max(raw VAL MSE,1e-12)',
    objective='minimum actual complete VAL bytes, ties stricter epsilon then fixed candidate order',
    test='Selected parameters unchanged; verify TEST MSE <= (1+budget)*max(raw TEST MSE,1e-12); report all selections, fallback counts, test pass rates and common-pass paired bytes/CR.',
    order='Persist immutable selections from all VAL candidates before evaluating any new TEST task scores.',
    limitations='Eight inherited disjoint VAL origins per series; also used for original consumer selection. UCRv2 has no independent VAL and NASA VAL has no anomaly labels, so no accuracy/AP constrained-rate claim there.'),
 fairness='Fixed epsilon comparisons disclose achieved E/R/A and bytes; dual/triple contract comparisons separated; no universal superiority assumption.',
 uncertainty='Dataset/source macro first; repeated seeds and windows are not independent datasets. Paired descriptive distributions and positive/negative cases retained.',
 provenance='Official libraries identified by revision/hash; local SPEQC-inspired QS-ABS and any wrappers explicitly labelled; no attribution of adapted implementation as official paper reproduction.',
 timing='Concurrent execution wall times are diagnostic; no codec-speed ranking from these runs.',
)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=canonical(value)
    if exclusive and path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Immutable file differs: '+str(path))
        return
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');tmp.write_bytes(data);tmp.replace(path)
def frozen():
    d=read(RUN/'freeze.json')
    if d['protocol']!=PROTOCOL:raise RuntimeError('Protocol changed; use a new run')
    return d
def entries(task=None):return [s for s in frozen()['entries'] if task is None or s['task']==task]
def arrays(entry):
    p=RUN/entry['input']
    if sha(p)!=entry['input_sha256']:raise RuntimeError('Input changed '+entry['sid'])
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def points(entry,method):
    if method in LOSSLESS:return [dict(label='lossless',epsilon=0.,rho=1.,tau=0.)]
    return [STRESS] if entry['task']=='ucr' else POINTS
def splits(entry):return ['val','test'] if entry['task']=='forecast' else ['test']
def row_path(entry,split,method,point):return RUN/'compression'/entry['task']/entry['sid']/split/(method+'__'+point+'.json')
def rows(entry,split,require_complete=True):
    out=[]
    for method in METHODS:
        for point in points(entry,method):
            p=row_path(entry,split,method,point['label'])
            if not p.exists():
                if require_complete:raise RuntimeError('Compression row missing: '+str(p))
                continue
            r=read(p);r['row_path']=str(p.relative_to(RUN));out.append(r)
    return out
def decoded(row):
    p=RUN/row['reconstruction']
    if sha(p)!=row['reconstruction_sha256']:raise RuntimeError('Reconstruction changed')
    return np.load(p,allow_pickle=False)
@contextmanager
def gpu_lock():
    import fcntl
    with open('/tmp/nclab_downstream_gpu_20260915.lock','a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
