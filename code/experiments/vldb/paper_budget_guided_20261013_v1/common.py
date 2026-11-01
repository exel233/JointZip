from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.vldb.simple_correction_20260929 import common as legacy

ROOT = Path(__file__).resolve().parents[3]
BASE_RUN = ROOT / 'experiments/results/vldb_paper_budget_guided_20261013_v1'
STRATEGIES = ('grid', 'formula', 'union')
STRATEGY = os.environ.get('JOINTZIP_GUIDED_STRATEGY', 'formula')
if STRATEGY not in STRATEGIES:
    raise ValueError('Unknown JOINTZIP_GUIDED_STRATEGY: ' + STRATEGY)
RUN = BASE_RUN / 'strategies' / STRATEGY
PARENT = PARENT_RUN = legacy.PARENT_RUN
MAIN, PATCH, FOUNDATION, UCR, NASA = legacy.MAIN, legacy.PATCH, legacy.FOUNDATION, legacy.UCR, legacy.NASA
OLD_SIMPLE_RUN = ROOT / 'experiments/results/vldb_simple_correction_20260929_v1'
OLD_BUDGET_RUN = ROOT / 'experiments/results/vldb_paper_budget_refined_20261012_v2'
SEEDS, POINTS, STRESS = deepcopy(legacy.SEEDS), deepcopy(legacy.POINTS), deepcopy(legacy.STRESS)
CANONICAL_METHOD = {
    'qpet_guided_point': 'qpet_point', 'qpet_guided_pearson': 'qpet_pearson',
    'qpet_guided_acf': 'qpet_acf', 'qpet_guided_joint': 'qpet_joint',
}
BUDGET_METHODS = SIMPLE_METHODS = list(CANONICAL_METHOD)
BASELINE_METHODS = ['zstd']
METHODS = SIMPLE_METHODS + BASELINE_METHODS
LOSSLESS = ('zstd',)
OLD_SIMPLE_METHOD = {m: m.replace('qpet_guided_', 'qpet_simple_') for m in SIMPLE_METHODS}
METHOD_VERSION = 'formula_guided_scalar_legacy_cf6_v1'
FORMULA_BUDGET_POLICY = 'actual_base_zero_slack'
SIMPLE_SETTINGS = deepcopy(legacy.SIMPLE_SETTINGS)
SIMPLE_SETTINGS['candidate_profile'] = 'paper_budget_guided'

# This description is finalized with the native policy before production freeze.
PARAMETER_POLICY = dict(
    scope='Window scalar error parameters; no pointwise map and no new codec or correction.',
    formula='Original Pearson geometric and QPET deterministic/probabilistic ACF estimates.',
    objective='Actual base plus unchanged simple-correction bytes; no downstream score is used.',
    source='GuidedExperiment.inc is the authoritative frozen candidate schedule.',
    budgets=dict(sperr=22, serf_xor=8, serf_qt=8, lossless=1),
    initial_counts=dict(sperr=12, serf_xor=5, serf_qt=5),
    anchor='a=min(epsilon,b_R,b_A); a=0 requires exact candidate; disabled statistics omitted.',
    interval='L=min(a/4,epsilon/512) for SPERR; min(a/4,epsilon/64) for Serf. U=epsilon.',
    initial='Clipped/deduplicated L,a/2,a,2a,U; fill by geometric midpoints of widest log gaps.',
    refinement='Bounded log-interval refinement driven only by actual base+legacy-correction bytes, within each backend.',
    control='Same per-backend attempt limits and refinement; anchor uses geometric midpoint of fixed relative-error range and does not read source Pearson/ACF sensitivities.',
    limits='Failures count against attempts. Shared physical encodings cannot expose unproposed grid costs to formula search. Formula candidates may relax the anchor up to epsilon; exact final validation remains mandatory.',
    guarantee='Proposals are not nonlinear ACF guarantees; actual decode, repair and E/R/A validation remain mandatory.',
)
PROTOCOL = deepcopy(legacy.PROTOCOL)
PROTOCOL.update(
    schema='nclab.formula-guided-scalar.v1', strategy=STRATEGY, methods=METHODS,
    method_version=METHOD_VERSION, simple_settings=SIMPLE_SETTINGS,
    parameter_policy=PARAMETER_POLICY,
    selection='All original 172 inputs and thresholds; compare grid, pure formula and union in separate consumer roots.',
    reuse='Historical results remain unchanged. New grid/formula/union streams are independently decoded; common Zstd baseline retains explicit lineage. Consumer cache reuse requires original exact-input/runtime/Raw replay checks.',
    development_status='User-directed follow-up after inspecting prior experiments; not an unseen confirmatory benchmark.',
    timing='Independent standalone fresh grid/formula/union calls on a 30-condition paired panel; all-run shared-search wall time is diagnostic only.',
    policy_comparison='Union contains original grid and formula candidates; pure formula has no old-grid artifact fallback. Common lossless candidate is reported separately.',
)
PROTOCOL['quality_CR']['candidates'] = 'Each strategy: fixed three threshold settings per arm and common Zstd fallback; VAL selection only.'

def now():
    return datetime.now(timezone.utc).isoformat()

@lru_cache(maxsize=32768)
def _sha_stat(path, size, modified_ns):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def sha(path):
    path = Path(path)
    stat = path.stat()
    return _sha_stat(str(path.resolve()), stat.st_size, stat.st_mtime_ns)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()

def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    if exclusive and path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Immutable record differs: '+str(path))
        return
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)

def frozen():
    value = read(RUN/'freeze.json')
    if value['protocol'] != PROTOCOL:
        raise RuntimeError('Frozen guided protocol changed; use a new run')
    return value

def entries(task=None):
    return [entry for entry in frozen()['entries'] if task is None or entry['task'] == task]

def arrays(entry):
    path = RUN/entry['input']
    if sha(path) != entry['input_sha256']:
        raise RuntimeError('Frozen input changed: '+entry['sid'])
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}

def points(entry, method):
    if method in LOSSLESS:
        return [dict(label='lossless', epsilon=0., rho=1., tau=0.)]
    return [STRESS] if entry['task'] == 'ucr' else POINTS

def splits(entry):
    return ['val', 'test'] if entry['task'] == 'forecast' else ['test']

def row_path(entry, split, method, point):
    return RUN/'compression'/entry['task']/entry['sid']/split/(method+'__'+point+'.json')

def rows(entry, split, require_complete=True):
    result=[]
    for method in METHODS:
        for point in points(entry, method):
            path=row_path(entry, split, method, point['label'])
            if not path.exists():
                if require_complete:
                    raise RuntimeError('Missing compression row: '+str(path))
                continue
            row=read(path)
            row['row_path']=str(path.relative_to(RUN))
            result.append(row)
    return result

def decoded(row):
    path=RUN/row['reconstruction']
    if sha(path) != row['reconstruction_sha256']:
        raise RuntimeError('Saved reconstruction changed')
    return np.load(path, allow_pickle=False)

@contextmanager
def gpu_lock():
    import fcntl
    with open('/tmp/nclab_downstream_gpu_20260915.lock', 'a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
