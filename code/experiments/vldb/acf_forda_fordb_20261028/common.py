from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/results/vldb_acf_forda_fordb_20261028_v1'
WORK = Path('/opt/jointzip-home/.cache/vldb_acf_forda_fordb_20261028_v1')
DATASETS = ('FordA', 'FordB')
BACKENDS = ('sz3', 'zfp')
ARMS = {'ER': -1., 'ERA_tau005': .05, 'ERA_tau001': .01}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def setup_runtime():
    sys.path.insert(0, str(RUN / 'deps'))
    sys.path.insert(0, str(ROOT / 'tmp/vldb_main_20260907_python_deps'))
    sys.path.insert(0, str(ROOT))
    base = Path('/opt/jointzip-home/.local/cth_expansion_runs')
    prior = base / 'vldb_framework_ablation_20261017_v1/runtime'
    full = base / 'vldb_framework_full_20261015_v1/runtime'
    os.environ.update(json.loads((prior / 'ENVIRONMENT.json').read_text()))
    os.environ.update(MODERN_CODECS_PREFIX=str(full / 'modern-codecs'),
                      JOINTZIP_FULL_BRIDGE=str(full / 'fixed_backend'),
                      JOINTZIP_FULL_BRIDGE_MANIFEST=str(full / 'manifest.json'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    WORK.mkdir(parents=True, exist_ok=True)

