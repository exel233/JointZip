"""Shared, window-level acceptance used independently of each encoder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_RUN = ROOT / 'experiments/results/vldb_edbt_main_20260907_v1'
DEFAULT_RUN = ROOT / 'experiments/results/vldb_prior_comparison_20260911_pilot_v2'
POINTS = [dict(label='strict', epsilon=.1, rho=.999, tau=.01),
          dict(label='medium', epsilon=.25, rho=.995, tau=.02),
          dict(label='loose', epsilon=.5, rho=.99, tau=.05)]
DATASETS = ['ETTm1', 'ETTm2', 'Solar', 'Weather', 'ElecDem', 'Wind']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def canonical(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()

def write_json(path, x):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_bytes(canonical(x))
    temp.replace(p)

def _corr(x, y):
    xc = x-x.mean(axis=-1, keepdims=True)
    yc = y-y.mean(axis=-1, keepdims=True)
    xx = np.sum(xc*xc, axis=-1)
    yy = np.sum(yc*yc, axis=-1)
    den = np.sqrt(xx*yy)
    out = np.zeros_like(den)
    np.divide(np.sum(xc*yc, axis=-1), den, out=out, where=den>1e-15)
    return np.clip(out, -1, 1), xx, den

def audit(source, reconstruction, epsilon, rho, tau, scope='triple'):
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(reconstruction, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('audit requires equal finite [W,C,T] inputs')
    error = np.max(np.abs(x-y), axis=-1)
    corr, energy, _ = _corr(x, y)
    corr = np.where(energy <= 1e-15, np.all(x == y, axis=-1).astype(float), corr)
    acf_errors, degenerate = [], 0
    for lag in range(1, min(16, x.shape[-1]-2)+1):
        a, _, den_a = _corr(x[..., :-lag], x[..., lag:])
        b, _, den_b = _corr(y[..., :-lag], y[..., lag:])
        acf_errors.append(np.abs(a-b))
        degenerate += int(np.count_nonzero((den_a<=1e-15)|(den_b<=1e-15)))
    ae = np.max(acf_errors, axis=0) if acf_errors else np.zeros_like(corr)
    ep, rp, ap = error<=epsilon, corr>=rho, ae<=tau
    enabled = ep & rp & (ap if scope=='triple' else True)
    return dict(max_abs_error=float(error.max()), pearson_min=float(corr.min()),
                acf_max_abs=float(ae.max()), mse=float(np.mean((x-y)**2)),
                series_windows=int(corr.size), point_pass_count=int(ep.sum()),
                pearson_pass_count=int(rp.sum()), acf_pass_count=int(ap.sum()),
                enabled_pass_count=int(enabled.sum()), joint_pass_count=int((ep&rp&ap).sum()),
                enabled_pass=bool(enabled.all()), joint_pass=bool((ep&rp&ap).all()),
                source_degenerate_windows=int((energy<=1e-15).sum()),
                degenerate_acf_pairs=degenerate,
                tolerance=0.0, scope=scope)
