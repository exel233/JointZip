"""Download official safe-tensor checkpoints at fixed revisions; record provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'experiments/results/vldb_foundation_20260922_v1'
MODELS = {
    'chronos2': ('amazon/chronos-2', '29ec3766d36d6f73f0696f85560a422f50e8498c'),
    'chronos_bolt_small': ('amazon/chronos-bolt-small', '772f3d25d38aec6d914c8949dab4462e2d46f5d8'),
    'timesfm3': ('google/timesfm-3.0-pytorch', '43046b85ec22d584a13f8098c2ed39c889e129c2'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from huggingface_hub import snapshot_download
    RUN.mkdir(parents=True, exist_ok=True)
    dest = RUN / 'models.json'
    if dest.exists():
        raise RuntimeError('Existing model manifest must not be silently overwritten')
    out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': {},
           'packages': {k: importlib.metadata.version(k) for k in (
               'torch', 'numpy', 'chronos-forecasting', 'timesfm', 'transformers',
               'huggingface-hub', 'safetensors', 'accelerate', 'einops')}}
    for name, (repo, rev) in MODELS.items():
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=[
            '*.json', '*.safetensors', 'README.md', 'LICENSE'], max_workers=2))
        out['models'][name] = {'model_id': repo, 'revision': rev,
            'local_path': str(path),
            'files_sha256': {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()},
            'point_forecast': '0.5 quantile (median)',
            'context_length': 96, 'prediction_length': 24,
            'cross_learning': False, 'fine_tuning': False,
            'weight_license': 'timesfm-non-commercial-license-v1.0' if name == 'timesfm3' else 'Apache-2.0'}
        (RUN / f'model_download_{name}.json').write_text(json.dumps(out['models'][name], indent=2)+'\n')
        print(json.dumps({'downloaded': name, 'path': str(path)}), flush=True)
    dest.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
