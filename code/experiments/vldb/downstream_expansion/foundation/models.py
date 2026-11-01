"""Pinned, inference-only official models. No fitted wrappers or test selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np

QUANTILES = [i / 10 for i in range(1, 10)]
BATCH = 128


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class OfficialModel:
    def __init__(self, name, manifest):
        import torch
        self.name = name
        self.torch = torch
        self.manifest = manifest
        path = Path(manifest['local_path'])
        for fname, checksum in manifest['files_sha256'].items():
            if _sha(path / fname) != checksum:
                raise RuntimeError('Checkpoint content mismatch: ' + fname)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.manual_seed(20260922)
        torch.cuda.manual_seed_all(20260922)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if name in ('chronos2', 'chronos_bolt_small'):
            from chronos import BaseChronosPipeline
            self.model = BaseChronosPipeline.from_pretrained(
                str(path), device_map='cuda', torch_dtype=torch.float32,
                local_files_only=True)
            self.model.model.eval()
        elif name == 'timesfm3':
            from timesfm3 import TimesFM3Forecaster
            self.model = TimesFM3Forecaster.from_pretrained(
                str(path), device='cuda', per_core_batch_size=BATCH)
        else:
            raise ValueError(name)
        self.metadata = dict(name=name, model_id=manifest['model_id'],
            revision=manifest['revision'], precision='float32', device='cuda',
            torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), context_length=96, horizon=24,
            quantile_levels=QUANTILES, point_forecast='median (0.5 quantile)',
            batch_size=BATCH, seed_for_runtime=20260922, deterministic=True,
            tf32=False, fine_tuning=False, cross_learning=False,
            timesfm_flags=dict(use_symmetric_averaging=False, make_positive=False,
                sort_quantiles=True, use_znorm=False, padding_mode='none'),
            packages={p: importlib.metadata.version(p) for p in (
                'chronos-forecasting','timesfm','transformers','numpy',
                'huggingface-hub','safetensors','accelerate','einops')})

    def predict(self, values):
        torch = self.torch
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 96 or not np.isfinite(values).all():
            raise ValueError('Expected finite batch x 96 float32 contexts')
        qs = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(values), BATCH):
                x = np.ascontiguousarray(values[i:i+BATCH])
                if self.name == 'chronos2':
                    q, _ = self.model.predict_quantiles(
                        inputs=torch.from_numpy(x[:, None, :]),
                        prediction_length=24, quantile_levels=QUANTILES,
                        batch_size=BATCH, context_length=96, cross_learning=False)
                    result = torch.cat(q, dim=0).cpu().float().numpy()
                elif self.name == 'chronos_bolt_small':
                    q, _ = self.model.predict_quantiles(
                        inputs=torch.from_numpy(x), prediction_length=24,
                        quantile_levels=QUANTILES)
                    result = q.cpu().float().numpy()
                else:
                    outs = list(self.model.predict_batch(list(x), horizon=24,
                        return_quantiles=True, use_symmetric_averaging=False,
                        make_positive=False, sort_quantiles=True,
                        use_znorm=False, padding_mode='none'))
                    result = np.stack([v.quantiles for v in outs]).astype(np.float32)
                if result.shape != (len(x),24,9) or not np.isfinite(result).all():
                    raise RuntimeError(f'Invalid forecast shape/content: {result.shape}')
                qs.append(result)
        quantiles = np.concatenate(qs)
        torch.cuda.synchronize()
        return dict(point=quantiles[:, :, 4].copy(), quantiles=quantiles,
            quantile_levels=QUANTILES, seconds=time.perf_counter()-start)


def load_model(name, model_manifest):
    return OfficialModel(name, model_manifest)
