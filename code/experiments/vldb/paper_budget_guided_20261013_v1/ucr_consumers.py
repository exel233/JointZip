"""Fresh UCR scoring for one frozen guided strategy, without model fitting.

All four constraint arms and the shared lossless control are scored anew. Models,
preprocessing, device, batches and inputs match the parent expansion manifest.
Fresh raw controls must reproduce its saved outputs before any codec is scored.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import common as c
from experiments.vldb.downstream_expansion.classification import evaluate as cm
from experiments.vldb.downstream_expansion.retrieval import run as rm

HERE = Path(__file__).resolve()
OUT = c.RUN / 'evaluation/ucr'
MANIFEST = OUT / 'consumer_manifest.json'
PARENT_MANIFEST = c.PARENT_RUN / 'evaluation/ucr/consumer_manifest.json'
V2 = c.ROOT / 'experiments/results/vldb_ucr_classification_standard_protocol_v2_20260915'
RETRIEVAL = c.ROOT / 'experiments/results/vldb_ucr_retrieval_20260915_v1'
ARCH = c.ROOT / 'experiments/vldb/downstream_expansion/classification/models.py'
CONSUMERS = ['one_nn', 'logistic', 'resnet_v1', 'resnet_v2']
RETRIEVAL_FIELDS = list(rm.FIELDS)
CLASS_FIELDS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'cross_entropy',
                'raw_prediction_agreement', 'raw_correct_to_wrong', 'raw_wrong_to_correct']


def relative(path):
    return str(Path(path).relative_to(c.ROOT)).replace('\\', '/')


def configure(device):
    import torch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen UCR CUDA device unavailable')
        torch.cuda.manual_seed_all(20260922)
    return dict(device=device, torch=torch.__version__, numpy=np.__version__,
                sklearn=importlib.metadata.version('scikit-learn'),
                python=platform.python_version(), batch=64, threads=1,
                deterministic=True, tf32=False,
                gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)


def model_folder(name, version, seed):
    return (c.UCR if version == 'resnet_v1' else V2) / 'resnet' / name / str(seed)


def parent_protocol():
    """Require the same already-frozen consumer, not a new hardware choice."""
    value = c.read(PARENT_MANIFEST)
    if value['expansion_freeze_sha256'] != c.sha(c.PARENT_RUN / 'freeze.json'):
        raise RuntimeError('Parent UCR run freeze changed')
    for path, digest in value['sources'].items():
        if c.sha(c.ROOT / path) != digest:
            raise RuntimeError('Parent frozen UCR dependency changed: ' + path)
    current = {entry['sid']: entry['input_sha256'] for entry in c.entries('ucr')}
    previous = {entry['sid']: entry['input_sha256'] for entry in value['input_entries']}
    if current != previous:
        raise RuntimeError('Simple-correction UCR inputs differ from parent')
    if value['seeds'] != c.SEEDS:
        raise RuntimeError('Simple-correction UCR seeds differ from parent')
    canonical = [c.CANONICAL_METHOD.get(method, method) for method in c.METHODS]
    if len(canonical) != len(set(canonical)) or not set(canonical) <= set(value['methods']):
        raise RuntimeError('Guided UCR method roster is not a unique subset of the frozen parent roster')
    return value


def freeze(device=None):
    parent = parent_protocol()
    device = parent['runtime']['device'] if device is None else device
    if device != parent['runtime']['device']:
        raise RuntimeError('Requested UCR device differs from parent frozen device')
    runtime = configure(device)
    if runtime != parent['runtime']:
        raise RuntimeError('Simple-correction UCR runtime differs from parent')
    if MANIFEST.exists():
        value = verify()
        if value['runtime']['device'] != device:
            raise RuntimeError('Requested device differs from frozen UCR consumer')
        return value
    c.frozen()
    files = [HERE, HERE.with_name('common.py'), PARENT_MANIFEST,
             c.PARENT_RUN / 'freeze.json', Path(cm.__file__), Path(rm.__file__),
             ARCH, Path(rm.__file__).with_name('resnet_snapshot.py'),
             c.UCR / 'freeze.json', c.UCR / 'model_protocol.json', V2 / 'protocol.json',
             RETRIEVAL / 'protocol.json']
    models = []
    for entry in c.entries('ucr'):
        name = entry['sid']
        for task, consumers in (
                ('classification', CONSUMERS),
                ('retrieval', ['shape', 'resnet_v1', 'resnet_v2'])):
            for consumer in consumers:
                for seed in (c.SEEDS if consumer.startswith('resnet') else [0]):
                    raw_row_path = (c.PARENT_RUN / 'evaluation/ucr' / task
                                    / consumer / name / str(seed) / 'raw.json')
                    raw_row = c.read(raw_row_path)
                    raw_values = c.PARENT_RUN / raw_row['per_query_file']
                    if (raw_row['status'] != 'complete'
                            or raw_row['consumer_manifest_sha256'] != c.sha(PARENT_MANIFEST)
                            or c.sha(raw_values) != raw_row['per_query_sha256']):
                        raise RuntimeError('Parent UCR raw output identity differs')
                    files += [raw_row_path, raw_values]
        linear = c.UCR / 'baselines' / name
        files += [linear / 'linear_model.npz', linear / 'linear_predictions.npz',
                  linear / 'one_nn_predictions.npz']
        for version in ('resnet_v1', 'resnet_v2'):
            for seed in c.SEEDS:
                folder = model_folder(name, version, seed)
                checkpoint = folder / 'model.pt'
                evaluation = c.read(folder / 'evaluation.json')
                if c.sha(checkpoint) != evaluation['checkpoint_sha256']:
                    raise RuntimeError('Historical checkpoint identity differs')
                files += [checkpoint, folder / 'training.json', folder / 'evaluation.json',
                          folder / 'predictions.npz']
                files += [RETRIEVAL / 'rows' / version / name / str(seed) / 'raw.npz']
                models.append(dict(dataset=name, consumer=version, seed=seed,
                                   checkpoint=relative(checkpoint), checkpoint_sha256=c.sha(checkpoint)))
        files += [RETRIEVAL / 'rows/shape' / name / '0/raw.npz']
    value = dict(schema='nclab.paper-budget-guided.ucr-consumers.v1',
        expansion_freeze_sha256=c.sha(c.RUN / 'freeze.json'), created_utc=c.now(),
        runtime=runtime, methods=c.METHODS, seeds=c.SEEDS,
        parent_consumer_manifest=relative(PARENT_MANIFEST),
        parent_consumer_manifest_sha256=c.sha(PARENT_MANIFEST),
        parent_compression_freeze_sha256=c.sha(c.PARENT_RUN / 'freeze.json'),
        evaluation_policy='Fresh scoring of raw plus all declared arms for this strategy; no parent metric/per-query output is copied.',
        strategy=c.STRATEGY,
        parent_raw_policy='Require bitwise-identical raw probabilities/predictions/rankings on the same frozen device before scoring reconstructed queries.',
        sources={relative(p): c.sha(p) for p in sorted(set(files))}, models=models,
        input_entries=[dict(sid=e['sid'], input=e['input'], input_sha256=e['input_sha256'])
                       for e in c.entries('ucr')],
        classification=dict(one_nn='Old float64 centered unit-L2 2-2dot shape distance; first TRAIN index wins exact ties. Degenerate query maps to zero vector.',
            logistic='Restore numeric sklearn coefficients/classes/mean/std; never fit. Use official sklearn predict_proba.',
            resnet='Same frozen local PyTorch port checkpoints as historical v1/v2, not official TensorFlow execution.',
            normalization='v1 checkpoint global mean/std in float32; v2 per-query float64 population z-normalization then float32, only zero std replaced by 1.',
            labels='Original official TRAIN/TEST labels unchanged; all TEST queries scored.'),
        retrieval=dict(reference='Complete original TRAIN only; query compressed, reference/model/labels unbilled.',
            features='Shape centered unit-L2; deep GAP immediately before classifier, no feature normalization.',
            ranking='Float64 squared Euclidean distance, clipped at zero, stable original TRAIN order for exact ties.',
            relevance='Identical original class labels; classifier representations were supervised on TRAIN.',
            metrics='Full-ranking AP; binary nDCG@10; Recall@10 denominator all relevant TRAIN items; raw-ranking stability separate.',
            scope='Derived label-relevance retrieval, not Matrix Profile or an independent representation benchmark.'),
        protocol='Fixed original epsilon/rho/tau stress point; no task-quality parameter selection on UCR. v1/v2 reported separately, no best-of-version.',
        evidence='Input/codec/model failures retained explicitly. Consumer manifest frozen before new TEST predictions. Input hashes registered before first scoring.',
        device_note='Parent expansion device/package/batch identity is retained for every fresh raw/codec call. September-15 historical diagnostics are reported separately; no historical task scores are inserted into this matrix.')
    c.write(MANIFEST, value, exclusive=True)
    return value


def verify():
    value = c.read(MANIFEST)
    if value['expansion_freeze_sha256'] != c.sha(c.RUN / 'freeze.json'):
        raise RuntimeError('UCR expansion freeze differs')
    parent = parent_protocol()
    if (value['parent_consumer_manifest_sha256'] != c.sha(PARENT_MANIFEST)
            or value['parent_compression_freeze_sha256'] != c.sha(c.PARENT_RUN / 'freeze.json')
            or value['runtime'] != parent['runtime']
            or value['methods'] != c.METHODS):
        raise RuntimeError('Frozen UCR parent or method identity changed')
    for path, digest in value['sources'].items():
        if c.sha(c.ROOT / path) != digest:
            raise RuntimeError('Frozen UCR dependency changed: ' + path)
    for entry in value['input_entries']:
        if c.sha(c.RUN / entry['input']) != entry['input_sha256']:
            raise RuntimeError('Frozen UCR input changed: ' + entry['sid'])
    return value


def output_file(task, consumer, name, seed, key, suffix='.json'):
    return OUT / task / consumer / name / str(seed) / (key + suffix)


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def register_rows(entry, partial=False):
    available = c.rows(entry, 'test', require_complete=not partial)
    for row in available:
        path = c.RUN / row['row_path']
        identity = dict(row_path=row['row_path'], row_sha256=c.sha(path),
                        status=row['status'], method=row['method'], point=row['point'])
        if row['status'] == 'complete':
            for key in ('reconstruction', 'reconstruction_sha256'):
                identity[key] = row[key]
            if c.sha(c.RUN / row['reconstruction']) != row['reconstruction_sha256']:
                raise RuntimeError('Codec reconstruction identity differs')
        key = row['method'] + '__' + row['point']['label']
        c.write(OUT / 'registered_inputs' / entry['sid'] / (key + '.json'), identity, exclusive=True)
    return available


def row_info(row):
    if row is None:
        return dict(method='raw', point='raw', configuration=None, complete_cr=1.0,
                    compression_row=None, compression_row_sha256=None)
    return dict(method=row['method'], point=row['point']['label'], configuration=row['point'],
                artifact_bytes=row.get('artifact_bytes'), complete_cr=row.get('cr'),
                strict_audit=row.get('strict_audit'), compression_row=row['row_path'],
                compression_row_sha256=c.sha(c.RUN / row['row_path']))


def key_for(row):
    return 'raw' if row is None else row['method'] + '__' + row['point']['label']


def finished(path, row, manifest_sha):
    if not path.exists():
        return False
    value = c.read(path)
    if value['consumer_manifest_sha256'] != manifest_sha or value['compression_row_sha256'] != row_info(row)['compression_row_sha256']:
        raise RuntimeError('Resumed UCR row belongs to different evidence: ' + str(path))
    if value['status'] == 'complete' and c.sha(c.RUN / value['per_query_file']) != value['per_query_sha256']:
        raise RuntimeError('Saved UCR prediction changed')
    return True


def persist(task, consumer, entry, seed, row, manifest_sha, metrics=None, values=None, error=None):
    path = output_file(task, consumer, entry['sid'], seed, key_for(row))
    value = dict(task=task, consumer=consumer, dataset=entry['dataset'], sid=entry['sid'],
                 seed=seed, status='complete' if error is None else 'failed',
                 consumer_manifest_sha256=manifest_sha, **row_info(row))
    if error is None:
        arr = path.with_suffix('.npz')
        save_npz(arr, **values)
        value.update(metrics=metrics, queries=len(values['yquery']),
                     per_query_file=str(arr.relative_to(c.RUN)), per_query_sha256=c.sha(arr))
    else:
        value.update(error=error, metrics=None)
    c.write(path, value, exclusive=True)


def shape_prediction(query, reference_unit, yreference):
    return yreference[np.argmin(np.maximum(0., 2. - 2. * (rm.unit(query) @ reference_unit.T)), axis=1)]


def logistic_model(name):
    with np.load(c.UCR / 'baselines' / name / 'linear_model.npz', allow_pickle=False) as weights:
        model = LogisticRegression(C=float(weights['C']), max_iter=3000, solver='lbfgs', random_state=20260915)
        model.coef_ = weights['coef']; model.intercept_ = weights['intercept']; model.classes_ = weights['classes']
        model.n_features_in_ = model.coef_.shape[1]
        mean, std = float(weights['mean']), float(weights['std'])
    if std <= 0:
        raise RuntimeError('Invalid saved logistic normalizer')
    return model, mean, std


def check_parent_raw(entry, consumer, seed, probability=None, prediction=None, order=None):
    """Bind and compare parent raw output without reusing its task metrics."""
    targets = [('classification', consumer, {'prediction': prediction})]
    if probability is not None:
        targets[0][2]['probability'] = probability
    if order is not None:
        targets.append(('retrieval', 'shape' if consumer == 'one_nn' else consumer,
                        {'ranking': order}))
    checks = []
    for task, name, arrays in targets:
        path = c.PARENT_RUN / 'evaluation/ucr' / task / name / entry['sid'] / str(seed) / 'raw.json'
        row = c.read(path)
        if (row['status'] != 'complete' or row['method'] != 'raw'
                or row['consumer_manifest_sha256'] != c.sha(PARENT_MANIFEST)
                or (row['task'], row['consumer'], row['sid'], row['seed'])
                != (task, name, entry['sid'], seed)):
            raise RuntimeError('Parent UCR raw row identity differs: ' + str(path))
        per_query = c.PARENT_RUN / row['per_query_file']
        if c.sha(per_query) != row['per_query_sha256']:
            raise RuntimeError('Parent UCR raw arrays changed: ' + str(per_query))
        with np.load(per_query, allow_pickle=False) as previous:
            for key, array in arrays.items():
                expected = previous[key]
                if (array is None or array.dtype != expected.dtype
                        or array.shape != expected.shape
                        or array.tobytes(order='C') != expected.tobytes(order='C')):
                    raise RuntimeError('Fresh UCR raw output differs from parent: '
                                       + str(path) + ':' + key)
        checks.append(dict(task=task, consumer=name, row=relative(path),
                           row_sha256=c.sha(path), per_query_file=relative(per_query),
                           per_query_sha256=c.sha(per_query), fields=list(arrays),
                           bitwise_equal=True))
    return checks


def record_raw_check(entry, consumer, seed, probability=None, prediction=None, order=None):
    name = entry['sid']; result = dict(dataset=name, consumer=consumer, seed=seed)
    result['parent_expansion_checks'] = check_parent_raw(
        entry, consumer, seed, probability=probability, prediction=prediction, order=order)
    result['parent_consumer_manifest_sha256'] = c.sha(PARENT_MANIFEST)
    if consumer in ('one_nn', 'logistic'):
        oldpath = c.UCR / 'baselines' / name / ('one_nn_predictions.npz' if consumer == 'one_nn' else 'linear_predictions.npz')
    else:
        oldpath = model_folder(name, consumer, seed) / 'predictions.npz'
    with np.load(oldpath, allow_pickle=False) as old:
        previous = old['raw_pred'] if consumer == 'one_nn' else old['raw_prob'].argmax(1)
        result['classification_changed_raw_labels'] = int(np.sum(prediction != previous))
        if probability is not None:
            result['classification_max_probability_difference'] = float(np.max(np.abs(probability.astype(np.float64) - old['raw_prob'])))
            result['classification_probabilities_allclose_1e5'] = bool(np.allclose(probability, old['raw_prob'], rtol=1e-5, atol=1e-5))
    if order is not None:
        version = 'shape' if consumer == 'one_nn' else consumer
        with np.load(RETRIEVAL / 'rows' / version / name / str(seed) / 'raw.npz', allow_pickle=False) as old:
            result['retrieval_changed_raw_top1'] = int(np.sum(order[:, 0] != old['ranking'][:, 0]))
            result['retrieval_changed_raw_full_rankings'] = int(np.sum(np.any(order != old['ranking'], axis=1)))
    result['interpretation'] = 'Fresh raw output exactly reproduces the parent expansion. September-15 differences below remain historical diagnostics. Every new comparison is freshly scored with the same frozen consumer/device; no best-of-run selection.'
    c.write(OUT / 'raw_controls' / consumer / name / f'{seed}.json', result, exclusive=True)


def cpu_one(entry, compression, manifest_sha):
    data = c.arrays(entry); x = data['raw']; y = data['ytest']; yt = data['ytrain']
    classes = len(data['class_labels']); archive = rm.unit(data['train'])
    raw_nn = shape_prediction(x, archive, yt); raw_order = rm.ranking(rm.unit(x), archive)
    linear, mean, std = logistic_model(entry['sid'])
    raw_prob = linear.predict_proba(cm.normalized(x, mean, std)); raw_pred = raw_prob.argmax(1)
    record_raw_check(entry, 'one_nn', 0, prediction=raw_nn, order=raw_order)
    record_raw_check(entry, 'logistic', 0, probability=raw_prob, prediction=raw_pred)
    for row in [None, *compression]:
        jobs = [('classification', 'one_nn'), ('classification', 'logistic'), ('retrieval', 'shape')]
        todo = [(task, cons) for task, cons in jobs if not finished(output_file(task, cons, entry['sid'], 0, key_for(row)), row, manifest_sha)]
        if not todo:
            continue
        if row is not None and row['status'] != 'complete':
            error = dict(stage='compression', status=row['status'], detail=row.get('error'))
            for task, cons in todo:
                persist(task, cons, entry, 0, row, manifest_sha, error=error)
            continue
        try:
            q = x if row is None else c.decoded(row)[:, 0, :]
            if q.shape != x.shape or not np.isfinite(q).all():
                raise ValueError('Invalid UCR reconstruction')
            pred_nn = raw_nn if row is None else shape_prediction(q, archive, yt)
            prob = raw_prob if row is None else linear.predict_proba(cm.normalized(q, mean, std))
            pred = prob.argmax(1)
            order = raw_order if row is None else rm.ranking(rm.unit(q), archive)
            for task, cons in todo:
                if task == 'classification':
                    p, pp, rp = (pred_nn, None, raw_nn) if cons == 'one_nn' else (pred, prob, raw_pred)
                    metrics = cm.summary(y, p, pp, rp, classes)
                    values = dict(yquery=y, prediction=p, raw_prediction=rp)
                    if pp is not None:
                        values['probability'] = pp
                else:
                    values = rm.evaluate_ranking(order, y, yt, raw_order)
                    metrics = {k: float(v.mean()) for k, v in values.items()}
                    values.update(ranking=order, yquery=y, yreference=yt)
                persist(task, cons, entry, 0, row, manifest_sha, metrics, values)
        except Exception as exc:
            error = dict(stage='consumer', type=type(exc).__name__, detail=str(exc), traceback=traceback.format_exc())
            for task, cons in todo:
                path = output_file(task, cons, entry['sid'], 0, key_for(row))
                if not path.exists():
                    persist(task, cons, entry, 0, row, manifest_sha, error=error)


def deep_one(entry, compression, version, seed, manifest_sha, device):
    import torch
    from experiments.vldb.downstream_expansion.classification.models import ResNetTSC
    data = c.arrays(entry); x = data['raw']; y = data['ytest']; yt = data['ytrain']
    folder = model_folder(entry['sid'], version, seed)
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    if int(checkpoint['seed']) != seed:
        raise RuntimeError('ResNet seed differs')
    source_hashes = [v for k, v in checkpoint['code_identity'].items() if k.split('/')[-1] == 'models.py']
    if source_hashes != [c.sha(ARCH)]:
        raise RuntimeError('ResNet architecture differs')
    if version == 'resnet_v2' and checkpoint['normalization'] != 'per_query_population_znorm_float64_to_float32':
        raise RuntimeError('ResNet v2 preprocessing differs')
    consumer = dict(normalization='global' if version == 'resnet_v1' else 'per_series_z', epsilon=0., ddof=0)
    model = ResNetTSC(checkpoint['classes']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()

    @torch.inference_mode()
    def predict(values):
        z = rm.normalize(values, checkpoint, consumer)
        features, probs = [], []
        for batch in torch.from_numpy(z[:, None, :]).split(64):
            feature = model.blocks(batch.to(device)).mean(-1)
            probs.append(model.classifier(feature).softmax(-1).cpu().numpy())
            features.append(feature.cpu().numpy())
        return np.concatenate(features), np.concatenate(probs)

    archive, _ = predict(data['train']); raw_feature, raw_prob = predict(x)
    raw_pred = raw_prob.argmax(1); raw_order = rm.ranking(raw_feature, archive)
    record_raw_check(entry, version, seed, probability=raw_prob, prediction=raw_pred, order=raw_order)
    for row in [None, *compression]:
        tasks = [task for task in ('classification', 'retrieval') if not finished(output_file(task, version, entry['sid'], seed, key_for(row)), row, manifest_sha)]
        if not tasks:
            continue
        if row is not None and row['status'] != 'complete':
            for task in tasks:
                persist(task, version, entry, seed, row, manifest_sha,
                        error=dict(stage='compression', status=row['status'], detail=row.get('error')))
            continue
        try:
            q = x if row is None else c.decoded(row)[:, 0, :]
            if q.shape != x.shape or not np.isfinite(q).all():
                raise ValueError('Invalid UCR reconstruction')
            feature, prob = (raw_feature, raw_prob) if row is None else predict(q)
            pred = prob.argmax(1)
            order = raw_order if row is None else rm.ranking(feature, archive)
            for task in tasks:
                if task == 'classification':
                    metrics = cm.summary(y, pred, prob, raw_pred, checkpoint['classes'])
                    values = dict(yquery=y, probability=prob, prediction=pred, raw_prediction=raw_pred)
                else:
                    values = rm.evaluate_ranking(order, y, yt, raw_order)
                    metrics = {k: float(v.mean()) for k, v in values.items()}
                    values.update(ranking=order, yquery=y, yreference=yt)
                persist(task, version, entry, seed, row, manifest_sha, metrics, values)
        except Exception as exc:
            error = dict(stage='consumer', type=type(exc).__name__, detail=str(exc), traceback=traceback.format_exc())
            for task in tasks:
                if not output_file(task, version, entry['sid'], seed, key_for(row)).exists():
                    persist(task, version, entry, seed, row, manifest_sha, error=error)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def summarize():
    results = []
    for task in ('classification', 'retrieval'):
        for path in sorted((OUT / task).glob('*/*/*/*.json')):
            r = c.read(path)
            if r['status'] == 'complete' and c.sha(c.RUN / r['per_query_file']) != r['per_query_sha256']:
                raise RuntimeError('UCR summary input array changed')
            results.append(r)
    groups = []
    specs = dict(classification=['one_nn', 'logistic', 'resnet_v1', 'resnet_v2'],
                 retrieval=['shape', 'resnet_v1', 'resnet_v2'])
    for task, consumers in specs.items():
        for consumer in consumers:
            repeats = 3 if consumer.startswith('resnet') else 1
            for method in ['raw', *c.METHODS]:
                selected = [r for r in results if (r['task'], r['consumer'], r['method']) == (task, consumer, method)]
                ok = [r for r in selected if r['status'] == 'complete']
                expected = len(c.entries('ucr')) * repeats
                full = len(selected) == expected
                fields = CLASS_FIELDS if task == 'classification' else RETRIEVAL_FIELDS
                metrics = None
                if len(ok) == expected:
                    metrics = {k: (None if any(r['metrics'].get(k) is None for r in ok) else
                                   float(np.mean([np.mean([r['metrics'][k] for r in ok if r['dataset'] == e['dataset']])
                                                  for e in c.entries('ucr')]))) for k in fields}
                groups.append(dict(task=task, consumer=consumer, method=method, expected=expected,
                    attempted=len(selected), complete=len(ok), failed=len(selected)-len(ok),
                    all_accounted=full, macro_metrics=metrics,
                    source_files=[str(output_file(task, consumer, r['sid'], r['seed'],
                                                  'raw' if method == 'raw' else method + '__' + r['point']).relative_to(c.RUN)) for r in selected]))
    value = dict(schema='nclab.paper-budget-guided.ucr-summary.v1', consumer_manifest_sha256=c.sha(MANIFEST),
                 parent_consumer_manifest_sha256=c.sha(PARENT_MANIFEST),
                 all_accounted=all(g['all_accounted'] for g in groups), groups=groups,
                 result_rows=len(results), failed_rows=sum(r['status'] != 'complete' for r in results),
                 expected_rows=len(c.entries('ucr')) * (1 + len(c.METHODS)) * (8 + 7),
                 aggregation='Seed means within dataset, then six datasets equally; configuration rows are not independent datasets.')
    c.write(OUT / 'SUMMARY.json', value)
    return dict(rows=len(results), failed=value['failed_rows'], all_accounted=value['all_accounted'])


def check():
    """Synthetic, data-independent arithmetic and checkpoint-restoration checks."""
    rm.metric_check()
    rng = np.random.default_rng(20260922)
    for classes in (2, 3):
        x = rng.normal(size=(40, 7)); y = np.arange(40) % classes
        fitted = LogisticRegression(max_iter=200).fit(x, y)
        restored = LogisticRegression()
        restored.coef_ = fitted.coef_.copy(); restored.intercept_ = fitted.intercept_.copy()
        restored.classes_ = fitted.classes_.copy(); restored.n_features_in_ = 7
        if not np.array_equal(fitted.predict_proba(x), restored.predict_proba(x)):
            raise AssertionError('Numeric logistic restore changed predictions')
    reference = np.array([[0., 1., 2.], [2., 1., 0.], [0., 1., 2.]])
    labels = np.array([0, 1, 2])
    if shape_prediction(np.array([[0., 1., 2.], [1., 1., 1.]]), rm.unit(reference), labels).tolist() != [0, 0]:
        raise AssertionError('Shape stable ties / degenerate query failed')
    return dict(passed=True, checks=['hand AP/nDCG/Recall', 'stable ranking ties', 'degenerate shape vector',
                                    'binary and multiclass numeric logistic restore'], new_TEST_scored=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'freeze', 'evaluate', 'summarize'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Defaults to the parent frozen UCR device; changing it is forbidden.')
    parser.add_argument('--dataset')
    parser.add_argument('--consumer', choices=['all', 'cpu', 'resnet_v1', 'resnet_v2'], default='all')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.stage == 'check':
        print(json.dumps(check()), flush=True); return
    if args.stage == 'freeze':
        value = freeze(args.device)
        print(json.dumps(dict(frozen=True, device=value['runtime']['device'], manifest_sha256=c.sha(MANIFEST))), flush=True); return
    protocol = verify()
    if args.stage == 'summarize':
        print(json.dumps(summarize()), flush=True); return
    device = protocol['runtime']['device']
    if args.device is not None and device != args.device:
        raise RuntimeError('CLI device differs from frozen UCR device')
    if configure(device) != protocol['runtime']:
        raise RuntimeError('UCR runtime differs from frozen package/device identity')
    manifest_sha = c.sha(MANIFEST)
    selected = [e for e in c.entries('ucr') if args.dataset is None or e['sid'] == args.dataset]
    if not selected:
        raise ValueError('No selected UCR dataset')
    for entry in selected:
        compression = register_rows(entry, args.partial)
        start = time.perf_counter()
        if args.consumer in ('all', 'cpu'):
            cpu_one(entry, compression, manifest_sha)
        for version in ('resnet_v1', 'resnet_v2'):
            if args.consumer not in ('all', version):
                continue
            for seed in c.SEEDS:
                with c.gpu_lock() if device == 'cuda' else nullcontext():
                    deep_one(entry, compression, version, seed, manifest_sha, device)
                print(json.dumps(dict(dataset=entry['sid'], consumer=version, seed=seed, completed=True)), flush=True)
        print(json.dumps(dict(dataset=entry['sid'], seconds=time.perf_counter()-start, **summarize())), flush=True)


if __name__ == '__main__':
    main()
