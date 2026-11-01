"""Reproducible NASA anomaly panel; official TSB-AD architecture/metrics adapters.

Run in WSL: python -m experiments.vldb.downstream_expansion.anomaly.run STAGE.
No TEST label or compressed score is used for training, tuning, or sample selection.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

from experiments.vldb.main_benchmark import native_bridge as nb
from experiments.vldb.prior_comparison.common import audit, sha, write_json, canonical

ROOT = Path(__file__).resolve().parents[4]
UP = ROOT / 'tmp/upstreams/TSB-AD-20260915'
DATA = ROOT / 'tmp/external_utility/tsb_ad/TSB-AD-U'
OUT = ROOT / 'experiments/results/vldb_tsb_nasa_downstream_20260915_v1'
sys.path.insert(0, str(UP))
sys.path.insert(0, str(ROOT / 'tmp/downstream_expansion_deps'))
METHODS = ['qpet_point', 'qpet_pearson', 'qpet_acf', 'qpet_joint', 'sperr', 'sz2016', 'sz3', 'zfp', 'zstd']
POINTS = [dict(label='strict', epsilon=.1, rho=.999, tau=.01),
          dict(label='medium', epsilon=.25, rho=.995, tau=.02),
          dict(label='loose', epsilon=.5, rho=.99, tau=.05)]
SEEDS = [20260915, 20260916, 20260917]
CONTEXT = 150


def catalog():
    p = UP / 'Datasets/File_List/TSB-AD-U-Eva.csv'
    return sorted(n for n in pd.read_csv(p)['file_name'] if '_MSL_' in n or '_SMAP_' in n)


def read_data(name):
    frame = pd.read_csv(DATA / name)
    if frame.isna().any().any() or frame.shape[1] != 2:
        raise ValueError('expected finite univariate CSV without implicit row dropping')
    x = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    labels = frame['Label'].to_numpy(dtype=np.int64)
    assert np.isfinite(x).all() and set(labels).issubset({0, 1})
    tr = int(re.search(r'_tr_(\d+)_', name).group(1))
    cut = int(.8 * tr)
    # Fit scaling on actual optimization TRAIN, excluding held-out normal VAL.
    mean, std = np.float32(x[:cut].mean()), np.float32(x[:cut].std())
    if std < 1e-8:
        std = np.float32(1.)
    z = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    return z, labels, tr, cut, float(mean), float(std)


def series_id(name):
    return name[:-4]


def row_path(name, point, method):
    return OUT / 'compression' / series_id(name) / point / f'{method}.json'


def inputs(name):
    z, labels, tr, cut, mean, std = read_data(name)
    tail = z[tr:]
    padded = np.pad(tail, (0, (-len(tail)) % 128), mode='edge')
    return z, labels, tr, cut, mean, std, padded.reshape(-1, 1, 128)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        verify()
        return
    entries = []
    for name in catalog():
        z, labels, tr, cut, mean, std, blocks = inputs(name)
        entries.append(dict(file=name, source_sha256=sha(DATA / name), length=len(z),
                            train_end=tr, optimization_train_end=cut, test_length=len(z)-tr,
                            train_anomaly_count=int(labels[:tr].sum()),
                            test_anomaly_count=int(labels[tr:].sum()), blocks=len(blocks),
                            mean=mean, std=std, padded_source_sha256=hashlib.sha256(blocks.tobytes()).hexdigest()))
    assert len(entries) == 24
    protocol = dict(schema='nclab.tsb_nasa.frozen_consumer.v1',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        dataset='All 24 MSL/SMAP files in official TSB-AD-U Eval list; no Tuning files',
        upstream_commit='6beac72e11d1155ade40870492c00d0d1cfdcaaf', entries=entries,
        methods=METHODS, points=POINTS, seeds=SEEDS,
        block_scope='128 samples in TEST suffix, final block edge padded; task uses trimmed suffix. No global or cross-block ACF guarantee.',
        task='Offline decoded TEST archive. Both residual target and history are decoded. Raw TRAIN prefix provides initial context. Unchanged labels.',
        training='First 80% of filename tr prefix for optimization; last20% for chronological validation targets, with preceding context. Scaling uses optimization TRAIN only.',
        lstm=dict(source='TSB_AD/models/LSTMAD.py:LSTMModel, unchanged official architecture',
                  context=CONTEXT, pred_len=1, hidden_dim=20, num_layers=2, batch_size=128,
                  epochs=50, patience=3, lr=.0008, step_lr_every=5, gamma=.75,
                  validation='earliest minimum raw VAL MSE checkpoint, no compressed/TEST tuning',
                  deviations='explicit TRAIN scaling, context-aware VAL avoids empty windows on short prefixes, best-checkpoint restore; official wrapper normalizes splits independently and does not restore best checkpoint'),
        ridge=dict(context=CONTEXT, alphas=[.001,.01,.1,1.,10.], selection='raw VAL MSE, then smaller alpha'),
        knn=dict(context=CONTEXT, k=1, dictionary='all optimization TRAIN z-normalized windows',
                 scores='squared z-normalized Euclidean AB-join distance, ending-time alignment; fixed TRAIN dictionary, not transductive TSB Sub-KNN'),
        metrics=dict(primary=['VUS_PR','AP'], secondary=['ROC_AUC','f1_at_val99','precision_at_val99','recall_at_val99'],
                     vus='official generate_curve, opt_mem, 250 thresholds; window=official find_length_rank(raw optimization TRAIN) frozen per file',
                     operational_threshold='99th percentile of raw normal VAL scores, same model/codec threshold, strict >; no point adjustment or TEST oracle threshold',
                     aggregation='equal-file macro with paired file bootstrap; NASA-only scope, two parent domains; three LSTM seeds are repetitions not independent datasets'),
        compression='full SCS2+NOR2+PRV1, three bit-identical replays; normalized float32 error. TRAIN/model/labels unbilled. Same epsilon comparisons only in this new panel; no same-rate claim.',
        gate_tolerance=0., inference_precision='torch float32, score float64',
        freeze_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'experiments/vldb/main_benchmark/native_bridge.py',
            ROOT/'experiments/vldb/prior_comparison/common.py',
            ROOT/'tools/qpet_correlation/bin/qpet_correlation_batch',
            UP/'TSB_AD/models/LSTMAD.py', UP/'TSB_AD/evaluation/basic_metrics.py',
            UP/'TSB_AD/utils/slidingWindows.py',
            ROOT/'tools/sperr/bin/sperr_batch', ROOT/'tools/sz2016/bin/sz2016_batch',
            ROOT/'tmp/modern-codecs/bin/sz3', ROOT/'tmp/modern-codecs/bin/zfp',
            ROOT/'experiments/vldb/main_benchmark/modern_codecs.py',
            ROOT/'src/neural_compression_lab/classical.py',
            ROOT/'src/neural_compression_lab/metrics.py',
            ROOT/'src/neural_compression_lab/runner.py',
            ROOT/'experiments/vldb/utility/artifacts.py']})
    write_json(OUT / 'protocol.json', protocol)
    print(json.dumps(dict(stage='frozen', files=len(entries), test_points=sum(e['test_length'] for e in entries), blocks=sum(e['blocks'] for e in entries), compression_rows=len(entries)*len(POINTS)*len(METHODS), protocol_sha256=sha(OUT/'protocol.json'))), flush=True)


def verify():
    p = json.loads((OUT/'protocol.json').read_text())
    for file, digest in p['freeze_files'].items():
        if sha(ROOT/file) != digest:
            raise RuntimeError(f'frozen source changed: {file}')
    for e in p['entries']:
        if sha(DATA/e['file']) != e['source_sha256']:
            raise RuntimeError('frozen dataset changed')
    return p


def compress_one(task):
    name, point, method = task
    dest = row_path(name, point['label'], method)
    if dest.exists():
        return json.loads(dest.read_text())['status']
    nb.install_linux_transport()
    torch.set_num_threads(1)
    result = dict(file=name, point=point, method=method)
    started = time.perf_counter()
    try:
        z, labels, tr, cut, mean, std, blocks = inputs(name)
        enc = nb.encode(method, blocks, point['epsilon'], point['tau'], point['rho'])
        decoded, record = nb.persist_replay(dest.with_suffix('.scs2'), enc, blocks,
            channel=name, mean=mean, std=std, protocol_sha=sha(OUT/'protocol.json'),
            freeze_sha=sha(OUT/'protocol.json'), method_index=METHODS.index(method), point_index=POINTS.index(point))
        np.save(dest.with_suffix('.npy'), decoded)
        gates = audit(blocks, decoded, point['epsilon'], point['rho'], point['tau'])
        effective = nb.effective_point(method, point['epsilon'], point['tau'], point['rho'])
        if not method.startswith('qpet_'):
            effective = dict(epsilon=point['epsilon'], min_correlation=-1., acf_epsilon=2.)
        result.update(status='ok', storage=record, audit=gates, effective_point=effective,
            enabled_gate_pass=audit(blocks, decoded, effective['epsilon'], effective['min_correlation'], effective['acf_epsilon'])['joint_pass'],
            reconstruction_sha256=sha(dest.with_suffix('.npy')), valid_test_points=len(z)-tr,
            task_complete_cr=4*(len(z)-tr)/record['artifact_bytes'])
    except Exception as e:
        result.update(status='failed', error=repr(e), traceback=traceback.format_exc())
    result['wall_seconds'] = time.perf_counter()-started
    write_json(dest, result)
    return result['status']


def windows(z):
    return np.lib.stride_tricks.sliding_window_view(z, CONTEXT+1)


def znorm(x):
    v = np.asarray(x, dtype=np.float64)
    v = v-v.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n>1e-12)


def fit_linear(z, tr, cut):
    w = windows(z[:tr])
    train = w[:cut-CONTEXT]
    val = w[cut-CONTEXT:]
    candidates = []
    for alpha in [.001,.01,.1,1.,10.]:
        model = Ridge(alpha=alpha).fit(train[:,:-1], train[:,-1])
        candidates.append((float(np.mean((model.predict(val[:,:-1])-val[:,-1])**2)), alpha, model))
    mse, alpha, model = min(candidates, key=lambda a:a[:2])
    return model, dict(alpha=alpha, validation_mse=mse)


def knn_scores(z, dictionary):
    q = znorm(np.lib.stride_tricks.sliding_window_view(z, CONTEXT))
    out = []
    for i in range(0, len(q), 256):
        # Exact Euclidean distance including degenerate normalized vectors.
        a = q[i:i+256]
        dist = (a*a).sum(1)[:,None] + (dictionary*dictionary).sum(1)[None,:] - 2*a@dictionary.T
        out.append(np.maximum(0., dist.min(axis=1))*CONTEXT)
    return np.pad(np.concatenate(out), (CONTEXT-1,0), constant_values=np.nan)


def ridge_scores(z, model):
    w = windows(z)
    s = (model.predict(w[:,:-1]).astype(np.float64)-w[:,-1])**2
    return np.pad(s,(CONTEXT,0),constant_values=np.nan)


def all_archives(name):
    z, labels, tr, cut, _, _ = read_data(name)
    yield 'raw', z
    for point in POINTS:
        for method in METHODS:
            p = row_path(name, point['label'], method)
            if not p.exists() or json.loads(p.read_text())['status'] != 'ok':
                continue
            if sha(p.with_suffix('.npy')) != json.loads(p.read_text())['reconstruction_sha256']:
                raise RuntimeError('persisted reconstruction hash mismatch')
            rec = np.load(p.with_suffix('.npy')).ravel()[:len(z)-tr]
            yield point['label']+'__'+method, np.concatenate([z[:tr],rec])


def train_deep(name, seed):
    from TSB_AD.models.LSTMAD import LSTMModel
    dest = OUT/'models'/series_id(name)/f'lstm_{seed}'
    dest.parent.mkdir(parents=True,exist_ok=True)
    z, labels, tr, cut, _, _ = read_data(name)
    device = torch.device('cuda')
    model = LSTMModel(CONTEXT,1,20,1,2,128,device).to(device)
    if dest.with_suffix('.pt').exists():
        model.load_state_dict(torch.load(dest.with_suffix('.pt'),map_location=device,weights_only=True))
        return model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Recreate after seeding; initialization is part of the repeated experiment.
    model = LSTMModel(CONTEXT,1,20,1,2,128,device).to(device)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w = windows(z[:tr]).copy()
    x = torch.tensor(w[:,:-1,None],device=device)
    y = torch.tensor(w[:,-1,None],device=device)
    ntrain = cut-CONTEXT
    optimizer=torch.optim.Adam(model.parameters(),lr=.0008)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=.75)
    best=float('inf'); stale=0; history=[]; best_state=None
    began=time.perf_counter()
    for epoch in range(1,51):
        model.train(); order=torch.randperm(ntrain,device=device)
        for ix in order.split(128):
            optimizer.zero_grad()
            pred=model(x[ix])[0]
            loss=torch.mean((pred-y[ix])**2)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pv=torch.cat([model(b)[0] for b in x[ntrain:].split(256)])
            vm=float(torch.mean((pv-y[ntrain:])**2))
        history.append(dict(epoch=epoch,validation_mse=vm))
        scheduler.step()
        if vm<best:
            best=vm; stale=0; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=3: break
    assert best_state is not None
    torch.save(best_state,dest.with_suffix('.pt'))
    model.load_state_dict(best_state)
    write_json(dest.with_suffix('.json'),dict(file=name,seed=seed,best_epoch=best_epoch,history=history,
         train_seconds=time.perf_counter()-began,weights_sha256=sha(dest.with_suffix('.pt')),
         torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    return model


def deep_scores(z, model):
    w=windows(z)
    scores=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(w),256):
            b=w[i:i+256]
            x=torch.tensor(b[:,:-1,None].copy(),device=model.device)
            p=model(x)[0].cpu().numpy().ravel().astype(np.float64)
            scores.append((p-b[:,-1])**2)
    return np.pad(np.concatenate(scores),(CONTEXT,0),constant_values=np.nan)


def score_one(task):
    name, model_name, seed = task
    dest=OUT/'scores'/series_id(name)/f'{model_name}_{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    z,labels,tr,cut,_,_=read_data(name)
    if model_name=='lstm':
        with open('/tmp/nclab_downstream_gpu_20260915.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            model=train_deep(name,seed)
            score_archives(name,model_name,seed,dest,lambda a:deep_scores(a,model),cut,tr)
            del model; torch.cuda.empty_cache()
    elif model_name=='ridge':
        model,meta=fit_linear(z,tr,cut)
        write_json(dest/'model.json',meta)
        score_archives(name,model_name,seed,dest,lambda a:ridge_scores(a,model),cut,tr)
    else:
        dictionary=znorm(np.lib.stride_tricks.sliding_window_view(z[:cut],CONTEXT))
        score_archives(name,model_name,seed,dest,lambda a:knn_scores(a,dictionary),cut,tr)
    actual=len(list(dest.glob('*.npz')))
    expected=1+len(POINTS)*len(METHODS)
    result=dict(file=name,model=model_name,seed=seed,
        status='scores_complete' if actual==expected else 'partial_upstream_pending_or_failed',
        score_files=actual,expected_score_files=expected)
    write_json(dest/'coverage.json',result)
    return result


def score_archives(name, model_name, seed, dest, fn, cut, tr):
    z,labels,_,_,_,_=read_data(name)
    rawpath=dest/'raw.npz'
    if not rawpath.exists():
        raw=fn(z)
        threshold=float(np.quantile(raw[cut:tr],.99))
        np.savez_compressed(rawpath,scores=raw[tr:],threshold=threshold,labels=labels[tr:])
    threshold=float(np.load(rawpath)['threshold'])
    for key,decoded in all_archives(name):
        output=dest/(key+'.npz')
        if output.exists(): continue
        scores=fn(decoded)[tr:]
        assert np.isfinite(scores).all() and len(scores)==len(z)-tr
        if key.endswith('__zstd'):
            if not np.array_equal(scores,np.load(rawpath)['scores']):
                raise RuntimeError('lossless input must reproduce raw anomaly scores exactly')
        np.savez_compressed(output,scores=scores,threshold=threshold,labels=labels[tr:])


def metrics_one(path):
    from TSB_AD.evaluation.basic_metrics import generate_curve
    from TSB_AD.utils.slidingWindows import find_length_rank
    path=Path(path); dest=OUT/'metrics'/path.relative_to(OUT/'scores').with_suffix('.json')
    if dest.exists(): return 'exists'
    data=np.load(path); score=data['scores']; label=data['labels']; threshold=float(data['threshold'])
    z,_,tr,cut,_,_=read_data(path.parent.parent.name+'.csv')
    window=int(find_length_rank(z[:cut],rank=1))
    started=time.perf_counter()
    pred=score>threshold
    result=dict(file=path.parent.parent.name+'.csv', model=path.parent.name,
        codec_key=path.stem, score_sha256=sha(path), metrics=dict(
            AP=float(average_precision_score(label,score)),ROC_AUC=float(roc_auc_score(label,score)),
            f1_at_val99=float(f1_score(label,pred,zero_division=0)),
            precision_at_val99=float(precision_score(label,pred,zero_division=0)),
            recall_at_val99=float(recall_score(label,pred,zero_division=0))),
        frozen_score_threshold=threshold, vus_window=window, test_anomaly_fraction=float(label.mean()))
    curve=generate_curve(label.astype(int),score,window,version='opt_mem',thre=250)
    result['metrics'].update(VUS_ROC=float(curve[-2]),VUS_PR=float(curve[-1]))
    result['metric_seconds']=time.perf_counter()-started
    write_json(dest,result)
    return 'ok'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['freeze','compress','score','metrics'])
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--model',choices=['all','linear','lstm'],default='all')
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    torch.set_num_threads(1)
    if args.stage=='freeze': freeze(); return
    verify()
    if args.stage=='compress':
        jobs=[(n,p,m) for n in catalog() for p in POINTS for m in METHODS]
        fn=compress_one
    elif args.stage=='score':
        jobs=[]
        for n in catalog():
            if args.model in ['all','linear']: jobs +=[(n,'ridge',0),(n,'knn',0)]
            if args.model in ['all','lstm']: jobs +=[(n,'lstm',s) for s in SEEDS]
        fn=score_one
    else:
        jobs=sorted((OUT/'scores').glob('*/*/*.npz')); fn=metrics_one
    if args.limit: jobs=jobs[:args.limit]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(fn,j):j for j in jobs}
        for count,future in enumerate(cf.as_completed(pending),1):
            job=pending[future]
            try: result=future.result()
            except Exception as e:
                result=dict(status='failed',error=repr(e),traceback=traceback.format_exc())
                write_json(OUT/'failures'/(hashlib.sha256(str(job).encode()).hexdigest()+'.json'),dict(stage=args.stage,job=str(job),result=result))
            print(json.dumps(dict(stage=args.stage,completed=count,total=len(jobs),job=str(job),result=result)),flush=True)


if __name__=='__main__': main()
