"""Separate float64/SVD Ridge check prompted by TRAIN/VAL numerical warnings.

Keeps all original float32 results. No TEST metric selects this solver or alpha.
"""
import concurrent.futures as cf
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from . import run as r

BASE=r.OUT
OUT=BASE/'supplements/ridge_svd64'


def freeze():
    protocol=dict(schema='nclab.tsb.ridge-svd64.sensitivity.v1',
        original_protocol_sha256=r.sha(BASE/'protocol.json'),code_sha256=r.sha(Path(__file__)),
        motivation='TRAIN/VAL-only numerical diagnostic found float32 Cholesky sensitivity; all original results retained.',
        files=r.catalog(),model='sklearn Ridge, float64 fit/predict, solver=svd',
        alphas=[.001,.01,.1,1.,10.],selection='minimum raw chronological VAL MSE, smaller alpha tie',
        input='same frozen normalized float32 data promoted to float64, not a new compression experiment',
        task_and_metrics='same offline decoded TEST target/history, official AP/VUS, VAL99 threshold; independent result directory',
        expected_rows=24*28)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:r.write_json(p,protocol)


def score():
    for name in r.catalog():
        z,labels,tr,cut,_,_=r.read_data(name)
        w=r.windows(z[:tr]).astype(np.float64)
        a=w[:cut-r.CONTEXT];v=w[cut-r.CONTEXT:]
        candidates=[]
        for alpha in [.001,.01,.1,1.,10.]:
            m=Ridge(alpha=alpha,solver='svd').fit(a[:,:-1],a[:,-1])
            vm=float(np.mean((m.predict(v[:,:-1])-v[:,-1])**2))
            candidates.append((vm,alpha,m))
        vm,alpha,model=min(candidates,key=lambda t:t[:2])
        dest=OUT/'scores'/r.series_id(name)/'ridge_svd64_0'
        dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'weights.npz',coef=model.coef_,intercept=model.intercept_,alpha=alpha)
        r.write_json(dest/'model.json',dict(alpha=alpha,val_mse=vm,
            candidates=[dict(alpha=a,validation_mse=b) for b,a,_ in candidates],
            weights_sha256=r.sha(dest/'weights.npz')))
        raw=r.ridge_scores(z.astype(np.float64),model)
        threshold=float(np.quantile(raw[cut:tr],.99))
        for key,x in r.all_archives(name):
            p=dest/(key+'.npz')
            if p.exists():continue
            scores=r.ridge_scores(x.astype(np.float64),model)[tr:]
            if key.endswith('__zstd'):assert np.array_equal(scores,raw[tr:])
            assert np.isfinite(scores).all()
            np.savez_compressed(p,scores=scores,labels=labels[tr:],threshold=threshold)
        n=len([p for p in dest.glob('*.npz') if p.name!='weights.npz'])
        r.write_json(dest/'coverage.json',dict(actual=n,expected=28,complete=n==28))
        print(json.dumps(dict(stage='ridge_svd64_score',file=name,actual=n,expected=28)),flush=True)


def metric(path):
    # Only relocate the unchanged official-metric adapter's output directory.
    r.OUT=OUT
    return r.metrics_one(path)


def metrics():
    paths=[p for p in sorted((OUT/'scores').glob('*/*/*.npz')) if p.name!='weights.npz']
    with cf.ProcessPoolExecutor(max_workers=2) as pool:
        for index,result in enumerate(pool.map(metric,paths),1):
            if index%25==0 or index==len(paths):print(json.dumps(dict(stage='ridge_svd64_metrics',completed=index,total=len(paths),result=result)),flush=True)
    records=[json.loads(p.read_text()) for p in (OUT/'metrics').glob('*/*/*.json')]
    rows=[]
    for key in ['raw']+[p['label']+'__'+m for p in r.POINTS for m in r.METHODS]:
        group=[x for x in records if x['codec_key']==key]
        rows.append(dict(codec=key,files=len(group),metrics={k:float(np.mean([x['metrics'][k] for x in group])) for k in ['AP','VUS_PR','ROC_AUC']} if group else {}))
    r.write_json(OUT/'summary.json',dict(status='complete' if len(records)==672 else 'partial',actual=len(records),expected=672,rows=rows))


if __name__=='__main__':
    r.torch.set_num_threads(1)
    r.verify();freeze()
    if sys.argv[-1]=='metrics':metrics()
    else:score()
