"""Inspect TRAIN only and freeze preprocessing / feature consumers before TEST."""
from __future__ import annotations
import io
import json
import zipfile
from .common import *
setup_runtime()
import numpy as np
import scipy
import sklearn
import pycatch22
from scipy.signal import correlate

def first_minimum(row):
    centered = np.asarray(row, dtype=np.float64) - np.mean(row,dtype=np.float64)
    ac = correlate(centered, centered, mode='full', method='fft')[len(row)-1:]
    ac = ac / ac[0] if ac[0] else np.zeros(len(row))
    local = np.flatnonzero((ac[1:-1] < ac[:-2]) & (ac[1:-1] < ac[2:])) + 1
    return int(local[0]) if len(local) else len(row)

def main():
    info = {'created_utc': now(), 'split_inspected': 'TRAIN only', 'datasets': {},
            'lag_rule': 'Smallest common L in 16,32,64,128 covering first minimum plus right neighbor (k+1<=L) for >=99% in each TRAIN set; if none use128 and disclose coverage.',
            'traditional_acf': 'global-mean centered, linear lagged dot product divided by lag0 (biased full-series ACF); first strict local minimum; absent=length. Distinct from JointZip slice-Pearson contract.',
            'versions': {'numpy':np.__version__, 'scipy':scipy.__version__, 'sklearn':sklearn.__version__, 'pycatch22':'0.4.5'}}
    lag_lists=[]
    for dataset in DATASETS:
        archive=RUN/'downloads'/(dataset+'.zip')
        with zipfile.ZipFile(archive) as z:
            raw=z.read(dataset+'_TRAIN.txt')
        dest=RUN/'inputs'/(dataset+'_TRAIN.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        data=np.loadtxt(io.BytesIO(raw),dtype=np.float64)
        labels=data[:,0].astype(np.int64);values=np.ascontiguousarray(data[:,1:],dtype=np.float32)
        assert values.shape[1]==500 and set(labels)=={-1,1} and np.isfinite(values).all()
        lags=np.array([first_minimum(x) for x in values]);lag_lists.append(lags)
        np.savez(RUN/'inputs'/(dataset+'_train.npz'),values=values,labels=labels,first_minimum_lag=lags)
        info['datasets'][dataset]={'source_url':f'https://www.timeseriesclassification.com/aeon-toolkit/{dataset}.zip',
            'archive_sha256':sha(archive),'train_text_sha256':sha(dest),'shape':list(values.shape),
            'class_counts':{str(c):int(sum(labels==c)) for c in [-1,1]},
            'global_train_mean':float(values.mean(dtype=np.float64)),'global_train_std':float(values.std(dtype=np.float64)),
            'row_mean_abs_max':float(abs(values.mean(axis=1,dtype=np.float64)).max()),
            'row_sample_std_minmax': [float(values.std(axis=1,ddof=1,dtype=np.float64).min()),float(values.std(axis=1,ddof=1,dtype=np.float64).max())],
            'first_minimum_quantiles':{str(q):float(np.quantile(lags,q)) for q in [0,.5,.9,.95,.99,1]},
            'coverage':{str(L):float(np.mean(lags+1<=L)) for L in [16,32,64,128]},
            'catch22_first_minimum_first8':[float(pycatch22.CO_FirstMin_ac(x.tolist())) for x in values[:8]],
            'traditional_first_minimum_first8':lags[:8].tolist()}
        print(json.dumps({dataset:info['datasets'][dataset]}),flush=True)
    info['selected_max_lag']=next((L for L in [16,32,64,128] if all(np.mean(l+1<=L)>=.99 for l in lag_lists)),128)
    write(RUN/'TRAIN_PREPARATION.json',info)
    print(json.dumps({'selected_max_lag':info['selected_max_lag']}),flush=True)

if __name__=='__main__': main()
