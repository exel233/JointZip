"""Frozen full-test FordA/FordB ER/ERA experiment with resumable batch outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import io
import json
import os
import pickle
import platform
import time
import traceback
import zipfile
from .common import *
setup_runtime()
import numpy as np
import scipy
from scipy.signal import periodogram
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import pycatch22
from .prepare import first_minimum
from experiments.vldb.framework_ablation_20261017 import codec, container

CHUNK = 16
FEATURE_KINDS = ('catch22', 'spectrum32', 'first_minimum')

def scientific_files():
    relatives=['experiments/vldb/framework_ablation_20261017/'+n for n in ('codec.py','container.py','native_cache.py')]
    relatives+=['experiments/vldb/main_benchmark/modern_codecs.py']
    return list(Path(__file__).parent.glob('*.py'))+[ROOT/p for p in relatives]

def runtime_files():
    paths={str(codec.REPAIR),str(codec.ANCHORS),str(codec.BRIDGE_MANIFEST)}
    for b in BACKENDS:paths.add(codec.identity(b)['binary'])
    paths.add('/opt/jointzip-home/.local/cth_expansion_runs/vldb_framework_ablation_20261017_v1/runtime/ENVIRONMENT.json')
    return sorted(paths)

def features(values):
    all_values=[]
    for row in values:
        result=pycatch22.catch22_all(row.tolist())
        named=dict(zip(result['names'],result['values']))
        names=sorted(named)
        all_values.append([named[n] for n in names])
    c=np.asarray(all_values,dtype=np.float64)
    c[~np.isfinite(c)]=np.nan
    frequencies,power=periodogram(values,fs=1.,window='boxcar',detrend='constant',
                                 scaling='density',axis=1)
    bins=np.minimum((frequencies[1:]*64).astype(int),31)
    spectrum=np.stack([np.bincount(bins,weights=r[1:],minlength=32) for r in power])
    spectrum/=np.maximum(spectrum.sum(axis=1,keepdims=True),np.finfo(float).tiny)
    first=c[:,names.index('CO_FirstMin_ac')][:,None]
    return {'catch22':c,'spectrum32':spectrum,'first_minimum':first},names

def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)

def read_protocol():return json.loads((RUN/'PROTOCOL.json').read_text())

def train_and_freeze():
    if (RUN/'PROTOCOL.json').exists():
        protocol=read_protocol()
        for path,digest in protocol['producer_sha256'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen producer changed: '+path)
        for name,record in protocol['datasets'].items():
            if sha(RUN/record['models'])!=record['models_sha256']:raise RuntimeError('Frozen model changed')
            if sha(RUN/'inputs'/(name+'_train.npz'))!=record['train_npz_sha256']:raise RuntimeError('Frozen TRAIN changed')
        for path,digest in protocol['runtime_sha256'].items():
            if sha(path)!=digest:raise RuntimeError('Frozen runtime changed: '+path)
        for path,digest in protocol['dependency_sha256'].items():
            if sha(RUN/path)!=digest:raise RuntimeError('Frozen dependency changed: '+path)
        return protocol
    prep=json.loads((RUN/'TRAIN_PREPARATION.json').read_text())
    assert json.loads((RUN/'SMOKE_COMPLETE.json').read_text())['complete']
    assert json.loads((RUN/'TRAIN_ASSEMBLY_VALIDATION.json').read_text())['passed']
    protocol=dict(schema='jointzip.acf-ford-real.v1',created_utc=now(),
        hypotheses='Literature-motivated low-frequency/autocorrelation class structure; added A can preserve useful dynamics beyond E/R, without assuming every consumer improves.',
        literature={'catch22':'https://doi.org/10.1007/s10618-019-00647-x',
                    'dataset':'https://www.timeseriesclassification.com/'},
        datasets={},backends=list(BACKENDS),arms=ARMS,epsilon=1.,rho=.99,
        max_lag=prep['selected_max_lag'],lag_selection=prep,
        input='Official existing per-record z-scored sequences cast to float32, no additional fitted normalization or padding; one record=one 500-point window.',
        scope='All official TEST records: FordA1320/FordB810, no case filtering or retuning. TRAIN only fits consumers and selects lag coverage.',
        consumers={'primary':'catch22 all22 sorted by name + TRAIN median imputation + StandardScaler + LogisticRegression(C=1,class_weight=balanced,solver=lbfgs,max_iter=5000,random_state=20261028)',
          'complementary':'32 equal-width frequency bands of relative one-sided boxcar periodogram energy excluding DC; same raw-TRAIN fitted pipeline',
          'mechanism_only':'CO_FirstMin_ac alone with same pipeline; traditional global-mean ACF definition differs from JointZip slice-Pearson contract'},
        metrics=['balanced_accuracy','accuracy','AUROC','prediction_flip_vs_raw','label_improved_declined_vs_ER',
                 'full_JZF1_CR','actual_E_R_slice_ACF','traditional_ACF_gap','first_minimum_error','lossless','repair_values_bytes'],
        codec='Unchanged framework_ablation_20261017 codec; original formula22/12/512; original simple legacy greedy correction formats0-6. Disable A with tau=-1 for ER.',
        archive='One complete JZF1 file per dataset/backend/arm, formed losslessly from completed batch records; full metadata billed. Batch files are checkpoints, not reported denominator.',
        cache='Exact native unrepaired candidate cache shared across arms; no encoding speed claims.',
        test_feedback='No TEST performance is used to select datasets, parameters, lag, consumer, or cases.',
        chunk_size=CHUNK,versions={'python':platform.python_version(),'numpy':np.__version__,
         'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pycatch22':'0.4.5'},
        native={b:codec.identity(b) for b in BACKENDS},
        dependency_sha256={str(p.relative_to(RUN)):sha(p) for p in sorted((RUN/'deps').rglob('*')) if p.is_file() and p.suffix in ('.so','.py')},
        runtime_sha256={p:sha(p) for p in runtime_files()},
        runtime_environment={k:v for k,v in os.environ.items() if k.startswith('JOINTZIP_') or k in ('MODERN_CODECS_PREFIX','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        producer_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(scientific_files())})
    for dataset in DATASETS:
        train=np.load(RUN/'inputs'/(dataset+'_train.npz'))
        x,y=train['values'],train['labels']
        started=time.perf_counter();feat,names=features(x)
        models={};fit_info={}
        for kind in FEATURE_KINDS:
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=5000,random_state=20261028))
            model.fit(feat[kind],y);models[kind]=model
            fit_info[kind]={'n_features':int(feat[kind].shape[1]),'nonfinite_features':int(np.count_nonzero(~np.isfinite(feat[kind]))),
                           'iterations':model[-1].n_iter_.tolist()}
            if max(model[-1].n_iter_)>=5000:raise RuntimeError('Consumer failed to converge')
        path=RUN/'models'/(dataset+'.pkl');path.parent.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(models,protocol=5))
        save_npz(RUN/'features'/(dataset+'_train.npz'),labels=y,**feat)
        protocol['datasets'][dataset]={**prep['datasets'][dataset],
            'models':str(path.relative_to(RUN)),'models_sha256':sha(path),
            'train_npz_sha256':sha(RUN/'inputs'/(dataset+'_train.npz')),
            'catch22_feature_names':names,'training_seconds':time.perf_counter()-started,'fit_info':fit_info}
        print(json.dumps({'stage':'train_complete','dataset':dataset,'seconds':time.perf_counter()-started}),flush=True)
    write(RUN/'PROTOCOL.json',protocol)
    return protocol

def test_inputs(protocol):
    summary={}
    binding=RUN/'TEST_BINDING.json'
    prior=json.loads(binding.read_text()) if binding.exists() else None
    for dataset,expected in [('FordA',1320),('FordB',810)]:
        target=RUN/'inputs'/(dataset+'_test.npz')
        if not target.exists():
            archive=RUN/'downloads'/(dataset+'.zip')
            assert sha(archive)==protocol['datasets'][dataset]['archive_sha256']
            with zipfile.ZipFile(archive) as z:raw=z.read(dataset+'_TEST.txt')
            data=np.loadtxt(io.BytesIO(raw),dtype=np.float64)
            x=np.ascontiguousarray(data[:,1:],dtype=np.float32);y=data[:,0].astype(np.int64)
            assert x.shape==(expected,500) and set(y)=={-1,1} and np.isfinite(x).all()
            save_npz(target,values=x,labels=y)
            text=RUN/'inputs'/(dataset+'_TEST.txt');text.write_bytes(raw)
        data=np.load(target)
        assert data['values'].shape==(expected,500) and set(data['labels'])=={-1,1} and np.isfinite(data['values']).all()
        summary[dataset]={'count':expected,'npz_sha256':sha(target),'labels':{str(k):int(sum(data['labels']==k)) for k in [-1,1]}}
    if prior is not None:
        if summary!=prior:raise RuntimeError('Frozen TEST binding changed')
    else:write(binding,summary)

def chunk_task(task):
    dataset,backend,start,stop=task
    protocol=read_protocol();lag=protocol['max_lag']
    x=np.load(RUN/'inputs'/(dataset+'_test.npz'))['values'][start:stop,None,:]
    x=np.ascontiguousarray(x,dtype=np.float32)
    native_cache=codec.NativeCandidateCache(WORK/'native_cache'/backend)
    result=[]
    for arm,tau in ARMS.items():
        name=f'{dataset}_{backend}_{arm}_{start:04d}_{stop:04d}'
        folder=RUN/'chunks'/name;record=folder/'COMPLETE.json'
        if record.exists():
            saved=json.loads(record.read_text())
            if saved['source_sha256']!=hashlib.sha256(x.tobytes()).hexdigest():raise RuntimeError('Checkpoint source differs '+name)
            for f in ('stream.jzp','decoded.npy','metadata.json'):
                if sha(folder/f)!=saved['sha256'][f]:raise RuntimeError('Checkpoint differs '+name)
            result.append({'name':name,'cached':True});continue
        folder.mkdir(parents=True,exist_ok=True)
        work=WORK/'work'/name
        started=time.perf_counter()
        encoded=codec.encode(backend,x,1.,tau,.99,lag,True,work,native_cache=native_cache)
        if encoded.metadata['audit']['numerical_pass_windows']!=len(x):raise RuntimeError('Contract violation')
        (folder/'stream.jzp').write_bytes(encoded.payload)
        np.save(folder/'decoded.npy',encoded.reconstruction)
        write(folder/'metadata.json',encoded.metadata)
        saved={'dataset':dataset,'backend':backend,'arm':arm,'start':start,'stop':stop,
               'seconds':time.perf_counter()-started,'source_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
               'sha256':{f:sha(folder/f) for f in ('stream.jzp','decoded.npy','metadata.json')}}
        write(record,saved);result.append({'name':name,'seconds':saved['seconds']})
        print(json.dumps({'stage':'chunk',**result[-1]}),flush=True)
    return result

def compress_all(workers):
    tasks=[]
    for dataset,count in [('FordA',1320),('FordB',810)]:
        for start in range(0,count,CHUNK):
            for backend in BACKENDS:tasks.append((dataset,backend,start,min(start+CHUNK,count)))
    completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(chunk_task,t):t for t in tasks}
        for future in as_completed(futures):
            future.result();completed+=1
            write(RUN/'STATUS.json',{'phase':'compression','groups_completed':completed,'groups_total':len(tasks),'updated_utc':now()})

def traditional_acf(x,L):
    x=np.asarray(x,dtype=float);x=x-x.mean(axis=1,keepdims=True)
    denom=np.sum(x*x,axis=1)
    return np.stack([np.sum(x[:,:-k]*x[:,k:],axis=1)/denom for k in range(1,L+1)],axis=1)

def score_predictions(y,model,feat):
    pred=model.predict(feat);score=model.predict_proba(feat)[:,list(model.classes_).index(1)]
    return pred,score,dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),AUROC=float(roc_auc_score(y,score)))

def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze():
    protocol=read_protocol();lag=protocol['max_lag'];summary=[];mechanisms=[];contrasts=[]
    for dataset,count in [('FordA',1320),('FordB',810)]:
        data=np.load(RUN/'inputs'/(dataset+'_test.npz'));x,y=data['values'],data['labels']
        models=pickle.loads((RUN/protocol['datasets'][dataset]['models']).read_bytes())
        raw_features,feature_names=features(x)
        assert feature_names==protocol['datasets'][dataset]['catch22_feature_names']
        raw_acf=traditional_acf(x,lag);raw_min=np.asarray([first_minimum(r) for r in x])
        predictions={}
        for kind in FEATURE_KINDS:
            pred,score,metric=score_predictions(y,models[kind],raw_features[kind]);predictions['raw',kind]=(pred,score)
            summary.append(dict(dataset=dataset,backend='Raw',arm='Raw',consumer=kind,n=count,**metric,full_bytes=x.nbytes,CR=1.,flip_vs_raw=0))
        save_npz(RUN/'features'/(dataset+'_raw_test.npz'),labels=y,**raw_features)
        for backend in BACKENDS:
            for arm,tau in ARMS.items():
                parts=[];records=[];counts={};all_evidence=[]
                for start in range(0,count,CHUNK):
                    stop=min(start+CHUNK,count);folder=RUN/'chunks'/f'{dataset}_{backend}_{arm}_{start:04d}_{stop:04d}'
                    receipt=json.loads((folder/'COMPLETE.json').read_text())
                    payload=(folder/'stream.jzp').read_bytes();meta=json.loads((folder/'metadata.json').read_text())
                    for name in ('stream.jzp','decoded.npy','metadata.json'):
                        assert sha(folder/name)==receipt['sha256'][name]
                    assert hashlib.sha256(np.ascontiguousarray(x[start:stop,None,:]).tobytes()).hexdigest()==receipt['source_sha256']
                    header,chunk_records=codec.unframe(payload)
                    assert header==dict(codec=backend,shape=(stop-start,1,500),epsilon=1.,tau=tau,rho=.99,max_lag=lag)
                    records.extend(chunk_records);parts.append(np.load(folder/'decoded.npy'))
                    all_evidence.extend(meta['windows'])
                    for key,value in meta['counts'].items():counts[key]=counts.get(key,0)+value
                restored=np.concatenate(parts,axis=0)
                payload,_=codec._frame(backend,restored.shape,1.,tau,.99,lag,records)
                full,_=container.frame(payload,{'task':'classification','mean':0.,'std':1.,'channel':dataset},'test',restored.shape)
                out=RUN/'artifacts'/f'{dataset}_{backend}_{arm}.jzf';out.parent.mkdir(exist_ok=True);out.write_bytes(full)
                persisted=out.read_bytes();assert persisted==full
                decoded,_=container.decode(persisted,WORK/'final_decode'/f'{dataset}_{backend}_{arm}')
                assert np.array_equal(decoded.view('u4'),restored.view('u4'))
                checks=codec.audit(x[:,None,:],decoded,1.,tau,.99,lag)
                assert checks['numerical_pass_windows']==count
                save_npz(RUN/'reconstructions'/f'{dataset}_{backend}_{arm}.npz',values=decoded[:,0,:],labels=y)
                f,names=features(decoded[:,0,:]);assert names==feature_names
                save_npz(RUN/'features'/f'{dataset}_{backend}_{arm}.npz',labels=y,**f)
                final_acf=traditional_acf(decoded[:,0,:],lag);final_min=np.asarray([first_minimum(r) for r in decoded[:,0,:]])
                mech=dict(dataset=dataset,backend=backend,arm=arm,n=count,full_bytes=len(full),CR=x.nbytes/len(full),
                    max_E=checks['max_abs_error'],min_R=checks['min_pearson'],max_slice_ACF=checks['max_acf_error'],
                    enabled_strict_pass=checks['strict_pass_windows'],enabled_numerical_pass=checks['numerical_pass_windows'],
                    slice_ACF_pass005=sum(r['max_acf_error']<=.05+64*np.finfo(float).eps for r in checks['rows']),
                    slice_ACF_pass001=sum(r['max_acf_error']<=.01+64*np.finfo(float).eps for r in checks['rows']),
                    traditional_ACF_MAE=float(np.mean(abs(final_acf-raw_acf))),traditional_ACF_max=float(abs(final_acf-raw_acf).max()),
                    first_minimum_MAE=float(np.mean(abs(final_min-raw_min))),first_minimum_changed=int(sum(final_min!=raw_min)),
                    lossless_windows=counts['lossless_windows'],repair_values=counts['correction_values'],
                    correction_payload_bytes=sum(len(r['correction']) for r in records))
                mechanisms.append(mech)
                per_window=[dict(index=i,label=int(y[i]),first_min_raw=int(raw_min[i]),first_min_decoded=int(final_min[i]),
                                 traditional_ACF_MAE=float(np.mean(abs(final_acf[i]-raw_acf[i]))),**checks['rows'][i]) for i in range(count)]
                csv_write(RUN/'artifacts'/f'{dataset}_{backend}_{arm}_windows.csv',per_window)
                for kind in FEATURE_KINDS:
                    pred,score,metric=score_predictions(y,models[kind],f[kind]);predictions[backend,arm,kind]=(pred,score)
                    summary.append(dict(dataset=dataset,backend=backend,arm=arm,consumer=kind,n=count,**metric,
                                        full_bytes=len(full),CR=x.nbytes/len(full),flip_vs_raw=int(sum(pred!=predictions['raw',kind][0]))))
                print(json.dumps({'stage':'analyzed','dataset':dataset,'backend':backend,'arm':arm}),flush=True)
        for backend in BACKENDS:
            for arm in ('ERA_tau005','ERA_tau001'):
                for kind in FEATURE_KINDS:
                    p0,s0=predictions[backend,'ER',kind];p1,s1=predictions[backend,arm,kind]
                    delta=(p1==y).astype(float)-(p0==y).astype(float)
                    rng=np.random.default_rng(20261028);boot=[]
                    a=delta[y==-1];b=delta[y==1]
                    for _ in range(5000):boot.append(.5*(rng.choice(a,len(a)).mean()+rng.choice(b,len(b)).mean()))
                    contrasts.append(dict(dataset=dataset,backend=backend,consumer=kind,before='ER',after=arm,
                        accuracy_change_pp=float(100*delta.mean()),balanced_accuracy_change_pp=float(50*(a.mean()+b.mean())),
                        descriptive_BA_CI95_low_pp=float(100*np.quantile(boot,.025)),descriptive_BA_CI95_high_pp=float(100*np.quantile(boot,.975)),
                        AUROC_change=float(roc_auc_score(y,s1)-roc_auc_score(y,s0)),label_improved=int(sum(delta>0)),
                        label_tied=int(sum(delta==0)),label_declined=int(sum(delta<0))))
        save_npz(RUN/'predictions'/f'{dataset}.npz',labels=y,**{'__'.join(k):np.stack(v) for k,v in predictions.items()})
    csv_write(RUN/'downstream.csv',summary);csv_write(RUN/'mechanism.csv',mechanisms);csv_write(RUN/'contrasts.csv',contrasts)
    write(RUN/'RESULTS.json',dict(downstream=summary,mechanism=mechanisms,contrasts=contrasts))
    from .report import generate
    generate()
    write(RUN/'COMPLETE.json',dict(complete=True,windows=2130*2*3,streams=12,consumer_metrics=42,
          protocol_sha256=sha(RUN/'PROTOCOL.json'),completed_utc=now()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--train-only',action='store_true');parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    lock=(WORK/'RUN.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another experiment driver already holds RUN.lock')
    driver=dict(pid=os.getpid(),parent_pid=os.getppid(),started_utc=now(),active=True,workers=args.workers)
    write(RUN/'DRIVER.json',driver)
    try:
        write(RUN/'STATUS.json',dict(phase='train_or_verify_freeze',updated_utc=now()))
        p=train_and_freeze()
        if args.train_only:return
        test_inputs(p)
        if not args.analyze_only:compress_all(args.workers)
        write(RUN/'STATUS.json',dict(phase='analysis',updated_utc=now()))
        analyze();write(RUN/'STATUS.json',dict(phase='complete',updated_utc=now()))
    except Exception:
        failure=dict(error=traceback.format_exc(),updated_utc=now())
        write(RUN/'failures'/(str(time.time_ns())+'.json'),failure)
        write(RUN/'FAILED.json',failure);raise
    finally:
        write(RUN/'DRIVER.json',{**driver,'active':False,'finished_utc':now()})

if __name__=='__main__':main()
