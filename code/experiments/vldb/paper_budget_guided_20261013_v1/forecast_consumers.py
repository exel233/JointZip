"""Frozen forecast consumers and validation-only task-quality rate selection."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import traceback
import numpy as np
from . import common as c

HERE = Path(__file__).resolve()
OUT = c.RUN / 'evaluation/forecast'
MANIFEST = OUT / 'consumer_manifest.json'
SELECTION = OUT / 'quality_selections.json'
DLINEAR = c.ROOT / 'experiments/results/vldb_edbt_dlinear_20260907_v1'
FAMILIES = ['ridge','dlinear','patchtst','chronos2','chronos_bolt_small','timesfm3']
FOUNDATIONS = FAMILIES[3:]
INSTANCES = [(m,s) for m in FAMILIES for s in (c.SEEDS if m=='patchtst' else [0])]
BATCH = 128

def rel(p): return str(Path(p).relative_to(c.ROOT)).replace('\\','/')
def code_files():
    base=c.ROOT/'experiments/vldb'
    paths=[HERE,HERE.with_name('common.py'),base/'main_benchmark/tasks_data.py',base/'downstream_models/dlinear.py',
           base/'downstream_expansion/forecasting/runner.py',base/'downstream_expansion/foundation/models.py']
    official=c.ROOT/'tmp/upstreams/PatchTST-20260915/PatchTST_supervised'
    paths += sorted((official/'models').glob('*.py'))+sorted((official/'layers').glob('*.py'))
    # DLinear dynamically imports this pinned official source.
    from experiments.vldb.downstream_models import dlinear
    for name in ('OFFICIAL','OFFICIAL_ROOT','MODEL_FILE','OFFICIAL_MODEL'):
        value=getattr(dlinear,name,None)
        if isinstance(value,Path):
            paths += [value] if value.is_file() else sorted(value.rglob('*.py'))
    return sorted(set(paths))

def freeze():
    if MANIFEST.exists(): return verify()
    c.frozen(); files=code_files()
    files += [c.MAIN/'freeze.json',DLINEAR/'freeze.json',c.PATCH/'freeze.json',c.FOUNDATION/'models.json',
              c.PARENT_RUN/'evaluation/forecast/consumer_manifest.json',
              c.PARENT_RUN/'freeze.json']
    for e in c.entries('forecast'):
        files += [c.MAIN/'models'/e['dataset']/(e['sid']+'.json')]
    for ds in sorted({e['dataset'] for e in c.entries('forecast')}):
        files += [DLINEAR/'models'/(ds+ext) for ext in ('.json','.npz')]
        for s in c.SEEDS: files += [c.PATCH/'models'/str(s)/(ds+ext) for ext in ('.json','.npz')]
    value=dict(schema='nclab.paper-budget-guided.forecast-consumers.v1',created_utc=c.now(),strategy=c.STRATEGY,
        expansion_freeze_sha256=c.sha(c.RUN/'freeze.json'),files={rel(p):c.sha(p) for p in sorted(set(files))},
        instances=[dict(model=m,seed=s) for m,s in INSTANCES],
        protocol=dict(input='Same TRAIN-normalized contexts; Ridge fixed original order<=96, all other consumers final96; uncompressed future24.',
            fitting='None. All original raw-trained weights reused. Six DLinear models share channel-independent weights within a source.',
            inference='FP32 CUDA neural inference, no TF32, deterministic; Ridge original float64 NumPy multiplication.',
            batching='Raw first; concatenate only pending nonreused unique panels. Every call has exactly128 contexts; repeat first context to pad the final call. Raw is replayed alone and compared bitwise with the parent run before reuse.',
            reuse='Parent predictions may be reused only for unchanged baseline streams or simple reconstructions bitwise equal to the corresponding parent reconstruction, after immutable source/model/runtime checks and fresh per-series Raw bitwise replay. Metrics and current-byte accounting are recomputed. Older differently batched legacy scores are not imported.',
            prediction='Foundation median q0.5 plus nine quantiles; other models original point predictions.',
            models='Official pinned PatchTST/DLinear/Chronos/TimesFM code via original adapters; original fitted NumPy Ridge.',
            order='All six-family VAL selections persisted before any new forecast TEST scoring.',
            failures='Retain codec/model failures; they cannot win validation selection. Common lossless Zstd fallback must reproduce raw quality.',
            eligibility='Validation candidate must also pass its enabled numeric codec contract with the predeclared1e-7 tolerance; zero-tolerance outcomes separate. Original CAMEO has no E/R/max-lag promise and is marked unconstrained, eligible by task quality only.',
            aggregation='Average PatchTST seed MSE before selecting a shared configuration; origins within series, series within source, six source means equally weighted.',
            quality_CR=c.PROTOCOL['quality_CR']))
    c.write(MANIFEST,value,exclusive=True)
    return dict(manifest_sha256=c.sha(MANIFEST),instances=len(INSTANCES))

def verify():
    v=c.read(MANIFEST)
    if v['expansion_freeze_sha256']!=c.sha(c.RUN/'freeze.json'): raise RuntimeError('Expansion freeze changed')
    for p,h in v['files'].items():
        if c.sha(c.ROOT/p)!=h: raise RuntimeError('Frozen forecast dependency changed: '+p)
    return v

def configure(model,seed):
    import torch
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1);torch.manual_seed(seed or 20260922)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    packages={p:importlib.metadata.version(p) for p in ('numpy','torch')}
    if model in FOUNDATIONS:
        packages.update({p:importlib.metadata.version(p) for p in ('chronos-forecasting','timesfm','transformers','safetensors','accelerate','einops')})
    v=dict(model=model,seed=seed,python=platform.python_version(),packages=packages,device='cpu' if model=='ridge' else 'cuda',
           gpu=None if model=='ridge' else torch.cuda.get_device_name(0),batch=BATCH,tf32=False,deterministic=True)
    c.write(OUT/'runtime'/(model+'__'+str(seed)+'.json'),v,exclusive=True)
    return v

def load(model,seed,dataset,sid):
    if model=='ridge': return c.read(c.MAIN/'models'/dataset/(sid+'.json'))
    if model=='dlinear':
        from experiments.vldb.downstream_models.dlinear import load_model
        return load_model(DLINEAR,dataset,device='cuda')[0]
    if model=='patchtst':
        from experiments.vldb.downstream_expansion.forecasting.runner import load_model
        return load_model(c.PATCH,dataset,seed)[0]
    from experiments.vldb.downstream_expansion.foundation.models import load_model
    return load_model(model,c.read(c.FOUNDATION/'models.json')['models'][model])

def predict(model,net,x):
    x=np.ascontiguousarray(x[:,-96:],dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=96 or not np.isfinite(x).all(): raise ValueError('Invalid context')
    points=[];quantiles=[]
    for start in range(0,len(x),BATCH):
        b=x[start:start+BATCH];n=len(b)
        if n<BATCH: b=np.concatenate([b,np.repeat(b[:1],BATCH-n,axis=0)])
        if model=='ridge':
            from experiments.vldb.main_benchmark.tasks_data import predict_ridge
            p=predict_ridge(net,b[:,None,:])
        elif model=='dlinear':
            from experiments.vldb.downstream_models.dlinear import predict as pred
            p=pred(net,b[:,None,:],device='cuda')[:,:,0]
        elif model=='patchtst':
            from experiments.vldb.downstream_expansion.forecasting.runner import predict as pred
            p=pred(net,b,batch=BATCH)
        else:
            output=net.predict(b);p=output['point'];quantiles.append(output['quantiles'][:n])
        points.append(p[:n])
    result=dict(point=np.concatenate(points))
    if quantiles: result['quantiles']=np.concatenate(quantiles)
    if result['point'].shape!=(len(x),24) or not all(np.isfinite(v).all() for v in result.values()): raise RuntimeError('Invalid predictions')
    return result

def metric(y,p,raw,naive):
    y=np.asarray(y,dtype=np.float64);z=np.asarray(p['point'],dtype=np.float64);r=np.asarray(raw['point'],dtype=np.float64)
    errors=np.mean((z-y)**2,axis=1);re=np.mean((r-y)**2,axis=1);ne=np.mean((np.asarray(naive,dtype=np.float64)-y)**2,axis=1)
    good=re<ne
    out=dict(mse=float(errors.mean()),mae=float(np.abs(z-y).mean()),raw_mse=float(re.mean()),naive_mse=float(ne.mean()),
             prediction_mse=float(np.mean((z-r)**2)),raw_good_count=int(good.sum()),
             usefulness_failure_count=int((good&(errors>=ne)).sum()),
             catastrophic_count=int(((errors>=10*np.maximum(re,1e-12))&(errors>=ne)).sum()),
             origins=len(y),per_origin_mse=errors.tolist(),per_origin_raw_mse=re.tolist())
    if 'quantiles' in p:
        q=np.asarray(p['quantiles'],dtype=np.float64);err=y[:,:,None]-q;levels=np.arange(1,10)/10
        out.update(pinball=float(np.maximum(levels*err,(levels-1)*err).mean()),coverage80=float(((y>=q[:,:,0])&(y<=q[:,:,-1])).mean()),
                   width80=float((q[:,:,-1]-q[:,:,0]).mean()),quantile_crossing_fraction=float(np.any(np.diff(q,axis=-1)<0,axis=-1).mean()))
    return out

def output_path(model,seed,e,split,key): return OUT/'rows'/model/str(seed)/e['sid']/split/(key+'.json')
def checked_result(path,e=None,row=None):
    v=c.read(path)
    if v['consumer_manifest_sha256']!=c.sha(MANIFEST): raise RuntimeError('Saved consumer result belongs to another manifest')
    if e is not None and v['input_sha256']!=e['input_sha256']: raise RuntimeError('Saved consumer input differs')
    if v['compression_row'] is not None and c.sha(c.RUN/v['compression_row'])!=v['compression_row_sha256']: raise RuntimeError('Saved compression result changed')
    if row is not None and v['compression_row_sha256']!=c.sha(c.RUN/row['row_path']): raise RuntimeError('Resume codec row differs')
    if v['status']=='complete' and c.sha(c.ROOT/v['prediction'])!=v['prediction_sha256']: raise RuntimeError('Saved forecast prediction changed')
    return v

def npz_write(p,values):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+str(os.getpid())+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**values)
    tmp.replace(p)


@lru_cache(maxsize=None)
def verify_parent_runtime(model, seed):
    """Verify the immutable parent and matching execution semantics, not its label."""
    from experiments.vldb.baseline_expansion_20260922 import forecast_consumers as parent
    parent.verify()
    previous = c.read(c.PARENT_RUN/'evaluation/forecast/runtime'/(model+'__'+str(seed)+'.json'))
    current = c.read(OUT/'runtime'/(model+'__'+str(seed)+'.json'))
    if previous != current:
        raise RuntimeError('Parent forecast runtime differs; no cached prediction reuse')
    # Inference/preprocessing functions are retained verbatim in this adapter.
    import inspect
    for name in ('configure', 'load', 'predict', 'metric'):
        if inspect.getsource(globals()[name]) != inspect.getsource(getattr(parent, name)):
            raise RuntimeError('Parent inference function changed: '+name)
    return parent


def parent_prediction(model, seed, entry, split, key, row=None):
    parent = verify_parent_runtime(model, seed)
    is_simple = row is not None and row['method'] in c.SIMPLE_METHODS
    if is_simple:
        key=c.CANONICAL_METHOD[row['method']]+'__'+row['point']['label']
    path = parent.output_path(model, seed, entry, split, key)
    previous = parent.checked_result(path, entry)
    if previous['status'] != 'complete':
        raise RuntimeError('Parent prediction is not complete: '+str(path))
    binding = dict(result=rel(path), result_sha256=c.sha(path),
        consumer_manifest=rel(parent.MANIFEST), consumer_manifest_sha256=c.sha(parent.MANIFEST),
        prediction=previous['prediction'], prediction_sha256=previous['prediction_sha256'])
    if row is not None:
        old = c.read(c.PARENT_RUN/previous['compression_row'])
        fields=('point','raw_bytes','shape') if is_simple else ('method', 'point', 'artifact_sha256', 'reconstruction_sha256', 'artifact_bytes', 'raw_bytes', 'shape')
        for field in fields:
            if old[field] != row[field]:
                raise RuntimeError('Parent baseline compression differs: '+field)
        if is_simple:
            source_path=c.PARENT_RUN/old['reconstruction']
            if c.sha(source_path)!=old['reconstruction_sha256']:
                raise RuntimeError('Parent reconstruction changed')
            original=np.load(source_path,allow_pickle=False)
            current=c.decoded(row)
            if not np.array_equal(original.view(np.uint32),current.view(np.uint32)):
                return None
            binding.update(match_basis='bitwise_complete_reconstruction',
                parent_reconstruction_sha256=old['reconstruction_sha256'],
                current_reconstruction_sha256=row['reconstruction_sha256'],
                array_sha256=hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest())
        else:
            binding['match_basis']='identical_complete_baseline_stream_and_reconstruction'
        binding.update(compression_row=rel(c.PARENT_RUN/previous['compression_row']),
                       compression_row_sha256=previous['compression_row_sha256'])
    with np.load(c.ROOT/previous['prediction'], allow_pickle=False) as values:
        prediction = {k:values[k].copy() for k in values.files}
    return prediction, binding

def evaluate_entry(model,seed,net,e,split,partial):
    a=c.arrays(e);source=a['val_source' if split=='val' else 'source'];future=a['val_future' if split=='val' else 'future']
    codec_rows=c.rows(e,split,require_complete=not partial)
    panel=[('raw',None,source[:,0,-96:])]
    for row in codec_rows:
        key=row['method']+'__'+row['point']['label'];rp=c.RUN/row['row_path']
        binding=dict(row=row['row_path'],sha256=c.sha(rp),status=row['status'],reconstruction_sha256=row.get('reconstruction_sha256'))
        c.write(OUT/'registered_inputs'/e['sid']/split/(key+'.json'),binding,exclusive=True)
        panel.append((key,row,c.decoded(row)[:,0,-96:] if row['status']=='complete' else None))
    # Skip complete panels without loading/inferencing model inputs again.
    pending=[]
    for item in panel:
        path=output_path(model,seed,e,split,item[0])
        if path.exists(): checked_result(path,e,item[1])
        else: pending.append(item)
    if not pending: return dict(complete=0,existing=len(panel))
    reuse={}
    for key,row,x in pending:
        if row is not None and row['status']=='complete':
            cached=parent_prediction(model,seed,e,split,key,row)
            if cached is not None: reuse[key]=cached
    required={key for key,_,_ in pending if key not in reuse} | {'raw'}
    unique=[];by_hash={};indices={}
    for key,row,x in panel:
        if key not in required: continue
        if x is None: continue
        x=np.ascontiguousarray(x,dtype=np.float32);h=hashlib.sha256(x.tobytes()).hexdigest()
        if h not in by_hash: by_hash[h]=len(unique);unique.append(x)
        elif not np.array_equal(unique[by_hash[h]].view(np.uint32),x.view(np.uint32)): raise RuntimeError('Hash collision')
        indices[key]=by_hash[h]
    merged=np.concatenate(unique);prediction=predict(model,net,merged);n=len(source)
    pred={key:{k:v[index*n:(index+1)*n] for k,v in prediction.items()} for key,index in indices.items()}
    raw=pred['raw'];naive=np.repeat(source[:,0,-1:],24,axis=1)
    # A separately padded raw call verifies fixed-batch replay independently of companion methods.
    replay=predict(model,net,source[:,0,-96:])
    check={k:dict(bitwise=bool(np.array_equal(v,replay[k])),max_abs=float(np.max(np.abs(v.astype(np.float64)-replay[k])))) for k,v in raw.items()}
    if not all(q['bitwise'] for q in check.values()): raise RuntimeError('Raw forecast changed with batch companions: '+json.dumps(check))
    parent_raw,parent_raw_binding=parent_prediction(model,seed,e,split,'raw')
    parent_check={k:bool(np.array_equal(v,parent_raw[k])) for k,v in raw.items()}
    if set(raw)!=set(parent_raw) or not all(parent_check.values()):
        raise RuntimeError('Parent Raw forecast is not bitwise equal; baseline reuse refused')
    c.write(OUT/'raw_replays'/model/str(seed)/e['sid']/(split+'.json'),
        dict(check=check,passed=True,parent_check=parent_check,parent_raw=parent_raw_binding),exclusive=True)
    pred.update({key:value[0] for key,value in reuse.items()})
    count=0
    for key,row,x in pending:
        path=output_path(model,seed,e,split,key)
        result=dict(model=model,seed=seed,dataset=e['dataset'],sid=e['sid'],split=split,
            method='raw' if row is None else row['method'],point='raw' if row is None else row['point']['label'],
            input_sha256=e['input_sha256'],consumer_manifest_sha256=c.sha(MANIFEST),
            compression_row=None if row is None else row['row_path'],compression_row_sha256=None if row is None else c.sha(c.RUN/row['row_path']),
            artifact_bytes=None if row is None else row.get('artifact_bytes'),cr=1. if row is None else row.get('cr'),
            strict_audit=None if row is None else row.get('strict_audit'),numerical_audit=None if row is None else row.get('numerical_audit'),
            enabled_contract=None if row is None else row.get('enabled_contract'),
            enabled_pass_numerical=True if row is None else row.get('enabled_pass_numerical'),
            enabled_pass_strict=True if row is None else row.get('enabled_pass_strict'))
        if x is None: result.update(status='codec_failed',error=row.get('error'))
        else:
            if key in reuse:
                pp=c.ROOT/reuse[key][1]['prediction']
                result.update(prediction_origin='parent_identical_input_and_fresh_raw_replay',
                              parent_prediction_binding=reuse[key][1])
            else:
                pp=path.with_suffix('.npz');npz_write(pp,pred[key])
                result['prediction_origin']='fresh_fixed_consumer'
            result.update(status='complete',metrics=metric(future,pred[key],raw,naive),prediction=rel(pp),prediction_sha256=c.sha(pp))
        c.write(path,result,exclusive=True);count+=1
    return dict(written=count,unique_panels=len(unique),total_panels=len(panel),origins=n)

def evaluate(model,seed,split,partial=False,dataset=None):
    verify()
    if (model,seed) not in INSTANCES: raise ValueError('Unknown model/seed')
    if split=='test': verify_selections()
    lock=nullcontext() if model=='ridge' else c.gpu_lock()
    with lock:
        configure(model,seed);net=None;previous=None
        for e in c.entries('forecast'):
            if dataset and e['dataset']!=dataset: continue
            group=e['sid'] if model=='ridge' else e['dataset'] if model in ('dlinear','patchtst') else 'global'
            if group!=previous:
                net=None
                if model!='ridge':
                    import torch
                    torch.cuda.empty_cache()
                net=load(model,seed,e['dataset'],e['sid']);previous=group
            try: result=evaluate_entry(model,seed,net,e,split,partial)
            except Exception as exc:
                c.write(OUT/'errors'/model/str(seed)/e['sid']/(split+'.json'),dict(status='incomplete',error=str(exc),traceback=traceback.format_exc(),utc=c.now()))
                raise
            print(json.dumps(dict(model=model,seed=seed,sid=e['sid'],split=split,**result)),flush=True)
    c.write(OUT/'status'/(model+'__'+str(seed)+'__'+split+('__partial' if partial else '')+'.json'),dict(complete=True,partial=partial,dataset=dataset,utc=c.now()))

def family_result(model,e,split,key):
    paths=[output_path(model,s,e,split,key) for s in (c.SEEDS if model=='patchtst' else [0])]
    values=[checked_result(p,e) for p in paths]
    if len({(v['compression_row_sha256'],v['artifact_bytes']) for v in values})!=1: raise RuntimeError('Seeds do not share one identical codec configuration')
    value=dict(paths={rel(p):c.sha(p) for p in paths},status='complete' if all(v['status']=='complete' for v in values) else 'failed')
    if value['status']=='complete': value.update(mse=float(np.mean([v['metrics']['mse'] for v in values])),artifact_bytes=values[0]['artifact_bytes'],cr=values[0]['cr'],
        codec_eligible=all(v['enabled_pass_numerical'] is True or v['method']=='cameo' for v in values))
    return value

def select():
    verify()
    if SELECTION.exists(): return verify_selections()
    selections=[];all_files={}
    for e in c.entries('forecast'):
        for model in FAMILIES:
            raw=family_result(model,e,'val','raw');all_files.update(raw['paths'])
            if raw['status']!='complete': raise RuntimeError('Raw VAL unavailable')
            candidates={}
            for method in c.METHODS:
                for pt in c.points(e,method):
                    key=method+'__'+pt['label'];v=family_result(model,e,'val',key);all_files.update(v['paths'])
                    candidates[key]=dict(v,method=method,point=pt)
            fallback=candidates['zstd__lossless']
            if fallback['status']!='complete' or fallback['mse']!=raw['mse']: raise RuntimeError('Lossless fallback does not exactly reproduce raw')
            for method in c.METHODS:
                options=[method+'__'+p['label'] for p in c.points(e,method)]
                if 'zstd__lossless' not in options: options.append('zstd__lossless')
                for budget in c.PROTOCOL['quality_CR']['budgets']:
                    limit=(1+budget)*max(raw['mse'],1e-12)
                    eligible=[k for k in options if candidates[k]['status']=='complete' and candidates[k]['codec_eligible'] and candidates[k]['mse']<=limit]
                    selected=min(eligible,key=lambda k:(candidates[k]['artifact_bytes'],candidates[k]['point']['epsilon'],options.index(k)))
                    winner=candidates[selected]
                    selections.append(dict(dataset=e['dataset'],sid=e['sid'],model=model,method=method,budget=budget,
                        selected=selected,fallback=winner['method']=='zstd' and method!='zstd',raw_val_mse=raw['mse'],val_limit=limit,
                        val_mse=winner['mse'],val_bytes=winner['artifact_bytes'],candidates=options,eligible=eligible))
    value=dict(created_utc=c.now(),consumer_manifest_sha256=c.sha(MANIFEST),protocol=c.PROTOCOL['quality_CR'],
               val_result_sha256=all_files,selections=selections,count=len(selections))
    c.write(SELECTION,value,exclusive=True)
    return dict(count=len(selections),selection_sha256=c.sha(SELECTION))

def verify_selections():
    v=c.read(SELECTION)
    if v['consumer_manifest_sha256']!=c.sha(MANIFEST): raise RuntimeError('Consumer manifest changed')
    for p,h in v['val_result_sha256'].items():
        if c.sha(c.ROOT/p)!=h: raise RuntimeError('Selected VAL result changed: '+p)
    return dict(count=v['count'],selection_sha256=c.sha(SELECTION))

def check():
    y=np.array([[0.,2.],[2.,4.]])
    raw=dict(point=y.copy());p=dict(point=y+1)
    value=metric(y,p,raw,y)
    assert value['mse']==1 and value['raw_mse']==0 and value['mae']==1
    # Independent objective/tie fixture, with a failure and a feasible fallback.
    candidates=[dict(bytes=8,e=.5,mse=1.2),dict(bytes=9,e=.1,mse=1.05),dict(bytes=9,e=.25,mse=1.05),dict(bytes=20,e=0,mse=1.)]
    best=min((v for v in candidates if v['mse']<=1.1),key=lambda v:(v['bytes'],v['e']))
    assert best==candidates[1]
    print(json.dumps(dict(passed=True,checks=['MSE arithmetic','VAL eligibility','byte objective','stricter-epsilon tie'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate','select','check']);p.add_argument('--model',choices=FAMILIES);p.add_argument('--seed',type=int,default=0);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--partial',action='store_true');p.add_argument('--dataset')
    a=p.parse_args()
    if a.action=='freeze': print(json.dumps(freeze()))
    elif a.action=='select': print(json.dumps(select()))
    elif a.action=='check':check()
    else:evaluate(a.model,a.seed,a.split,a.partial,a.dataset)

if __name__=='__main__':main()
