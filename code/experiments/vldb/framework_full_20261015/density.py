"""Predeclared validation-only parameter-density diagnostic.

Six source domains, first series and first four inherited VAL contexts. The
main experiment remains at its frozen default budget. This diagnostic extends
the same proposal prefix to 44/88; no TEST information chooses a parameter.
Identical backend calls plus corrections are reused across prefixes/policies,
and their incremental wall time is explicitly not standalone encoder latency.
"""
from __future__ import annotations
import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import common as c
from . import codec,container

OUT=c.RUN/'density'
DATASETS=('ETTm1','ETTm2','ElecDem','Weather','Wind','Solar')
BACKENDS=('sz3','zfp')
POLICIES=('formula','noformula')
BUDGETS=(22,44,88)
DLINEAR=c.ROOT/'experiments/results/vldb_edbt_dlinear_20260907_v1'


def selected_entries():
    values=c.entries('forecast')
    return [min((e for e in values if e['dataset']==name),key=lambda e:e['sid']) for name in DATASETS]


def _rowpath(entry,point,backend,policy,budget):
    return OUT/'rows'/entry['sid']/point['label']/backend/policy/(str(budget)+'.json')


def freeze():
    entries=selected_entries()
    files=[Path(__file__),Path(codec.__file__),Path(container.__file__)]
    for entry in entries:
        files.extend([c.MAIN/'models'/entry['dataset']/(entry['sid']+'.json'),
                      DLINEAR/'models'/(entry['dataset']+'.json'),
                      DLINEAR/'models'/(entry['dataset']+'.npz')])
    value=dict(schema='jointzip.density.val.v1',parent_freeze_sha256=c.sha(c.RUN/'freeze.json'),
        code_and_models={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(set(files))},
        sources=[dict(sid=e['sid'],dataset=e['dataset'],input_sha256=e['input_sha256'],
                      val_indices=[0,1,2,3]) for e in entries],
        backends=BACKENDS,policies=POLICIES,budgets=BUDGETS,initial=12,
        points=[p for p in c.POINTS if p['label'] in ('strict','loose')],expected_conditions=144,
        consumer='Frozen Raw-trained Ridge/DLinear; fresh deterministic CPU inference; final96/future24; padded batch128; no training.',
        reuse='Identical cap/source/backend/constraint candidates reused; prefixes and each policy observations remain independent.',
        timing='Incremental work with candidate reuse and I/O; NOT standalone policy latency or kernel throughput.',
        scope='VAL-only first4contexts/source, not representative fullTEST or proof that formula alone causes improvement; main policy unchanged.')
    if (OUT/'PLAN.json').exists():
        if c.read(OUT/'PLAN.json')!=json.loads(json.dumps(value)):
            raise RuntimeError('Density plan/code/model changed')
    else:c.write(OUT/'PLAN.json',value,exclusive=True)
    return entries,value


@lru_cache(maxsize=6)
def _models(dataset,sid):
    from experiments.vldb.downstream_models import dlinear
    dlinear.configure_runtime('cpu')
    ridge=c.read(c.MAIN/'models'/dataset/(sid+'.json'))
    net,_=dlinear.load_model(DLINEAR,dataset,device='cpu')
    return ridge,net


def score(entry,source,future):
    from experiments.vldb.main_benchmark.tasks_data import predict_ridge
    from experiments.vldb.downstream_models import dlinear
    ridge,net=_models(entry['dataset'],entry['sid'])
    x=np.ascontiguousarray(source[:,:,-96:],dtype=np.float32)
    if len(x)<128:x=np.concatenate([x,np.repeat(x[:1],128-len(x),axis=0)])
    predictions={'ridge':predict_ridge(ridge,x)[:len(source)],
                 'dlinear':dlinear.predict(net,x,device='cpu')[:len(source),:,0]}
    if any(y.shape!=future.shape or not np.isfinite(y).all() for y in predictions.values()):
        raise RuntimeError('Density consumer output invalid')
    return {model:dict(mse=float(np.mean((y.astype(float)-future.astype(float))**2)),
                       mae=float(np.mean(np.abs(y.astype(float)-future.astype(float)))))
            for model,y in predictions.items()}


def _reuse_main_default(entry,point,backend,source,candidates,workdir):
    """Reuse completed main formula22 windows; unselected probes were not saved.

    Selected lossy candidates can also seed later probe caches, but are never
    presented as the complete old candidate set for expanded budgets.
    """
    rowpath=c.row_path(entry,'val',backend+'_jointzip',point['label'])
    if not rowpath.exists():return None
    row=c.read(rowpath)
    if row['status']!='complete':return None
    if row['input_sha256']!=entry['input_sha256'] or row['point']!=point or row['backend']!=backend or not row['framework']:
        raise RuntimeError('Main density reuse binding differs')
    for name in ('artifact','metadata','reconstruction'):
        if c.sha(c.RUN/row[name])!=row[name+'_sha256']:
            raise RuntimeError('Main density reuse artifact changed: '+name)
    metadata=c.read(c.RUN/row['metadata'])
    options=codec.search_options(backend,True)
    if metadata['search_options']!=options or metadata['identity']['binary_sha256']!=codec.identity(backend)['binary_sha256']:
        raise RuntimeError('Main default search cannot seed current density policy')
    _,inner=container.unframe((c.RUN/row['artifact']).read_bytes())
    header,records=codec.unframe(inner)
    if header['codec']!=backend or header['shape'][0]<len(source) or header['shape'][1:]!=tuple(source.shape[1:]):
        raise RuntimeError('Main density reuse source dimensions differ')
    records=records[:len(source)];evidence=metadata['windows'][:len(source)]
    provenance=dict(kind='completed_main_formula_default_subset',row=str(rowpath.relative_to(c.RUN)),
                    row_sha256=c.sha(rowpath),indices=list(range(len(source))),
                    note='selected payloads and complete default search evidence reused; unselected candidate payloads not available')
    windows=codec.WindowCache()
    windows.add_records(backend,source,point['epsilon'],point['tau'],point['rho'],16,True,
                        records,evidence,provenance,workdir=workdir)
    for index,row_source in enumerate(source[:,0,:]):
        saved=windows.get(backend,row_source,point['epsilon'],point['tau'],point['rho'],16,True,options)
        selected=dict(saved['record'])
        if selected['kind']!=1:continue
        selected.update({k:v for k,v in evidence[index]['selected'].items()
                         if k not in ('base_bytes','correction_bytes','base','correction','rebuilt')})
        selected['eligible']=True
        key=np.ldexp(1.,int(np.floor(np.log2(selected['cap'])))) if backend=='zfp' else selected['cap']
        candidates.put(backend,row_source,point['epsilon'],point['tau'],point['rho'],16,key,selected)
    return windows


def run():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        entries,plan=freeze();completed=0
        for entry in entries:
            arrays=c.arrays(entry)
            source=np.ascontiguousarray(arrays['val_source'][:4],dtype=np.float32)
            future=arrays['val_future'][:4]
            if source.shape!=(4,1,128) or future.shape!=(4,24):
                raise RuntimeError('Density source differs from declared panel')
            rawpath=OUT/'raw'/(entry['sid']+'.json')
            if rawpath.exists():raw=c.read(rawpath)
            else:
                raw=dict(sid=entry['sid'],input_sha256=entry['input_sha256'],metrics=score(entry,source,future))
                c.write(rawpath,raw,exclusive=True)
            for point in plan['points']:
                for backend in BACKENDS:
                    cachepath=OUT/'candidate_cache'/entry['sid']/point['label']/(backend+'.json')
                    cache=codec.CandidateCache.load(cachepath) if cachepath.exists() else codec.CandidateCache()
                    for policy in POLICIES:
                        previous_caps=None
                        for budget in BUDGETS:
                            resultpath=_rowpath(entry,point,backend,policy,budget)
                            name='__'.join((entry['sid'],point['label'],backend,policy,str(budget)))
                            directory=OUT/'artifacts'/name
                            directory.mkdir(parents=True,exist_ok=True)
                            checkpoint=directory/'compression.json'
                            c.write(OUT/'STATUS.json',dict(phase='density',current=name,completed=completed,total=144,updated_utc=c.now()))
                            if resultpath.exists():
                                row=c.read(resultpath)
                                if row['status']!='complete' or c.sha(c.RUN/row['artifact'])!=row['artifact_sha256'] or c.sha(c.RUN/row['reconstruction'])!=row['reconstruction_sha256']:
                                    raise RuntimeError('Completed density artifact changed')
                                previous_caps=row['caps'];completed+=1;continue
                            if checkpoint.exists():
                                comp=c.read(checkpoint)
                                if c.sha(directory/'stream.jzf')!=comp['artifact_sha256'] or c.sha(directory/'reconstruction.npy')!=comp['reconstruction_sha256']:
                                    raise RuntimeError('Density compressed checkpoint changed')
                                rebuilt=np.load(directory/'reconstruction.npy',allow_pickle=False)
                                metadata=comp['metadata']
                            else:
                                started=time.perf_counter()
                                reused=(_reuse_main_default(entry,point,backend,source,cache,directory/'main_reuse_decode')
                                        if policy=='formula' and budget==22 else None)
                                encoded=codec.encode(backend,source,point['epsilon'],point['tau'],point['rho'],16,
                                    True,directory/'work',search_policy=policy,candidate_budget=budget,candidate_cache=cache,
                                    warm_cache=reused,
                                    window_callback=lambda *_:cache.save(cachepath))
                                cache.save(cachepath)
                                stream,frame_meta=container.frame(encoded.payload,entry,'val',source.shape)
                                (directory/'stream.jzf').write_bytes(stream)
                                rebuilt,_=container.decode(stream,directory/'independent_decode')
                                if not np.array_equal(rebuilt.view(np.uint32),encoded.reconstruction.view(np.uint32)):
                                    raise RuntimeError('Density independent reconstruction changed')
                                np.save(directory/'reconstruction.npy',rebuilt,allow_pickle=False)
                                metadata=encoded.metadata
                                comp=dict(metadata=metadata,artifact_sha256=c.sha(directory/'stream.jzf'),
                                    reconstruction_sha256=c.sha(directory/'reconstruction.npy'),bytes=len(stream),
                                    raw_bytes=source.nbytes,CR=source.nbytes/len(stream),frame_metadata=frame_meta,
                                    incremental_wall_seconds=time.perf_counter()-started,
                                    timing_scope='encode/reuse/framing/independentdecode/I/O; candidate reuse makes this incremental, not standalone')
                                c.write(checkpoint,comp,exclusive=True)
                            caps=[[candidate['cap'] for candidate in item['candidates'] if candidate['kind']==1]
                                  for item in metadata['windows']]
                            if previous_caps is not None and any(new[:len(old)]!=old for old,new in zip(previous_caps,caps)):
                                raise RuntimeError('Larger density budget failed to preserve previous prefix')
                            previous_caps=caps
                            metrics=score(entry,rebuilt,future)
                            row=dict(status='complete',sid=entry['sid'],dataset=entry['dataset'],point=point,
                                backend=backend,policy=policy,budget=budget,split='val',indices=[0,1,2,3],
                                input_sha256=entry['input_sha256'],bytes=comp['bytes'],raw_bytes=source.nbytes,CR=comp['CR'],
                                artifact=str((directory/'stream.jzf').relative_to(c.RUN)),artifact_sha256=comp['artifact_sha256'],
                                reconstruction=str((directory/'reconstruction.npy').relative_to(c.RUN)),reconstruction_sha256=comp['reconstruction_sha256'],
                                metadata=str(checkpoint.relative_to(c.RUN)),counts=metadata['counts'],audit=metadata['audit'],caps=caps,
                                encoded_seconds=metadata['encode_seconds'],incremental_wall_seconds=comp['incremental_wall_seconds'],
                                metrics=metrics,raw_metrics=raw['metrics'],completed_utc=c.now())
                            c.write(resultpath,row,exclusive=True);completed+=1
                            print(json.dumps(dict(event='density_complete',condition=name,CR=row['CR'],completed=completed,total=144)),flush=True)
        if completed!=144:raise RuntimeError('Incomplete density conditions')
        summarize()
        c.write(OUT/'COMPLETE.json',dict(complete=True,conditions=completed,plan_sha256=c.sha(OUT/'PLAN.json'),completed_utc=c.now()))
        c.write(OUT/'STATUS.json',dict(phase='complete',completed=completed,total=144,updated_utc=c.now()))


def summarize():
    rows=[c.read(p) for p in sorted((OUT/'rows').rglob('*.json'))]
    if len(rows)!=144 or any(r['status']!='complete' for r in rows):raise RuntimeError('Cannot summarize incomplete density panel')
    flattened=[]
    for r in rows:
        flat={k:r[k] for k in ('dataset','sid','backend','policy','budget','bytes','raw_bytes','CR','encoded_seconds','incremental_wall_seconds')}
        flat['point']=r['point']['label'];flat.update(r['counts'])
        flat.update(joint_pass_windows=r['audit']['numerical_pass_windows'],windows=r['audit']['windows'])
        for model in ('ridge','dlinear'):
            flat[model+'_mse']=r['metrics'][model]['mse'];flat['raw_'+model+'_mse']=r['raw_metrics'][model]['mse']
        flattened.append(flat)
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    indexed={(r['sid'],r['point']['label'],r['backend'],r['policy'],r['budget']):r for r in rows}
    lines=['# 参数搜索密度消融（仅验证集）','',
        '6 个来源各取首条序列的前 4 个既有 VAL 窗口；SZ3/ZFP，strict/loose，公式/无公式，预算 22/44/88。共 144 条配置。',
        '后端、原 simple correction、初始 12 个候选和搜索规则固定；只延长后续细化。两策略保留各自观测历史。',
        '同一候选的压缩和修复可以复用；CSV 中耗时是新增工作量，不能拿它比较独立运行速度。',
        '公式22默认结果优先从已完成主实验取相同前4窗重封装；44/88仅能复用已保存的相同候选，不把默认版选中码流当成所有候选。',
        'Ridge/DLinear 使用已有 Raw 训练权重、相同原始未来值和新 CPU 推理；这里只用于诊断，不替代主实验 GPU 成绩。','',
        '| 策略 | 后端 | 预算 | 条件数 | 完整 CR / 同策略22几何均值 | CR改善/持平/下降 | Ridge MSE改善/持平/下降 | DLinear MSE改善/持平/下降 |',
        '|---|---|---:|---:|---:|---|---|---|']
    for policy in POLICIES:
        for backend in BACKENDS:
            for budget in BUDGETS:
                selected=[r for r in rows if (r['policy'],r['backend'],r['budget'])==(policy,backend,budget)]
                ratios=[];cr=[0,0,0];metrics={m:[0,0,0] for m in ('ridge','dlinear')}
                for r in selected:
                    base=indexed[(r['sid'],r['point']['label'],backend,policy,22)]
                    ratios.append(r['CR']/base['CR'])
                    cr[0 if r['bytes']<base['bytes'] else 1 if r['bytes']==base['bytes'] else 2]+=1
                    for m in metrics:
                        delta=r['metrics'][m]['mse']-base['metrics'][m]['mse']
                        metrics[m][1 if abs(delta)<=1e-12 else 0 if delta<0 else 2]+=1
                fmt=lambda v:'/'.join(map(str,v))
                lines.append(f"| {policy} | {backend} | {budget} | {len(selected)} | {np.exp(np.mean(np.log(ratios))):.6f} | {fmt(cr)} | {fmt(metrics['ridge'])} | {fmt(metrics['dlinear'])} |")
    lines.extend(['','完整流外层 Zstd 可能改变不同窗口组合的字节排序，因此即使候选集合嵌套，也逐条报告最终文件大小。',
                  '不依据这 4 个 VAL 窗口改变大实验方法。完整逐配置数据见 `results.csv`；候选轨迹和持久化码流在 `artifacts/`。'])
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if args.report_only:summarize()
    else:run()
