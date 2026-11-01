"""UCR retrieval with fixed TRAIN relevance and frozen model reuse.

CPU-only, no training. Freeze before evaluate. v1/v2 model consumers stay separate.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CLASS = ROOT/'experiments/results/vldb_ucr_classification_expansion_20260915_v1'
OUT = ROOT/'experiments/results/vldb_ucr_retrieval_20260915_v1'
NAMES = ['ECG200','ArrowHead','GunPoint','Earthquakes','ShapeletSim','Meat']
METHODS = ['raw','point_only','pearson','acf','joint','sperr','sz2016']
SEEDS = [20260915,20260916,20260917]
FIELDS = ['AP','nDCG_at_10','Recall_at_10','Precision_at_10',
          'raw_top1_retained','raw_top10_overlap','raw_top10_exact_order']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if immutable and path.exists():
        if path.read_bytes()!=data: raise RuntimeError(f'immutable record differs: {path}')
        return
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(data); temporary.replace(path)


def save_npz(path, **values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)


def relative(path):
    return str(Path(path).relative_to(ROOT)).replace('\\','/')


def freeze():
    if (OUT/'protocol.json').exists():
        verify(); return
    upstream=read(CLASS/'freeze.json')
    entries={}
    for name in NAMES:
        entry=upstream['datasets'][name]
        path=CLASS/entry['prepared_file'].replace('\\','/')
        assert sha(path)==entry['prepared_sha256']
        entries[name]=dict(path=relative(path),sha256=sha(path),train=entry['train'],
                           test=entry['test'],length=entry['length'],classes=entry['classes'])
    assert sum(e['test'] for e in entries.values())==804
    protocol=dict(schema='nclab.ucr.retrieval.fixed-consumer.v1',datasets=entries,
        methods=METHODS,seeds=SEEDS,k=10,
        input_freeze_sha256=sha(CLASS/'freeze.json'),
        references=dict(ucr='https://doi.org/10.1109/JAS.2019.1911747',
            matrix_profile='https://doi.org/10.1109/ICDM.2016.0179',
            resnet='https://doi.org/10.1109/IJCNN.2017.7966039'),
        population='All 804 official TEST queries; full uncompressed official TRAIN dictionary; no deduplication or filtering by retrieval/model quality.',
        relevance='Binary: reference official TRAIN class equals query official TEST class. Labels only grade saved rankings, never train a retrieval model or choose its settings.',
        distance=dict(shape='float64 centered unit-L2 vectors; zero centered energy maps to zero vector when L2 norm<=1e-15; squared Euclidean distance',
            resnet='Frozen classifier global-average-pooling features before final logits; float32 CPU inference, cast features to float64 for squared Euclidean distance; no feature centering, scaling, or L2 normalization.',
            ties='ascending exact computed distance; stable original TRAIN row order; no tolerance or random tie breaking',
            nearest='all references; official TRAIN/TEST roles; no artificial self-exclusion between distinct partitions'),
        metrics=dict(AP='Full-ranking sum(rel_at_i*precision_at_i)/number_of_relevant_TRAIN_items; mAP averages queries.',
            nDCG_at_10='binary gain/log2(rank+1), divided by ideal sum through min(10,relevant_count)',
            Recall_at_10='relevant hits among first min(10,TRAIN_count), divided by ALL relevant TRAIN items; not divided by k',
            Precision_at_10='relevant hits divided by effective k',
            raw_top1_retained='same reference row identity as raw query',
            raw_top10_overlap='size of intersection with raw top-k divided by effective k, not semantic recall',
            raw_top10_exact_order='identical ordered first k reference indices',
            zero_relevance='error: every TEST class must have at least one original TRAIN reference',
            aggregation='Within dataset all queries equally weighted; across six datasets macro; micro also labeled; deep seeds average within dataset first. Version v1/v2 never selected by best score.'),
        consumers=dict(shape='always available',
            resnet_v1='classification v1 checkpoints, checkpoint global mean/std preprocessing',
            resnet_v2='separately registered standard-training version with exact classifier preprocessing and source identity before first v2 retrieval score; no retraining'),
        registration='New supplemental codec arrays and checkpoint files are registered with hashes before their first score. Missing inputs/models are pending, never scored as zero.',
        scope='Derived UCR label-relevance retrieval consumer, not an independent representation benchmark. Query compression bytes only; raw TRAIN/model/labels are unbilled. Existing UCR epsilon=1 panel, not the EDBT128 panel.',
        runtime=dict(device='cpu',threads=1,batch=64,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        source_files={relative(p):sha(p) for p in sorted(HERE.glob('*.py'))})
    write(OUT/'protocol.json',protocol,immutable=True)
    register_version('resnet_v1',CLASS,'global',CLASS/'model_protocol.json')
    print(json.dumps(dict(frozen=True,queries=804,protocol_sha256=sha(OUT/'protocol.json'))),flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for file,digest in p['source_files'].items():
        assert sha(ROOT/file)==digest, f'frozen retrieval code changed: {file}'
    for entry in p['datasets'].values():
        assert sha(ROOT/entry['path'])==entry['sha256'], 'prepared source changed'
    assert sha(CLASS/'freeze.json')==p['input_freeze_sha256']
    return p


def register_version(version,folder,normalization,model_protocol,epsilon=1e-8,ddof=0):
    assert version in ('resnet_v1','resnet_v2')
    folder=Path(folder).resolve(); model_protocol=Path(model_protocol).resolve()
    assert model_protocol.exists(), 'classification protocol must exist before registration'
    value=dict(version=version,result_folder=relative(folder),normalization=normalization,
        epsilon=float(epsilon),ddof=int(ddof),model_protocol=relative(model_protocol),
        model_protocol_sha256=sha(model_protocol),
        architecture_sha256=sha(HERE/'resnet_snapshot.py'),
        architecture_source='classification/models.py snapshot; declared PyTorch port, not original TensorFlow execution')
    write(OUT/'consumers'/f'{version}.json',value,immutable=True)


def arrays(name,protocol):
    entry=protocol['datasets'][name]
    with np.load(ROOT/entry['path']) as data:
        value={key:data[key] for key in data.files}
    for method in ('pearson','acf'):
        meta=CLASS/'compression'/name/f'{method}.json'
        if meta.exists():
            row=read(meta); path=CLASS/row['reconstruction_file'].replace('\\','/')
            assert sha(path)==row['reconstruction_sha256']
            write(OUT/'registered_inputs'/name/f'{method}.json',
                dict(metadata=relative(meta),metadata_sha256=sha(meta),
                     reconstruction=relative(path),reconstruction_sha256=sha(path)),immutable=True)
            value[method]=np.load(path)
    for method in METHODS:
        if method in value:
            assert value[method].shape==value['raw'].shape and np.isfinite(value[method]).all()
    assert len(value['raw'])==entry['test'] and len(value['train'])==entry['train']
    assert np.isfinite(value['train']).all()
    assert set(value['ytest']).issubset(set(value['ytrain']))
    return value


def unit(values):
    z=np.asarray(values,dtype=np.float64)
    z=z-z.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(z,axis=1,keepdims=True)
    return np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-15)


def ranking(query,reference):
    q=np.asarray(query,dtype=np.float64); r=np.asarray(reference,dtype=np.float64)
    assert q.ndim==r.ndim==2 and q.shape[1]==r.shape[1]
    distance=np.maximum(0.,(q*q).sum(1)[:,None]+(r*r).sum(1)[None,:]-2*q@r.T)
    assert np.isfinite(distance).all()
    return np.argsort(distance,axis=1,kind='stable')


def evaluate_ranking(order,yquery,yreference,raw_order):
    rel=(yreference[order]==yquery[:,None])
    total=(yreference[None,:]==yquery[:,None]).sum(1)
    assert np.all(total>0)
    nref=order.shape[1]; k=min(10,nref)
    precision=np.cumsum(rel,axis=1)/np.arange(1,nref+1)[None,:]
    AP=(precision*rel).sum(1)/total
    discounts=1/np.log2(np.arange(1,k+1)+1)
    ideal=np.array([discounts[:min(k,int(n))].sum() for n in total])
    ndcg=(rel[:,:k]*discounts).sum(1)/ideal
    hits=rel[:,:k].sum(1)
    overlap=np.array([len(set(a[:k])&set(b[:k]))/k for a,b in zip(order,raw_order)])
    return dict(AP=AP,nDCG_at_10=ndcg,Recall_at_10=hits/total,Precision_at_10=hits/k,
        raw_top1_retained=(order[:,0]==raw_order[:,0]).astype(float),raw_top10_overlap=overlap,
        raw_top10_exact_order=np.all(order[:,:k]==raw_order[:,:k],axis=1).astype(float))


def model_class():
    spec=importlib.util.spec_from_file_location('nclab_retrieval_resnet_snapshot',HERE/'resnet_snapshot.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.ResNetTSC


def normalize(values,checkpoint,consumer):
    if consumer['normalization']=='global':
        std=float(checkpoint['std']); assert std>0
        return np.ascontiguousarray((values-float(checkpoint['mean']))/std,dtype=np.float32)
    if consumer['normalization']=='per_series_z':
        z=np.asarray(values,dtype=np.float64)
        mean=z.mean(axis=1,keepdims=True); std=z.std(axis=1,keepdims=True,ddof=consumer['ddof'])
        # The registered hfawaz v2 reference replaces EXACT zero std only.
        bad=(std==0) if consumer['epsilon']==0 else (std<consumer['epsilon'])
        std=np.where(bad,1.,std)
        return np.ascontiguousarray((z-mean)/std,dtype=np.float32)
    raise ValueError('unsupported registered preprocessing')


@torch.no_grad()
def embeddings(values,model,checkpoint,consumer):
    model.eval(); z=normalize(values,checkpoint,consumer)
    return np.concatenate([model.blocks(batch).mean(-1).cpu().numpy()
        for batch in torch.from_numpy(z[:,None,:]).split(64)])


def one(name,version,seed,protocol):
    data=arrays(name,protocol)
    available=[m for m in METHODS if m in data]
    destination=OUT/'rows'/version/name/str(seed)
    destination.mkdir(parents=True,exist_ok=True)
    consumer_identity=dict(version=version,seed=seed)
    model=None
    if version=='shape':
        features=lambda a:unit(a)
    else:
        config_path=OUT/'consumers'/f'{version}.json'
        if not config_path.exists(): return dict(dataset=name,consumer=version,seed=seed,status='pending_consumer_registration')
        consumer=read(config_path)
        assert sha(ROOT/consumer['model_protocol'])==consumer['model_protocol_sha256']
        path=ROOT/consumer['result_folder']/'resnet'/name/str(seed)/'model.pt'
        if not path.exists() or not path.with_name('training.json').exists():
            return dict(dataset=name,consumer=version,seed=seed,status='pending_checkpoint')
        identity=dict(checkpoint=relative(path),checkpoint_sha256=sha(path),consumer_sha256=sha(config_path))
        write(OUT/'registered_models'/version/name/f'{seed}.json',identity,immutable=True)
        if all((destination/f'{m}.json').exists() for m in available):
            for method in available:
                row=read(destination/f'{method}.json')
                assert row['consumer_identity']==dict(version=version,seed=seed,**identity)
                assert sha(destination/row['per_query_file'])==row['per_query_sha256']
            return read(destination/'coverage.json')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert sha(path)==identity['checkpoint_sha256'], 'checkpoint changed while loading'
        assert int(checkpoint['seed'])==seed
        if version=='resnet_v2':
            assert checkpoint['normalization']=='per_query_population_znorm_float64_to_float32'
            assert consumer['normalization']=='per_series_z' and consumer['epsilon']==0 and consumer['ddof']==0
        architecture_sources={k:v for k,v in checkpoint['code_identity'].items() if k.split('/')[-1]=='models.py'}
        assert list(architecture_sources.values())==[sha(HERE/'resnet_snapshot.py')], 'architecture snapshot mismatch'
        model=model_class()(checkpoint['classes']); model.load_state_dict(checkpoint['state_dict']); model.eval()
        consumer_identity.update(identity)
        features=lambda a:embeddings(a,model,checkpoint,consumer)
    # References and raw rankings are evaluated once with the same frozen consumer.
    train_features=features(data['train']); raw_features=features(data['raw'])
    raw_order=ranking(raw_features,train_features)
    for method in available:
        path=destination/f'{method}.json'
        if path.exists():
            row=read(path)
            assert row['consumer_identity']==consumer_identity
            assert sha(destination/row['per_query_file'])==row['per_query_sha256']
            continue
        query=raw_features if method=='raw' else features(data[method])
        order=raw_order if method=='raw' else ranking(query,train_features)
        per_query=evaluate_ranking(order,data['ytest'],data['ytrain'],raw_order)
        array_path=destination/f'{method}.npz'
        save_npz(array_path,**per_query,ranking=order,yquery=data['ytest'],yreference=data['ytrain'])
        write(path,dict(dataset=name,consumer=version,seed=seed,method=method,queries=len(data['raw']),
            references=len(data['train']),effective_k=min(10,len(data['train'])),status='complete',
            metrics={k:float(v.mean()) for k,v in per_query.items()},
            consumer_identity=consumer_identity,protocol_sha256=sha(OUT/'protocol.json'),
            per_query_file=array_path.name,per_query_sha256=sha(array_path)),immutable=True)
    coverage=dict(dataset=name,consumer=version,seed=seed,expected_methods=METHODS,
        completed_methods=[m for m in METHODS if (destination/f'{m}.json').exists()],
        pending_methods=[m for m in METHODS if not (destination/f'{m}.json').exists()])
    coverage['status']='complete' if not coverage['pending_methods'] else 'partial_inputs_pending'
    write(destination/'coverage.json',coverage)
    del model
    return coverage


def summarize():
    rows=[]
    for path in (OUT/'rows').glob('*/*/*/*.json'):
        if path.name!='coverage.json': rows.append(read(path))
    groups=[]
    for consumer in ['shape','resnet_v1','resnet_v2']:
        seeds=[0] if consumer=='shape' else SEEDS
        for method in METHODS:
            values=[r for r in rows if r['consumer']==consumer and r['method']==method]
            expected=len(NAMES)*len(seeds)
            group=dict(consumer=consumer,method=method,completed=len(values),expected=expected,
                       status='complete' if len(values)==expected else 'pending',metrics=None)
            if len(values)==expected:
                group['metrics']={field:float(np.mean([np.mean([r['metrics'][field] for r in values if r['dataset']==n]) for n in NAMES])) for field in FIELDS}
                group['micro_metrics']={field:float(sum(r['queries']*r['metrics'][field] for r in values)/(804*len(seeds))) for field in FIELDS}
            groups.append(group)
    write(OUT/'summary.json',dict(protocol_sha256=sha(OUT/'protocol.json'),groups=groups,rows=rows))
    lines=['# UCR检索效用：固定任务与版本分开报告','',
        '全部6集/804个TEST查询；完整原始TRAIN库。相同官方类别作为relevance。这里的mAP/nDCG/Recall是标签相关质量；raw-top10 overlap是原始结果保持，两者不混同。',
        'ResNet为已训练分类器的GAP嵌入派生消费者，不另训练，不是独立表示学习benchmark。v1与标准训练v2分别完整报告，不取较好版本。缺失模型/压缩结果保持pending。','',
        '|消费者|压缩|覆盖|mAP|nDCG@10|Recall@10|原始top10 overlap|','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['metrics']; numbers=[f"{m[k]:.6f}" for k in ['AP','nDCG_at_10','Recall_at_10','raw_top10_overlap']] if m else ['pending']*4
        lines.append('|'+ '|'.join([g['consumer'],g['method'],f"{g['completed']}/{g['expected']}",*numbers])+'|')
    lines+=['','表中为数据集等权宏平均，深度模型先在每个数据集内平均3个固定seed；micro与逐数据集结果保存在summary.json。',
        'Recall@10分母为该查询类别在TRAIN中的全部相关条目数，所以即使前10个全正确，类别参考数大于10时该值也不会等于1。',
        '传统shape距离等价于非退化序列的Pearson排序。ResNet嵌入不额外L2归一化，使用对应分类模型自己的冻结输入预处理。',
        '本报告不把邻居改变直接称为不可用，也不把跨模型版本或不同压缩率结果当作ACF单独因果证据。']
    (OUT/'SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(rows=len(rows),complete_groups=sum(g['status']=='complete' for g in groups),groups=len(groups))


def metric_check():
    # Hand-computable: relevant items at ranks 1 and 3 gives AP=(1+2/3)/2.
    order=np.array([[0,1,2]]); yq=np.array([1]); yr=np.array([1,0,1])
    result=evaluate_ranking(order,yq,yr,order)
    assert abs(result['AP'][0]-5/6)<1e-15 and result['Recall_at_10'][0]==1
    assert abs(result['nDCG_at_10'][0]-(1+.5)/(1+1/np.log2(3)))<1e-15
    assert result['raw_top10_overlap'][0]==1
    assert np.array_equal(ranking(np.array([[1.,0.]]),np.array([[1.,0.],[1.,0.],[0.,1.]])),np.array([[0,1,2]]))
    assert np.array_equal(unit(np.ones((2,5))),np.zeros((2,5)))
    return dict(passed=True,checks=['full-ranking AP denominator','nDCG ideal','semantic Recall denominator','stable ties','degenerate shape'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','check','evaluate','summarize','register-v2'])
    parser.add_argument('--consumer',choices=['all','shape','resnet_v1','resnet_v2'],default='all')
    parser.add_argument('--dataset',choices=NAMES); parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--folder'); parser.add_argument('--model-protocol')
    parser.add_argument('--normalization',choices=['global','per_series_z'])
    parser.add_argument('--epsilon',type=float,default=1e-8); parser.add_argument('--ddof',type=int,default=0)
    args=parser.parse_args(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if args.stage=='check': print(json.dumps(metric_check())); return
    if args.stage=='freeze': freeze(); return
    protocol=verify()
    if args.stage=='register-v2':
        assert args.folder and args.model_protocol and args.normalization
        register_version('resnet_v2',args.folder,args.normalization,args.model_protocol,args.epsilon,args.ddof); return
    if args.stage=='summarize': print(json.dumps(summarize())); return
    consumers=['shape','resnet_v1','resnet_v2'] if args.consumer=='all' else [args.consumer]
    with open('/tmp/nclab_retrieval_cpu_20260915.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for consumer in consumers:
            seeds=[0] if consumer=='shape' else ([args.seed] if args.seed else SEEDS)
            for name in ([args.dataset] if args.dataset else NAMES):
                for seed in seeds:
                    print(json.dumps(one(name,consumer,seed,protocol)),flush=True)
        print(json.dumps(summarize()),flush=True)


if __name__=='__main__': main()
