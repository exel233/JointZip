"""Frozen UCR classifiers and derived retrieval; reuse exact native outputs."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
from . import common as c
from . import consumer_common as h
from experiments.vldb.paper_budget_guided_20261013_v1 import ucr_consumers as old

OUT = c.RUN/'evaluation/ucr'


def output(task, model, entry, seed, key):
    return OUT/task/model/entry['sid']/str(seed)/(key+'.json')


def parent_result(task, model, entry, seed, row, query):
    key = h.historical_key(row)
    if key is None:
        return None
    path = c.PARENT_RUN/'evaluation/ucr'/task/model/entry['sid']/str(seed)/(key+'.json')
    if not path.exists():
        return None
    record, values, lineage = h.parent_values(path,'per_query_file','per_query_sha256')
    if record['consumer_manifest_sha256'] != c.sha(old.PARENT_MANIFEST):
        raise RuntimeError('Parent UCR manifest binding changed')
    if (record['task'],record['consumer'],record['sid'],record['seed']) != (task,model,entry['sid'],seed):
        raise RuntimeError('Parent UCR row identity differs')
    if row is not None:
        previous, cp = h.parent_reconstruction(entry,record)
        if not h.bits(previous[:,0,:],query):
            return None
        lineage.update(compression=cp,match_basis='bitwise_full_query_array')
    return values, record['metrics'], lineage


def deep_predictor(entry, version, seed, device):
    import torch
    from experiments.vldb.downstream_expansion.classification.models import ResNetTSC
    folder = old.model_folder(entry['sid'],version,seed)
    checkpoint = torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    if int(checkpoint['seed']) != seed:
        raise RuntimeError('ResNet checkpoint seed differs')
    code = [v for k,v in checkpoint['code_identity'].items() if k.split('/')[-1] == 'models.py']
    if code != [c.sha(old.ARCH)]:
        raise RuntimeError('ResNet architecture differs')
    if version == 'resnet_v2' and checkpoint['normalization'] != 'per_query_population_znorm_float64_to_float32':
        raise RuntimeError('ResNet v2 normalization differs')
    normalization = dict(normalization='global' if version == 'resnet_v1' else 'per_series_z',epsilon=0.,ddof=0)
    model = ResNetTSC(checkpoint['classes']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()

    @torch.inference_mode()
    def predict(values):
        z = old.rm.normalize(values,checkpoint,normalization)
        features, probabilities = [], []
        for batch in torch.from_numpy(z[:,None,:]).split(64):
            feature = model.blocks(batch.to(device)).mean(-1)
            probabilities.append(model.classifier(feature).softmax(-1).cpu().numpy())
            features.append(feature.cpu().numpy())
        return np.concatenate(features), np.concatenate(probabilities)
    return predict, checkpoint['classes']


def evaluate_entry(entry, model, seed, manifest_sha, device):
    data = c.arrays(entry); raw, y, yt = data['raw'],data['ytest'],data['ytrain']
    tasks = [('classification',model)]
    if model != 'logistic':
        tasks.append(('retrieval','shape' if model == 'one_nn' else model))
    panels = [None,*c.rows(entry,'test')]
    pending = []
    for row in panels:
        todo = []
        for task,name in tasks:
            result = h.base(entry,row,manifest_sha,task,name,seed)
            if not h.finished(output(task,name,entry,seed,h.key(row)),result):
                todo.append((task,name,result))
        if todo:
            pending.append((row,todo))
    if not pending:
        return dict(written=0,existing=len(panels)*len(tasks))
    classes = len(data['class_labels'])
    if model == 'one_nn':
        archive = old.rm.unit(data['train'])
        def predict(q):
            return dict(prediction=old.shape_prediction(q,archive,yt),
                        ranking=old.rm.ranking(old.rm.unit(q),archive))
    elif model == 'logistic':
        linear,mean,std = old.logistic_model(entry['sid'])
        def predict(q):
            probability = linear.predict_proba(old.cm.normalized(q,mean,std))
            return dict(probability=probability,prediction=probability.argmax(1))
    else:
        predictor,classes = deep_predictor(entry,model,seed,device)
        archive,_ = predictor(data['train'])
        def predict(q):
            feature,probability = predictor(q)
            return dict(probability=probability,prediction=probability.argmax(1),
                        ranking=old.rm.ranking(feature,archive))
    raw_output = predict(raw)
    # Pure read helper: verifies all original raw classifier/retrieval arrays.
    checks = old.check_parent_raw(entry,model,seed,probability=raw_output.get('probability'),
                                 prediction=raw_output['prediction'],order=raw_output.get('ranking'))
    replay_path = OUT/'raw_replays'/model/entry['sid']/(str(seed)+'.json')
    c.write(replay_path,dict(passed=True,checks=checks),exclusive=True)
    current = {h.digest(raw):raw_output}; written = reused = 0
    for row,todo in pending:
        if row is not None and row['status'] != 'complete':
            raise RuntimeError('Incomplete UCR compression row')
        query = raw if row is None else c.decoded(row)[:,0,:]
        if query.shape != raw.shape or not np.isfinite(query).all():
            raise RuntimeError('Invalid UCR reconstruction')
        cached = {task:parent_result(task,name,entry,seed,row,query) for task,name,_ in todo}
        need_prediction = any(value is None for value in cached.values())
        sha = h.digest(query)
        already = sha in current
        if need_prediction and not already:
            current[sha] = predict(query)
        for task,name,result in todo:
            result.update(consumer_input_sha256=sha, queries=len(query))
            if cached[task] is not None:
                values,metrics,lineage = cached[task]
                if not h.bits(values['yquery'],y):
                    raise RuntimeError('Historical UCR labels changed')
                origin = 'historical_identical_input_after_raw_replay'; reused += 1
            else:
                p = current[sha]
                if task == 'classification':
                    metrics = old.cm.summary(y,p['prediction'],p.get('probability'),raw_output['prediction'],classes)
                    values = dict(yquery=y,prediction=p['prediction'],raw_prediction=raw_output['prediction'])
                    if 'probability' in p:
                        values['probability'] = p['probability']
                else:
                    values = old.rm.evaluate_ranking(p['ranking'],y,yt,raw_output['ranking'])
                    metrics = {k:float(v.mean()) for k,v in values.items()}
                    values.update(ranking=p['ranking'],yquery=y,yreference=yt)
                origin = 'current_identical_consumer_input' if already else 'fresh_fixed_consumer'
                lineage = {'raw_replay':h.binding(replay_path)}
            h.persist(output(task,name,entry,seed,h.key(row)),result,values,metrics,origin,lineage)
            written += 1
    return dict(written=written,historical_reuse=reused,raw_validation_panels=1)


def run(consumer, seed=0, dataset=None):
    model = consumer.removeprefix('retrieval_')
    if model == 'shape':
        model = 'one_nn'
    if model not in old.CONSUMERS or seed not in (c.SEEDS if model.startswith('resnet') else [0]):
        raise ValueError('Unknown UCR model/seed')
    parent = c.read(old.PARENT_MANIFEST); h.ensure_parent_sources(parent,'sources')
    _,manifest_sha = h.freeze('ucr',old.PARENT_MANIFEST,[Path(old.__file__)])
    device = parent['runtime']['device']
    lock = c.gpu_lock() if model.startswith('resnet') and device == 'cuda' else nullcontext()
    with lock:
        actual = old.configure(device)
        if actual != parent['runtime']:
            raise RuntimeError('Frozen UCR device/package/batch runtime differs')
        c.write(OUT/'runtime.json',actual,exclusive=True)
        for entry in c.entries('ucr'):
            if dataset and entry['dataset'] != dataset:
                continue
            result = evaluate_entry(entry,model,seed,manifest_sha,device)
            print(json.dumps(dict(model=model,seed=seed,sid=entry['sid'],**result)),flush=True)
    if dataset is None:
        c.write(OUT/'status'/f'{model}__{seed}.json',dict(complete=True,consumer=model,seed=seed,consumer_manifest_sha256=manifest_sha))
