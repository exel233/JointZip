"""Same frozen NASA scores/TSB metrics, exact native reuse, no retraining."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
from . import common as c
from . import consumer_common as h
from experiments.vldb.paper_budget_guided_20261013_v1 import anomaly_consumers as old

OUT = c.RUN/'evaluation/anomaly'
INSTANCES = [('ridge_svd64',0),('knn',0)]+[('lstm',s) for s in c.SEEDS]
PARENT_MANIFEST = c.PARENT_RUN/'evaluation/anomaly/protocol.json'


def output(entry,model,seed,key):
    return OUT/'rows'/entry['sid']/(model+'_'+str(seed))/(key+'.json')


def parent_result(entry,model,seed,key,rec,labels,threshold,period,allow):
    if key is None or not allow:
        return None
    key = 'raw__raw' if key == 'raw' else key
    path = c.PARENT_RUN/'evaluation/anomaly/rows'/entry['sid']/(model+'_'+str(seed))/(key+'.json')
    if not path.exists():
        return None
    record,values,lineage = h.parent_values(path,'scores','scores_sha256')
    if record['protocol_sha256'] != c.sha(PARENT_MANIFEST) or record['input_sha256'] != entry['input_sha256']:
        raise RuntimeError('Historical anomaly input/protocol changed')
    if record['vus_window'] != period or not h.bits(values['labels'],labels):
        raise RuntimeError('Historical anomaly labels/VUS window changed')
    if not h.bits(values['threshold'],np.asarray(threshold)):
        return None
    if key != 'raw__raw':
        previous, cp = h.parent_reconstruction(entry,record)
        if not h.bits(previous.ravel()[:len(rec)],rec):
            return None
        lineage.update(compression=cp,match_basis='bitwise_decoded_true_suffix')
    return values,record['metrics'],lineage


def replay(entry,bound,spec,fn,data):
    z,labels = data['original'],data['labels']
    tr,cut = int(data['train_end']),int(data['fit_end'])
    full = fn(z); threshold = float(np.quantile(full[cut:tr],.99))
    rec = np.load(h.checked(bound['reconstructions']['strict__qpet_joint']['array']),allow_pickle=False).ravel()[:len(z)-tr]
    reference = fn(np.concatenate([z[:tr],rec]))[tr:]
    model,seed = spec['model'],spec['seed']; checks = {}
    for key,scores in [('raw__raw',full[tr:]),('qpet_joint__strict',reference)]:
        path = c.PARENT_RUN/'evaluation/anomaly/rows'/entry['sid']/(model+'_'+str(seed))/(key+'.json')
        record,values,lineage = h.parent_values(path,'scores','scores_sha256')
        if record['protocol_sha256'] != c.sha(PARENT_MANIFEST) or not h.bits(values['labels'],labels[tr:]):
            raise RuntimeError('Historical anomaly replay identity changed')
        checks[key] = dict(bitwise=h.bits(scores,values['scores']) and h.bits(np.asarray(threshold),values['threshold']),
                           close=bool(np.allclose(scores,values['scores'],atol=old.ATOL,rtol=old.RTOL))
                           and bool(np.isclose(threshold,values['threshold'],atol=old.ATOL,rtol=old.RTOL)),
                           parent=lineage)
    if not all(v['close'] for v in checks.values()):
        raise RuntimeError('Frozen NASA raw/reference replay exceeded tolerance')
    value = dict(passed=True,cache_reuse_allowed=all(v['bitwise'] for v in checks.values()),checks=checks,threshold=threshold)
    path = OUT/'raw_replays'/entry['sid']/(model+'_'+str(seed)+'.json')
    c.write(path,value,exclusive=True)
    return full[tr:],threshold,value,h.binding(path)


def evaluate_entry(entry,bound,model,seed,manifest_sha):
    data = c.arrays(entry)
    if entry['input_sha256'] != bound['input_sha256']:
        raise RuntimeError('NASA input differs from frozen consumer')
    z,labels = data['original'],data['labels']; tr = int(data['train_end'])
    labels = labels[tr:]; period = bound['period']
    if old.array_sha(labels) != bound['labels_sha256']:
        raise RuntimeError('NASA labels differ')
    panels = [None,*c.rows(entry,'test')]
    pending = [(r,h.base(entry,r,manifest_sha,'anomaly',model,seed)) for r in panels]
    pending = [(r,b) for r,b in pending if not h.finished(output(entry,model,seed,h.key(r)),b)]
    if not pending:
        return dict(written=0,existing=len(panels))
    imported=0; remaining=[]
    for row,result in pending:
        rec=z[tr:] if row is None else c.decoded(row).ravel()[:len(z)-tr]
        if rec.shape!=z[tr:].shape or not np.isfinite(rec).all():
            raise RuntimeError('Invalid decoded true NASA suffix')
        path=output(entry,model,seed,h.key(row))
        if h.import_completed(entry,row,result,path,'anomaly',model,seed,'test',rec): imported+=1
        else: remaining.append((row,result))
    pending=remaining
    if not pending:
        return dict(written=imported,completed_framework_reuse=imported,raw_and_reference_replays=0)
    spec = bound['consumers'][model+'_'+str(seed)]
    if spec.get('model_metadata'):
        h.checked(spec['model_metadata'])
    fn = old._load(spec,data)
    raw,threshold,check,replay_binding = replay(entry,bound,spec,fn,data)
    cache = {}; written = reused = 0
    for row,result in pending:
        if row is not None and row['status'] != 'complete':
            raise RuntimeError('Cannot score incomplete NASA compression')
        rec = z[tr:] if row is None else c.decoded(row).ravel()[:len(z)-tr]
        if rec.shape != z[tr:].shape or not np.isfinite(rec).all():
            raise RuntimeError('Invalid decoded true NASA suffix')
        sha = h.digest(rec)
        old_result = parent_result(entry,model,seed,h.historical_key(row),rec,labels,threshold,period,check['cache_reuse_allowed'])
        if old_result is not None:
            values,metrics,lineage = old_result
            origin = 'historical_identical_input_after_raw_and_reference_replay'; reused += 1
        elif sha in cache:
            values,metrics,previous = cache[sha]
            origin,lineage = 'current_identical_consumer_input',{'current_prediction':previous}
        else:
            scores = raw if row is None or h.bits(rec,z[tr:]) else fn(np.concatenate([z[:tr],rec]))[tr:]
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise RuntimeError('Nonfinite/misaligned NASA scores')
            values = dict(scores=scores,labels=labels,threshold=np.asarray(threshold))
            # Preserve expensive scores before computing the official VUS metric.
            h.save_npz(output(entry,model,seed,h.key(row)).with_suffix('.scored.npz'),values)
            metrics = old._metrics_for(values,period)
            origin,lineage = 'fresh_fixed_consumer_and_official_TSB_metrics',{}
        result.update(consumer=model+'_'+str(seed),consumer_input_sha256=sha,
                      threshold=threshold,vus_window=period,test_points=len(labels),
                      decoded_true_suffix_sha256=old.array_sha(rec),labels_sha256=old.array_sha(labels),
                      score_array_sha256=old.array_sha(values['scores']),raw_replay=replay_binding)
        path = output(entry,model,seed,h.key(row))
        h.persist(path,result,values,metrics,origin,lineage)
        cache[sha] = (values,metrics,h.binding(path)); written += 1
    return dict(written=written+imported,completed_framework_reuse=imported,
                historical_reuse=reused,raw_and_reference_replays=2)


def run(model,seed=0,dataset=None):
    if (model,seed) not in INSTANCES:
        raise ValueError('Unknown anomaly model/seed')
    parent = c.read(PARENT_MANIFEST)
    h.ensure_parent_sources(parent)
    if parent['runtime'] != old._runtime_identity():
        raise RuntimeError('Frozen NASA consumer runtime differs')
    _,manifest_sha = h.freeze('anomaly',PARENT_MANIFEST,[Path(old.__file__)])
    bounds = {e['sid']:e for e in parent['entries']}
    lock = c.gpu_lock() if model == 'lstm' else nullcontext()
    with lock:
        old._runtime()
        for entry in c.entries('anomaly'):
            if dataset and entry['dataset'] != dataset and entry['sid'] != dataset:
                continue
            result = evaluate_entry(entry,bounds[entry['sid']],model,seed,manifest_sha)
            print(json.dumps(dict(model=model,seed=seed,sid=entry['sid'],**result)),flush=True)
    if dataset is None:
        c.write(OUT/'status'/f'{model}__{seed}.json',dict(complete=True,model=model,seed=seed,consumer_manifest_sha256=manifest_sha))
