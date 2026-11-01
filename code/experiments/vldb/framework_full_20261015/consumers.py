"""Entry point for all frozen fixed-backend downstream consumers.

python -m experiments.vldb.framework_full_20261015.consumers forecast --model ridge --split val
python -m experiments.vldb.framework_full_20261015.consumers ucr --consumer resnet_v2 --seed 20260915
python -m experiments.vldb.framework_full_20261015.consumers anomaly --model lstm --seed 20260915
"""
from __future__ import annotations
import argparse
import json
import traceback
from . import common as c
from . import consumer_common as h

FORECAST_FAMILIES = ('ridge','dlinear','patchtst','chronos2','chronos_bolt_small','timesfm3')


def expected_rows():
    """Yield exact path and stable identity; no model imports or computation."""
    for entry in c.entries():
        task = entry['task']; sid = entry['sid']
        for split in (('val','test') if task == 'forecast' else ('test',)):
            roster = [dict(method=m,point=p) for m in c.METHODS for p in c.points(entry,m)]
            for row in [None,*roster]:
                key = h.key(row); method = 'raw' if row is None else row['method']
                point = 'raw' if row is None else row['point']['label']
                if task == 'forecast':
                    specs = [('forecast',m,s) for m in FORECAST_FAMILIES for s in (c.SEEDS if m == 'patchtst' else [0])]
                elif task == 'ucr':
                    specs = [(t,m,s) for t,models in (
                        ('classification',('one_nn','logistic','resnet_v1','resnet_v2')),
                        ('retrieval',('shape','resnet_v1','resnet_v2')))
                        for m in models for s in (c.SEEDS if m.startswith('resnet') else [0])]
                elif task == 'anomaly':
                    specs = [('anomaly','ridge_svd64',0),('anomaly','knn',0)]+[('anomaly','lstm',s) for s in c.SEEDS]
                else:
                    raise ValueError('Unknown task '+task)
                for result_task,model,seed in specs:
                    if task == 'forecast':
                        path = c.RUN/'evaluation/forecast/rows'/model/str(seed)/sid/split/(key+'.json')
                    elif task == 'ucr':
                        path = c.RUN/'evaluation/ucr'/result_task/model/sid/str(seed)/(key+'.json')
                    else:
                        path = c.RUN/'evaluation/anomaly/rows'/sid/(model+'_'+str(seed))/(key+'.json')
                    yield path, dict(task=result_task,model=model,seed=seed,sid=sid,split=split,
                                     method=method,point=point,input_sha256=entry['input_sha256'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task',choices=['forecast','ucr','anomaly','expected'])
    p.add_argument('--model'); p.add_argument('--consumer'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--dataset')
    args = p.parse_args()
    try:
        if args.task == 'forecast':
            from .consumer_forecast import run
            run(args.model,args.seed,args.split,args.dataset)
        elif args.task == 'ucr':
            from .consumer_ucr import run
            run(args.consumer or args.model,args.seed,args.dataset)
        elif args.task == 'anomaly':
            from .consumer_anomaly import run
            run(args.model,args.seed,args.dataset)
        else:
            from collections import Counter
            print(json.dumps(dict(Counter(v['task'] for _,v in expected_rows()))))
    except Exception as exc:
        error = dict(status='failed',arguments=vars(args),error=repr(exc),traceback=traceback.format_exc(),utc=c.now())
        name = f'{args.task}__{args.model or args.consumer or "all"}__{args.seed}__{args.split}.json'
        c.write(c.RUN/'evaluation/errors'/name,error)
        raise


if __name__ == '__main__':
    main()
