"""Real three-source consumer smoke, isolated from production scientific rows.

This harness only permits partial codec rosters inside this process. Production
consumers/common.py remain unchanged. Original complete input arrays are retained.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import traceback

NAME = 'vldb_framework_full_consumer_smoke_20261015_v1'
os.environ['JOINTZIP_FULL_RUN_NAME'] = NAME
receipt = Path('/opt/jointzip-home/.local/cth_expansion_runs/vldb_framework_pilot_20261014_v1/runtime/COPY_RECEIPT.json')
if receipt.exists():
    os.environ.update(json.loads(receipt.read_text())['environment'])

from . import common as c


def prepare():
    parent = c.read(c.PARENT_RUN/'freeze.json')
    roster = parent['entries']
    entries = [next(e for e in roster if e['task']=='forecast' and e['sid']=='ETTm1__000'),
               next(e for e in roster if e['task']=='ucr' and e['sid']=='ECG200'),
               next(e for e in roster if e['task']=='anomaly' and e['dataset']=='MSL')]
    for entry in entries:
        src,dst = c.PARENT_RUN/entry['input'],c.RUN/entry['input']
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copyfile(src,dst)
        if c.sha(dst)!=entry['input_sha256']: raise RuntimeError('Smoke source differs')
    c.write(c.RUN/'freeze.json',dict(protocol=c.PROTOCOL,entries=entries,
            smoke_only=True,scientific_results=False,parent_freeze_sha256=c.sha(c.PARENT_RUN/'freeze.json')),
            exclusive=True)
    return {'entries':[e['sid'] for e in entries],'full_original_arrays':True}


def compress():
    from .compress import one
    results=[]
    for entry in c.entries():
        point = c.STRESS if entry['task']=='ucr' else c.POINTS[0]
        if entry['task']!='ucr' and point['label']!='strict': raise RuntimeError('Strict order changed')
        split = 'val' if entry['task']=='forecast' else 'test'
        for framework in (False,True):
            value=one((entry,split,point,'sz3',framework))
            print(json.dumps(value),flush=True);results.append(value)
    return {'conditions':results,'real_framework_search':True}


def reframe_nasa():
    """Repair only the smoke's pre-fix outer true-count header, preserving it."""
    from . import container
    import numpy as np
    changes=[]
    for entry in c.entries('anomaly'):
        for row in c.rows(entry,'test',require_complete=False):
            path=c.RUN/row['row_path']; artifact=c.RUN/row['artifact']
            meta,payload=container.unframe(artifact.read_bytes())
            if meta['true_count']==entry['true_length']: continue
            history=c.RUN/'smoke_reframe_history'/path.stem
            history.mkdir(parents=True,exist_ok=True)
            if not (history/'row.json').exists(): shutil.copyfile(path,history/'row.json')
            if not (history/'artifact.jzf').exists(): shutil.copyfile(artifact,history/'artifact.jzf')
            framed,metadata=container.frame(payload,entry,'test',tuple(row['shape']))
            decoded,_=container.decode(framed,c.RUN/'work/smoke_reframe'/path.stem)
            if decoded.tobytes()!=c.decoded(row).tobytes(): raise RuntimeError('Header-only reframe changed values')
            artifact.write_bytes(framed)
            row.pop('row_path',None)
            row.update(raw_bytes=int(entry['true_length'])*4,artifact_bytes=len(framed),
                       artifact_sha256=c.sha(artifact),container_metadata=metadata,
                       smoke_header_repair=dict(prior=str(history.relative_to(c.RUN)),only_outer_true_count=True))
            row['cr']=row['raw_bytes']/row['artifact_bytes']
            c.write(path,row);changes.append(str(path.relative_to(c.RUN)))
    return dict(corrected_rows=changes,scientific_reencoding=False)


def summarize():
    expected=['prepare__0','compress__0','reframe_nasa__0','forecast__ridge__0',
              'forecast__dlinear__0','forecast__timesfm3__0','ucr__one_nn__0',
              'ucr__resnet_v2__20260915','anomaly__ridge_svd64__0',
              'anomaly__knn__0','anomaly__lstm__20260915']
    stages={}
    for name in expected:
        path=c.RUN/'SMOKE_STAGES'/(name+'.json')
        if not path.exists() or not c.read(path).get('complete'): raise RuntimeError('Missing smoke stage '+name)
        stages[name]=c.sha(path)
    rows=[]
    for pattern in ('evaluation/forecast/rows/*/*/*/*/*.json','evaluation/ucr/classification/*/*/*/*.json',
                    'evaluation/ucr/retrieval/*/*/*/*.json','evaluation/anomaly/rows/*/*/*.json'):
        rows.extend(c.RUN.glob(pattern))
    results=[c.read(p) for p in rows]
    if len(results)!=30 or any(r.get('status')!='complete' for r in results):
        raise RuntimeError('Expected 30 complete smoke consumer rows')
    from collections import Counter
    value=dict(complete=True,smoke_only=True,main_experiment_evidence=False,stages=stages,
        consumers=len(results),tasks=dict(Counter(r['task'] for r in results)),
        origins=dict(Counter(r['prediction_origin'] for r in results)),
        compression_conditions=6,real_framework_search=True,
        note='Three full original source panels; selected frozen consumers. No scientific production result rows are included.')
    c.write(c.RUN/'SMOKE_COMPLETE.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','compress','reframe_nasa','forecast','ucr','anomaly','summarize'])
    parser.add_argument('--model');parser.add_argument('--seed',type=int,default=0)
    a=parser.parse_args()
    label=a.stage+('__'+a.model if a.model else '')+'__'+str(a.seed)
    try:
        if a.stage=='prepare': result=prepare()
        elif a.stage=='compress': result=compress()
        elif a.stage=='reframe_nasa': result=reframe_nasa()
        elif a.stage=='summarize': result=summarize()
        else:
            # Only this isolated smoke harness accepts its deliberately partial
            # codec roster. It cannot write into the production result root.
            original_rows=c.rows
            c.rows=lambda entry,split,require_complete=True: original_rows(entry,split,require_complete=False)
            if a.stage=='forecast':
                from .consumer_forecast import run
                run(a.model,a.seed,'val')
            elif a.stage=='ucr':
                from .consumer_ucr import run
                run(a.model,a.seed)
            else:
                from .consumer_anomaly import run
                run(a.model,a.seed)
            result={'consumer_stage':a.stage,'model':a.model,'seed':a.seed}
        result.update(complete=True,smoke_only=True,utc=c.now())
        c.write(c.RUN/'SMOKE_STAGES'/(label+'.json'),result)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        c.write(c.RUN/'SMOKE_FAILURES'/(label+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),utc=c.now()))
        raise


if __name__=='__main__':main()
