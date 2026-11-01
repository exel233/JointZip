"""Exact ablation roster closeout, without rehashing/replaying all payloads."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

from . import common as c
from .consumers import expected_rows

EXPECTED_INPUTS = dict(forecast=142, ucr=6, anomaly=24)
EXPECTED_CONSUMERS = dict(forecast=206752, classification=1488, retrieval=1302, anomaly=10920)
EXPECTED_COMPRESSION = 27900


def compression_expected():
    for entry in c.entries():
        for split in c.splits(entry):
            for method in c.METHODS:
                for point in c.points(entry, method):
                    from .report import method_parts
                    backend, arm = method_parts(method)
                    yield c.row_path(entry, split, method, point['label']), dict(
                        task=entry['task'], sid=entry['sid'], dataset=entry['dataset'], split=split,
                        method=method, backend=backend, arm=arm, point=point,
                        input_sha256=entry['input_sha256'], enabled_contract=c.contract(arm))


def _relative(path, root):
    return Path(path).relative_to(root).as_posix()


def _within(root, value):
    root=Path(root).resolve()
    path=(root/value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact path escapes result root')
    return path


def _nonempty(path):
    return path.is_file() and path.stat().st_size>0


def _count(audit, key, windows):
    value=audit[key]
    if isinstance(value, bool) or not isinstance(value, int) or not 0<=value<=windows:
        raise ValueError('invalid audit count '+key)
    return value


def _verify_audits(row):
    windows=row['shape'][0]*row['shape'][1]
    if windows<=0:
        raise ValueError('empty compression input')
    contract=row['enabled_contract']
    for name,suffix in (('strict_audit','strict'),('numerical_audit','numerical')):
        audit=row[name]
        if audit['series_windows']!=windows:
            raise ValueError('full ERA audit coverage differs')
        counts={field:_count(audit, field, windows) for field in
                ('point_pass_count','pearson_pass_count','acf_pass_count','joint_pass_count')}
        enabled=(counts['point_pass_count']==windows
                 and ('R' not in contract or counts['pearson_pass_count']==windows)
                 and ('A' not in contract or counts['acf_pass_count']==windows))
        if row.get('enabled_pass_'+suffix) is not enabled:
            raise ValueError('enabled pass flag differs from actual enabled component audits')
        active=row['enabled_audit_'+suffix]
        if active['series_windows']!=windows:
            raise ValueError('enabled audit coverage differs')
        active_pass=_count(active,'joint_pass_count',windows)==windows
        if bool(active['joint_pass']) is not active_pass or active_pass is not enabled:
            raise ValueError('enabled audit joint status differs')
        if suffix=='numerical' and not enabled:
            raise ValueError('final enabled contract failed')
        for field in ('max_abs_error','pearson_min','acf_max_abs','mse'):
            if not isinstance(audit[field],(float,int)) or not math.isfinite(audit[field]):
                raise ValueError('nonfinite audit statistic '+field)


def check(*, partial=False):
    inputs=dict(Counter(e['task'] for e in c.entries()))
    compression=dict(expected=0,complete=0,missing=[],errors=[],unexpected=[])
    planned=set(); by_task=Counter(); by_method=Counter(); reuse=Counter()
    for path,identity in compression_expected():
        relative=_relative(path,c.RUN); planned.add(relative); compression['expected']+=1
        if not path.exists():
            compression['missing'].append(relative); continue
        try:
            row=c.read(path)
            for field,value in identity.items():
                if row.get(field)!=value:
                    raise ValueError('condition identity differs: '+field)
            if row.get('status')!='complete' or row.get('framework') is not True:
                raise ValueError('not a complete framework ablation result')
            if row.get('independent_decode_bitwise') is not True:
                raise ValueError('saved stream has no recorded independent replay')
            for field in ('artifact','reconstruction','metadata'):
                if not _nonempty(_within(c.RUN,row[field])):
                    raise ValueError('committed '+field+' missing or empty')
            size=_within(c.RUN,row['artifact']).stat().st_size
            if size!=row['artifact_bytes'] or row['raw_bytes']<=0:
                raise ValueError('invalid complete-file byte accounting')
            if not math.isclose(row['cr'],row['raw_bytes']/size,rel_tol=1e-12):
                raise ValueError('complete-file CR arithmetic differs')
            _verify_audits(row)
            if row['arm']=='ERA':
                if row.get('reuse',{}).get('kind')!='completed_current_ERA':
                    raise ValueError('existing ERA result was not reused')
            if row['arm']=='ERA_no_correction':
                ablation=row.get('correction_ablation',{})
                if ablation.get('fixed_logical_caps') is not True or ablation.get('repair_disabled') is not True:
                    raise ValueError('correction comparison is not fixed-pool repair-disabled')
                if not ablation.get('trace_row') or not ablation.get('trace_row_sha256'):
                    raise ValueError('fixed correction candidate trace is unbound')
                metadata=c.read(c.RUN/row['metadata'])
                trace_sha=ablation.get('trace_sha256')
                if (not trace_sha or metadata.get('trace_sha256')!=trace_sha
                        or any(w.get('trace_sha256')!=trace_sha for w in metadata['windows'])):
                    raise ValueError('no-correction selected windows do not bind the fixed candidate trace')
                selected=[w['selected'] for w in metadata['windows']]
                if len(selected)!=row['shape'][0]*row['shape'][1]:
                    raise ValueError('no-correction selected-window coverage differs')
                if any(s['count']!=0 or s['correction_bytes']!=0 for s in selected):
                    raise ValueError('repair data found in no-correction output')
            compression['complete']+=1; by_task[row['task']]+=1; by_method[row['method']]+=1
            reuse[row.get('reuse',{}).get('kind','new')]+=1
        except Exception as exc:
            compression['errors'].append(dict(path=relative,error=str(exc)))
    actual={_relative(p,c.RUN) for p in (c.RUN/'compression').glob('*/*/*/*.json')}
    compression.update(unexpected=sorted(actual-planned),by_task=dict(by_task),by_method=dict(by_method),reuse=dict(reuse))

    consumers=dict(expected=0,complete=0,missing=[],errors=[],unexpected=[])
    names=set(); expected_task=Counter(); complete_task=Counter(); origins=Counter()
    for path,identity in expected_rows():
        relative=_relative(path,c.RUN); names.add(relative); consumers['expected']+=1
        expected_task[identity['task']]+=1
        if not path.exists():
            consumers['missing'].append(relative); continue
        try:
            row=c.read(path)
            for field,value in identity.items():
                if row.get(field)!=value:
                    raise ValueError('consumer identity differs: '+field)
            if row.get('status')!='complete':
                raise ValueError('consumer is not complete')
            if not _nonempty(_within(c.ROOT,row['prediction'])):
                raise ValueError('consumer output arrays missing or empty')
            if row['method']!='raw':
                linked=_within(c.RUN,row['compression_row'])
                if _relative(linked,c.RUN.resolve()) not in planned or not linked.is_file():
                    raise ValueError('consumer references missing/unplanned compression row')
                if row.get('enabled_pass_numerical') is not True:
                    raise ValueError('consumer used a failed enabled contract')
            primary={'forecast':'mse','classification':'accuracy','retrieval':'AP','anomaly':'AP'}[row['task']]
            for metric in ([primary,'VUS_PR'] if row['task']=='anomaly' else [primary]):
                value=row['metrics'][metric]
                if not isinstance(value,(float,int)) or not math.isfinite(value):
                    raise ValueError('missing/nonfinite metric '+metric)
            consumers['complete']+=1; complete_task[row['task']]+=1
            origins[row.get('prediction_origin','unspecified')]+=1
        except Exception as exc:
            consumers['errors'].append(dict(path=relative,error=str(exc)))
    from .report import consumer_paths
    actual_consumers={_relative(p,c.RUN) for p in consumer_paths()}
    consumers.update(unexpected=sorted(actual_consumers-names),expected_by_task=dict(expected_task),
                     complete_by_task=dict(complete_task),prediction_origins=dict(origins))
    pending_errors=sorted(_relative(p,c.RUN) for p in (c.RUN/'evaluation/errors').glob('*.json'))
    expected_by_method={m:930 for m in c.METHODS}
    roster_ok=(inputs==EXPECTED_INPUTS and compression['expected']==EXPECTED_COMPRESSION
               and consumers['expected']==sum(EXPECTED_CONSUMERS.values())
               and dict(expected_task)==EXPECTED_CONSUMERS)
    passed=(roster_ok and dict(by_method)==expected_by_method and not pending_errors
            and all(not section[key] for section in (compression,consumers) for key in ('missing','errors','unexpected')))
    result=dict(schema='jointzip.framework-ablation.closeout.v1',passed=passed,partial=not passed,
        checked_utc=c.now(),inputs=inputs,expected_inputs=EXPECTED_INPUTS,rosters_match=roster_ok,
        compression=compression,consumers=consumers,unresolved_consumer_error_files=pending_errors,
        verification='Exact planned identities and coverage; committed output existence and sizes; enabled constraints checked against component audit counts; recorded independent replay. No repeated full-tree hashes.',
        scope='Six fixed backends, E/ER/EA/ERA and fixed-ERA-pool no-correction. No new timing or search-density experiment. Export is a separate delivery gate.')
    c.write(c.RUN/'audit/closeout/CLOSEOUT.json',result)
    if not passed and not partial:
        raise RuntimeError('Incomplete framework ablation; see audit/closeout/CLOSEOUT.json')
    return dict(passed=passed,compression_complete=compression['complete'],compression_expected=compression['expected'],
                consumers_complete=consumers['complete'],consumers_expected=consumers['expected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true')
    print(check(partial=parser.parse_args().partial),flush=True)
