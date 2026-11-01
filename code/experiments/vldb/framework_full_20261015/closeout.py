"""Exact planned-roster closeout without repeating complete-tree checksums."""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path

from . import common as c
from .consumers import expected_rows


def compression_expected():
    for entry in c.entries():
        for split in c.splits(entry):
            for method in c.METHODS:
                for point in c.points(entry,method):
                    yield c.row_path(entry,split,method,point['label']), dict(
                        task=entry['task'],sid=entry['sid'],dataset=entry['dataset'],split=split,
                        method=method,point=point,input_sha256=entry['input_sha256'])


def _within(root, value):
    path=(root/value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes result root')
    return path


def _nonempty(path):
    return path.is_file() and path.stat().st_size>0


def check(*, partial=False):
    entries=c.entries()
    input_count=dict(Counter(e['task'] for e in entries))
    expected_input=dict(forecast=142,ucr=6,anomaly=24)
    compression=dict(expected=0,complete=0,missing=[],errors=[],unexpected=[])
    compression_names=set(); by_task=Counter(); by_method=Counter()
    # Payloads were independently decoded when committed. This final check binds
    # exact identities and actual sizes; it does not hash/decode every payload again.
    for path,identity in compression_expected():
        relative=str(path.relative_to(c.RUN));compression_names.add(relative)
        compression['expected']+=1
        if not path.exists():
            compression['missing'].append(relative);continue
        try:
            row=c.read(path)
            for field,value in identity.items():
                if row.get(field)!=value:
                    raise ValueError('condition identity differs: '+field)
            if row.get('status')!='complete':
                raise ValueError('status is not complete')
            if row.get('independent_decode_bitwise') is not True:
                raise ValueError('saved stream was not independently replayed')
            for field in ('artifact','reconstruction','metadata'):
                target=_within(c.RUN,row[field])
                if not _nonempty(target):
                    raise ValueError('committed '+field+' is missing or empty')
            actual=_within(c.RUN,row['artifact']).stat().st_size
            if actual!=row['artifact_bytes']:
                raise ValueError('complete-file size differs')
            if row['raw_bytes']<=0 or not math.isclose(row['cr'],row['raw_bytes']/actual,rel_tol=1e-12):
                raise ValueError('complete-file CR arithmetic differs')
            if row.get('framework') is True:
                if row.get('enabled_contract')!='ERA' or row.get('enabled_pass_numerical') is not True:
                    raise ValueError('JointZip final enabled contract failed')
            elif row.get('enabled_contract')!='E':
                raise ValueError('Native contract label differs')
            for name in ('strict_audit','numerical_audit'):
                audit=row[name]
                if audit['series_windows']!=row['shape'][0]*row['shape'][1]:
                    raise ValueError('constraint coverage differs: '+name)
            compression['complete']+=1;by_task[row['task']]+=1;by_method[row['method']]+=1
        except Exception as exc:
            compression['errors'].append(dict(path=relative,error=str(exc)))
    actual_compression={str(p.relative_to(c.RUN)) for p in (c.RUN/'compression').glob('*/*/*/*.json')}
    compression['unexpected']=sorted(actual_compression-compression_names)
    compression['by_task']=dict(by_task);compression['by_method']=dict(by_method)

    consumers=dict(expected=0,complete=0,missing=[],errors=[],unexpected=[])
    names=set();expected_task=Counter();completed_task=Counter();origins=Counter()
    for path,identity in expected_rows():
        relative=str(path.relative_to(c.RUN));names.add(relative)
        consumers['expected']+=1;expected_task[identity['task']]+=1
        if not path.exists():
            consumers['missing'].append(relative);continue
        try:
            row=c.read(path)
            for field,value in identity.items():
                if row.get(field)!=value:
                    raise ValueError('consumer identity differs: '+field)
            if row.get('status')!='complete':
                raise ValueError('consumer status is not complete')
            values=_within(c.ROOT,row['prediction'])
            if not _nonempty(values):
                raise ValueError('consumer output arrays missing or empty')
            if row['method']!='raw':
                compression_row=_within(c.RUN,row['compression_row'])
                if str(compression_row.relative_to(c.RUN.resolve())) not in compression_names:
                    # Windows and POSIX relative paths keep the host convention.
                    if str(compression_row.relative_to(c.RUN.resolve())).replace('\\','/') not in {
                            n.replace('\\','/') for n in compression_names}:
                        raise ValueError('consumer references an unplanned compression row')
                if not compression_row.exists():
                    raise ValueError('consumer referenced compression missing')
            primary={'forecast':'mse','classification':'accuracy','retrieval':'AP','anomaly':'AP'}[row['task']]
            value=row['metrics'][primary]
            if not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError('missing/nonfinite primary metric')
            if row['task']=='anomaly' and not math.isfinite(row['metrics']['VUS_PR']):
                raise ValueError('missing/nonfinite VUS-PR')
            consumers['complete']+=1;completed_task[row['task']]+=1;origins[row['prediction_origin']]+=1
        except Exception as exc:
            consumers['errors'].append(dict(path=relative,error=str(exc)))
    from .report import consumer_paths
    actual_consumers={str(p.relative_to(c.RUN)) for p in consumer_paths()}
    consumers['unexpected']=sorted(actual_consumers-names)
    consumers.update(expected_by_task=dict(expected_task),complete_by_task=dict(completed_task),prediction_origins=dict(origins))
    pending_errors=sorted(str(p.relative_to(c.RUN)) for p in (c.RUN/'evaluation/errors').glob('*.json'))
    rosters_ok=input_count==expected_input and compression['expected']==9300 and consumers['expected']==75142
    all_complete=all(not section[field] for section in (compression,consumers) for field in ('missing','errors','unexpected'))
    passed=rosters_ok and all_complete and not pending_errors
    result=dict(schema='jointzip.fixed-backend-full.closeout.v1',passed=passed,partial=not passed,
        checked_utc=c.now(),inputs=input_count,expected_inputs=expected_input,rosters_match=rosters_ok,
        compression=compression,consumers=consumers,unresolved_consumer_error_files=pending_errors,
        verification='Exact planned configuration identity, committed success, output existence, complete-byte arithmetic and recorded independent stream replay; no repeated full-tree hashes.',
        scope='Scientific compression and downstream coverage. Density, standalone timing and Windows export completion are separate gates.')
    c.write(c.RUN/'audit/closeout/CLOSEOUT.json',result)
    if not passed and not partial:
        raise RuntimeError('Incomplete fixed-backend experiment; see audit/closeout/CLOSEOUT.json')
    return dict(passed=passed,compression_complete=compression['complete'],compression_expected=compression['expected'],
                consumers_complete=consumers['complete'],consumers_expected=consumers['expected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    print(check(partial=parser.parse_args().partial),flush=True)
