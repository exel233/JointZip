"""Bind existing inputs and dependencies once; never regenerate prior data."""
from pathlib import Path
import os
import platform
import shutil
import numpy as np
from . import common as c, codec

PROTOCOL=c.ROOT/'docs/vldb/FRAMEWORK_ABLATION_PROTOCOL_20261017.md'

def verify_dependency():
    deps=[]
    for root,ncomp,nconsumer in ((c.HISTORY_FULL,9300,75142),(c.HISTORY_SZ,1860,17014)):
        science=c.read(root/'execution/COMPLETE.json'); close=c.read(root/'audit/closeout/CLOSEOUT.json')
        delivery=c.read(c.ROOT/'tmp/runtime_framework_ablation_20261017/DEPENDENCY_DELIVERIES.json')[root.name]
        if science.get('complete') is not True or science.get('compression')!=ncomp or science.get('consumer_records')!=nconsumer:
            raise RuntimeError('Incomplete source run: '+str(root))
        if close.get('passed') is not True or delivery.get('complete') is not True:
            raise RuntimeError('Source closeout/delivery incomplete')
        for key,count in (('compression',ncomp),('consumers',nconsumer)):
            sect=close[key]
            if sect['complete']!=count or any(sect.get(k) for k in ('missing','errors','unexpected')):
                raise RuntimeError('Source roster not complete')
        deps.append(dict(run=str(root.relative_to(c.ROOT)),freeze_sha256=c.sha(root/'freeze.json'),
                         scientific_completed_utc=science.get('completed_utc'),compression=ncomp,consumers=nconsumer))
    return deps

def run():
    dependency=verify_dependency()
    c.RUN.mkdir(parents=True,exist_ok=True)
    from . import smoke_codec
    if not smoke_codec.READY.is_file():raise RuntimeError('Real six-backend preflight not completed')
    smoke=c.read(smoke_codec.READY)
    if not smoke.get('complete') or smoke['bindings']!=smoke_codec.bindings() or smoke['identities']!={b:codec.identity(b) for b in c.BACKENDS}:
        raise RuntimeError('Completed codec preflight binding differs')
    # These SQLite connections were closed at the successful preflight end.
    # Import completed unrepaired native work, rather than compute it again.
    for backend in c.BACKENDS:
        source=c.RUN/'audit/preflight_codec'/backend/'native_cache'
        destination=c.RUN/'native_candidate_cache'/backend
        if not destination.exists():
            if not (source/'native.sqlite3').is_file():raise RuntimeError('Missing preflight candidate cache')
            shutil.copytree(source,destination)
    parent=c.read(c.OLD_SIMPLE/'freeze.json'); entries=parent['entries']
    if len(entries)!=172: raise RuntimeError('Expected original 172 inputs')
    for entry in entries:
        src=c.OLD_SIMPLE/entry['input'];dst=c.RUN/entry['input'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try: os.link(src,dst)
            except OSError: shutil.copyfile(src,dst)
        if c.sha(dst)!=entry['input_sha256']: raise RuntimeError('Frozen input differs: '+entry['sid'])
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('native/*.cpp'))+[PROTOCOL]
    value=dict(schema='jointzip.framework-ablation.v1',protocol=c.PROTOCOL,entries=entries,dependency=dependency,
       preflight=dict(path=str(smoke_codec.READY.relative_to(c.ROOT)),sha256=c.sha(smoke_codec.READY)),
       source_binding=dict(run=str(c.OLD_SIMPLE.relative_to(c.ROOT)),freeze_sha256=c.sha(c.OLD_SIMPLE/'freeze.json')),
       code={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(files)},
       backends={b:codec.identity(b) for b in c.BACKENDS},
       repair=dict(binary=str(codec.REPAIR),sha256=c.sha(codec.REPAIR)),
       anchors=dict(binary=str(codec.ANCHORS),sha256=c.sha(codec.ANCHORS)),
       runtime=dict(kernel=platform.release(),python=platform.python_version(),numpy=np.__version__),
       expected=dict(inputs=172,compression=27900,consumer_records=220462,
                     consumer_by_task=dict(forecast=206752,classification=1488,retrieval=1302,anomaly=10920),
                     reused_era_compression=5580,reused_era_consumers=43596))
    c.write(c.RUN/'freeze.json',value,exclusive=True)
    print('Prepared frozen six-backend ablation inputs and policy',flush=True)

if __name__=='__main__': run()
