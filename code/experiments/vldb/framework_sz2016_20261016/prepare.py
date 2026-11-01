"""Bind existing inputs and dependencies once; never regenerate prior data."""
from pathlib import Path
import os
import platform
import shutil
import numpy as np
from . import common as c, codec

PROTOCOL=c.ROOT/'docs/vldb/FRAMEWORK_SZ2016_PROTOCOL_20261016.md'
DEPENDENCY=Path('/opt/jointzip-home/.local/cth_expansion_runs/vldb_framework_full_20261015_v1')
DEPENDENCY_EXPORT=c.ROOT/'experiments/results/vldb_framework_full_20261015_v1'

def verify_dependency():
    """Read small completion gates before any new experiment or timing work."""
    required={
        'science':DEPENDENCY/'execution/COMPLETE.json',
        'closeout':DEPENDENCY/'audit/closeout/CLOSEOUT.json',
        'density':DEPENDENCY/'density/COMPLETE.json',
        'timing':DEPENDENCY/'timing/COMPLETE.json',
        'delivery':DEPENDENCY_EXPORT/'DELIVERY.json',
    }
    for name,path in required.items():
        if not path.is_file():
            raise RuntimeError('Parent run is not fully delivered; missing '+name+': '+str(path))
    gates={name:c.read(path) for name,path in required.items()}
    for name in ('science','density','timing','delivery'):
        if gates[name].get('complete') is not True:
            raise RuntimeError('Parent completion gate failed: '+name)
    if (gates['science'].get('compression')!=9300 or gates['science'].get('consumer_records')!=75142
            or gates['closeout'].get('passed') is not True
            or gates['closeout'].get('rosters_match') is not True):
        raise RuntimeError('Parent scientific coverage differs')
    for section,count in (('compression',9300),('consumers',75142)):
        value=gates['closeout'].get(section,{})
        if value.get('expected')!=count or value.get('complete')!=count or any(value.get(k) for k in ('missing','errors','unexpected')):
            raise RuntimeError('Parent closeout coverage failed: '+section)
    if gates['closeout'].get('unresolved_consumer_error_files'):
        raise RuntimeError('Parent has unresolved consumer errors')
    if gates['density'].get('conditions')!=144:
        raise RuntimeError('Parent density coverage differs')
    if (gates['timing'].get('total')!=480 or gates['timing'].get('warmups')!=120
            or gates['timing'].get('measurements')!=360):
        raise RuntimeError('Parent standalone timing coverage differs')
    return dict(run='vldb_framework_full_20261015_v1',
                gates={name:dict(path=str(path),completed_utc=gates[name].get('completed_utc'))
                       for name,path in required.items()})

def run():
    dependency=verify_dependency()
    c.RUN.mkdir(parents=True,exist_ok=True)
    parent=c.read(c.OLD_SIMPLE/'freeze.json'); entries=parent['entries']
    if len(entries)!=172: raise RuntimeError('Expected original 172 inputs')
    for entry in entries:
        src=c.OLD_SIMPLE/entry['input'];dst=c.RUN/entry['input'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try: os.link(src,dst)
            except OSError: shutil.copyfile(src,dst)
        if c.sha(dst)!=entry['input_sha256']: raise RuntimeError('Frozen input differs: '+entry['sid'])
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('native/*.cpp'))+[PROTOCOL]
    value=dict(schema='jointzip.fixed-backend-sz2016.v1',protocol=c.PROTOCOL,entries=entries,dependency=dependency,
       source_binding=dict(run=str(c.OLD_SIMPLE.relative_to(c.ROOT)),freeze_sha256=c.sha(c.OLD_SIMPLE/'freeze.json')),
       code={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(files)},
       backends={b:codec.identity(b) for b in c.BACKENDS},
       repair=dict(binary=str(codec.REPAIR),sha256=c.sha(codec.REPAIR)),
       anchors=dict(binary=str(codec.ANCHORS),sha256=c.sha(codec.ANCHORS)),
       runtime=dict(kernel=platform.release(),python=platform.python_version(),numpy=np.__version__),
       expected=dict(inputs=172,compression=1860,consumer_records=17014,
                     consumer_by_task=dict(forecast=15904,classification=144,retrieval=126,anomaly=840),
                     timing_calls=96,timing_pairs=12))
    c.write(c.RUN/'freeze.json',value,exclusive=True)
    print('Prepared frozen SZ2016 extension inputs and policy',flush=True)

if __name__=='__main__': run()
