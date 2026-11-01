"""Bind existing inputs and dependencies once; never regenerate prior data."""
from pathlib import Path
import os
import platform
import shutil
import numpy as np
from . import common as c, codec

PROTOCOL=c.ROOT/'docs/vldb/FRAMEWORK_FULL_PROTOCOL_20261015.md'
def run():
    c.RUN.mkdir(parents=True,exist_ok=True)
    parent=c.read(c.OLD_SIMPLE/'freeze.json'); entries=parent['entries']
    if len(entries)!=172: raise RuntimeError('Expected original 172 inputs')
    for entry in entries:
        src=c.OLD_SIMPLE/entry['input'];dst=c.RUN/entry['input'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            try: os.link(src,dst)
            except OSError: shutil.copyfile(src,dst)
        if c.sha(dst)!=entry['input_sha256']: raise RuntimeError('Frozen input differs: '+entry['sid'])
    files=[p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('smoke_')]+list(Path(__file__).parent.glob('native/*.cpp'))+[PROTOCOL]
    value=dict(schema='jointzip.fixed-backend-full.v1',protocol=c.PROTOCOL,entries=entries,
       source_binding=dict(run=str(c.OLD_SIMPLE.relative_to(c.ROOT)),freeze_sha256=c.sha(c.OLD_SIMPLE/'freeze.json')),
       code={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted(files)},
       backends={b:codec.identity(b) for b in c.BACKENDS},
       repair=dict(binary=str(codec.REPAIR),sha256=c.sha(codec.REPAIR)),
       anchors=dict(binary=str(codec.ANCHORS),sha256=c.sha(codec.ANCHORS)),
       runtime=dict(kernel=platform.release(),python=platform.python_version(),numpy=np.__version__),
       expected=dict(inputs=172,compression=9300,consumer_records=75142))
    c.write(c.RUN/'freeze.json',value,exclusive=True)
    print('Prepared frozen full framework inputs and policy',flush=True)

if __name__=='__main__': run()
