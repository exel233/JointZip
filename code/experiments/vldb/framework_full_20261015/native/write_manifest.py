from pathlib import Path
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
upstreams={name:subprocess.check_output(['git','-C',str(ROOT/'tmp/upstreams'/name),'rev-parse','HEAD'],text=True).strip() for name in ('SPERR','Serf')}
record={'schema':'jointzip.fixed-backend-bridge.v1','binary_sha256':sha(HERE/'fixed_backend'),
 'source_sha256':sha(HERE/'fixed_backend.cpp'),'upstreams':upstreams,
 'libraries':{name:sha(ROOT/path) for name,path in {'SPERR':'tmp/builds/SPERR-0.8.5/src/libSPERR.a','Serf':'tmp/builds/Serf-b38450b/libserf.so'}.items()},
 'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
 'calls':'SPERR historical standalone SPECK1D_FLT baseline API; Serf Oct13 guided APIs; no search/correction in bridge'}
(HERE/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
