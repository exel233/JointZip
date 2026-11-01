"""Predeclared standalone paired timing, separate from reusable science jobs."""
from pathlib import Path
import time
import numpy as np
from . import common as c,codec,container

def run():
    groups={}
    for e in c.entries('forecast'): groups.setdefault(e['dataset'],e)
    if len(groups)!=6: raise RuntimeError('Timing panel must have six forecast sources')
    out=c.RUN/'timing';out.mkdir(parents=True,exist_ok=True)
    total=0
    for dataset,entry in sorted(groups.items()):
        source=np.ascontiguousarray(c.arrays(entry)['val_source'][:4],dtype=np.float32)
        for point in (c.POINTS[0],c.POINTS[2]):
            for backend in c.BACKENDS:
                for repeat in range(4):
                    # Deterministic alternating paired order limits position bias.
                    for framework in ((False,True) if repeat%2==0 else (True,False)):
                        key='__'.join((dataset,point['label'],backend,'jointzip' if framework else 'native',str(repeat)))
                        path=out/'rows'/(key+'.json')
                        if path.exists():
                            if c.read(path)['status']!='complete': raise RuntimeError('Invalid timing row')
                            total+=1;continue
                        work=out/'work'/key;work.mkdir(parents=True,exist_ok=True)
                        # Identity hashing is setup, matching the production native API.
                        codec.identity(backend)
                        start=time.perf_counter()
                        result=codec.encode(backend,source,point['epsilon'],point['tau'],point['rho'],16,framework,work)
                        verified_encode_wall=time.perf_counter()-start
                        frame_start=time.perf_counter()
                        stream,meta=container.frame(result.payload,entry,'val',source.shape)
                        encode_seconds=result.metadata['encode_seconds']+time.perf_counter()-frame_start
                        start=time.perf_counter();y,_=container.decode(stream,work/'decode');decode_seconds=time.perf_counter()-start
                        if y.tobytes()!=result.reconstruction.tobytes(): raise RuntimeError('Timing decode mismatch')
                        check=codec.audit(source,y,point['epsilon'],point['tau'],point['rho'],16)
                        if framework and check['numerical_pass_windows']!=len(source): raise RuntimeError('Timing contract failed')
                        c.write(path,dict(status='complete',dataset=dataset,sid=entry['sid'],point=point,backend=backend,
                            framework=framework,repeat=repeat,warmup=repeat==0,encode_seconds=encode_seconds,
                            verified_encode_wall_seconds=verified_encode_wall,
                            decode_seconds=decode_seconds,raw_bytes=source.nbytes,artifact_bytes=len(stream),cr=source.nbytes/len(stream),
                            windows=len(source),counts=result.metadata['counts'],audit=check,observed_utc=c.now()),exclusive=True)
                        total+=1;c.write(out/'STATUS.json',dict(completed=total,total=480,last=key,observed_utc=c.now()))
                        print('timing',total,'/480',key,flush=True)
    if total!=480: raise RuntimeError('Incomplete independent paired timing')
    c.write(out/'COMPLETE.json',dict(complete=True,total=480,warmups=120,measurements=360,source_count=6,backend_count=5,completed_utc=c.now()))

if __name__=='__main__':run()
