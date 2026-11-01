"""One first TRAIN window per dataset/backend/arm; diagnostic timing only."""
from __future__ import annotations
import json,time
from .common import *
setup_runtime()
import numpy as np
from experiments.vldb.framework_ablation_20261017 import codec,container

def main():
    lag=json.loads((RUN/'TRAIN_PREPARATION.json').read_text())['selected_max_lag']
    out=[]
    for dataset in DATASETS:
        data=np.load(RUN/'inputs'/(dataset+'_train.npz'))
        x=np.ascontiguousarray(data['values'][:1,None,:],dtype=np.float32)
        for backend in BACKENDS:
            cache=codec.NativeCandidateCache(WORK/'smoke_cache'/backend)
            for arm,tau in ARMS.items():
                target=RUN/'smoke'/(dataset+'_'+backend+'_'+arm);target.mkdir(parents=True,exist_ok=True)
                w=WORK/'smoke_work'/(dataset+'_'+backend+'_'+arm)
                started=time.perf_counter()
                value=codec.encode(backend,x,1.,tau,.99,lag,True,w,native_cache=cache)
                full,meta=container.frame(value.payload,dict(task='classification',mean=0.,std=1.,channel=dataset),'train_smoke',x.shape)
                decoded,_=container.decode(full,w/'independent_decode')
                assert np.array_equal(decoded.view('u4'),value.reconstruction.view('u4'))
                (target/'stream.jzf').write_bytes(full)
                np.save(target/'decoded.npy',decoded)
                write(target/'metadata.json',value.metadata)
                row=dict(dataset=dataset,backend=backend,arm=arm,seconds=time.perf_counter()-started,
                         bytes=len(full),cr=x.nbytes/len(full),audit=value.metadata['audit'],counts=value.metadata['counts'])
                out.append(row);write(RUN/'SMOKE_PROGRESS.json',out);print(json.dumps(row),flush=True)
    write(RUN/'SMOKE_COMPLETE.json',dict(complete=True,rows=out,completed_utc=now()))

if __name__=='__main__':main()
