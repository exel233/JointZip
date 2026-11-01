"""Validate framing and analysis APIs using persisted TRAIN smoke only."""
from .common import *
setup_runtime()
import numpy as np
from .pipeline import features,traditional_acf
from experiments.vldb.framework_ablation_20261017 import codec,container

def main():
    lag=json.loads((RUN/'TRAIN_PREPARATION.json').read_text())['selected_max_lag']
    rows=[]
    for dataset in DATASETS:
        x=np.load(RUN/'inputs'/(dataset+'_train.npz'))['values'][:1,None,:]
        for backend in BACKENDS:
            for arm,tau in ARMS.items():
                folder=RUN/'smoke'/f'{dataset}_{backend}_{arm}'
                original=(folder/'stream.jzf').read_bytes()
                meta,payload=container.unframe(original)
                header,records=codec.unframe(payload)
                assert header==dict(codec=backend,shape=(1,1,500),epsilon=1.,tau=tau,rho=.99,max_lag=lag)
                merged,_=codec._frame(backend,(2,1,500),1.,tau,.99,lag,records+records)
                complete,_=container.frame(merged,dict(task='classification',mean=0.,std=1.,channel=dataset),'train_validation',(2,1,500))
                path=RUN/'train_validation'/f'{dataset}_{backend}_{arm}.jzf';path.parent.mkdir(exist_ok=True);path.write_bytes(complete)
                y,_=container.decode(path.read_bytes(),WORK/'train_validation'/f'{dataset}_{backend}_{arm}')
                expected=np.repeat(np.load(folder/'decoded.npy'),2,axis=0)
                assert np.array_equal(y.view('u4'),expected.view('u4'))
                checks=codec.audit(np.repeat(x,2,axis=0),y,1.,tau,.99,lag)
                assert checks['numerical_pass_windows']==2
                m=json.loads((folder/'metadata.json').read_text());counts=m['counts']
                mech=dict(max_E=checks['max_abs_error'],min_R=checks['min_pearson'],max_slice_ACF=checks['max_acf_error'],
                    lossless_windows=counts['lossless_windows'],repair_values=counts['correction_values'],
                    correction_payload_bytes=sum(len(r['correction']) for r in records))
                feat,names=features(y[:,0,:]);assert len(names)==22 and feat['spectrum32'].shape==(2,32)
                assert np.isfinite(traditional_acf(y[:,0,:],lag)).all()
                assert np.allclose(feat['spectrum32'].sum(axis=1),1.)
                rows.append(dict(dataset=dataset,backend=backend,arm=arm,passed=True,**mech))
    write(RUN/'TRAIN_ASSEMBLY_VALIDATION.json',dict(passed=True,split='TRAIN only, duplicated first record tests concatenation',rows=rows,created_utc=now()))
    print(json.dumps({'passed':True,'train_assemblies':len(rows)}))

if __name__=='__main__':main()
