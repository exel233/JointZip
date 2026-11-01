"""Hash-verified replay and TRAIN-only split, before new model evaluation."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, OLD, DATASETS, PROTOCOL, sha, tensor_sha, write_json, code_identity

def main():
    sys.path.insert(0,str(ROOT/'src'))
    from experiments.vldb.external_utility import ucr_classification as ucr
    from experiments.vldb.external_utility.protocol import QPET_SPEC
    if os.name!='nt':
        from experiments.vldb.main_benchmark.native_bridge import install_linux_transport
        install_linux_transport()
    old=json.loads((OLD/'freeze.json').read_text())
    rows=json.loads((OLD/'results.json').read_text())['rows']
    previous=json.loads((ROOT/'experiments/vldb/main_results/ucr_existing_case_study_20260907/case_study.json').read_text())
    expected={(r['dataset'],r['method']):r['decoded_tensor_sha256'] for r in previous['validations']}
    freeze={'protocol':PROTOCOL,'old_freeze_sha256':sha(OLD/'freeze.json'),
            'old_results_sha256':sha(OLD/'results.json'),'datasets':{}, 'preparation_code':code_identity()}
    for name in DATASETS:
        item=next(r for r in old['selected_datasets'] if r['name']==name)
        archive, train_file,test_file=ucr._dataset_paths(ucr.DEFAULT_DATA,name)
        for p,key in [(archive,'archive_sha256'),(train_file,'train_file_sha256'),(test_file,'test_file_sha256')]:
            assert sha(p)==item[key], (name,key)
        train, yl, raw, yt=ucr.load_dataset(ucr.DEFAULT_DATA,name)
        train, raw, mean,std=ucr.train_standardize(train,raw)
        assert tensor_sha(train)==item['train_standardized_sha256']
        assert tensor_sha(raw)==item['test_standardized_sha256']
        labels=np.unique(yl)
        assert set(np.unique(yt)).issubset(set(labels))
        ytrain=np.searchsorted(labels,yl).astype(np.int64); ytest=np.searchsorted(labels,yt).astype(np.int64)
        rng=np.random.default_rng(PROTOCOL['split']['seed'])
        fit=[]; val=[]
        for label in range(len(labels)):
            indices=rng.permutation(np.flatnonzero(ytrain==label))
            n=max(1,int(np.ceil(.2*len(indices)))); assert len(indices)>n
            val.extend(indices[:n]);fit.extend(indices[n:])
        arrays={'train':train,'ytrain':ytrain,'ytest':ytest,'raw':raw,
                'fit_indices':np.sort(fit),'val_indices':np.sort(val),'class_labels':labels}
        entry={'train':len(train),'test':len(raw),'length':train.shape[-1],'classes':len(labels),
               'fit_indices':np.sort(fit).tolist(),'val_indices':np.sort(val).tolist(),
               'original_train_sha256':sha(train_file),'original_test_sha256':sha(test_file),'artifacts':{}}
        for row in [r for r in rows if r['dataset']==name]:
            method=row['variant'] if row['method']=='qpet_correlation' else row['method']
            p=OLD/row['artifact_file'];assert sha(p)==row['artifact_sha256']
            parsed=ucr.parse_scs2(p);ucr._verify_normalization(parsed,mean,std)
            if method in ('joint','point_only'):
                decoded,_=ucr.replay_qpet_scs2(p,QPET_SPEC,1,raw.shape[-1])
            else:
                decoded=ucr.replay_official_baseline_payload(method,parsed.payloads[0],(len(raw),1,raw.shape[-1]))
            assert tensor_sha(decoded)==expected[(name,method)], (name,method,'historical decoded hash differs')
            arrays[method]=decoded[:,0,:]
            entry['artifacts'][method]={'path':str(p.relative_to(ROOT)), 'sha256':sha(p),
                'complete_bytes':p.stat().st_size, 'complete_cr':row['compression_ratio'],
                'decoded_sha256':tensor_sha(decoded),'historical_raw_accuracy':row['raw_accuracy'],
                'historical_1nn_accuracy':row['reconstructed_accuracy']}
        dest=OUT/'inputs'/f'{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists(): raise RuntimeError('choose a new result directory; inputs already exist')
        np.savez(dest,**arrays)
        entry.update(prepared_file=str(dest.relative_to(OUT)),prepared_sha256=sha(dest))
        freeze['datasets'][name]=entry
        print(json.dumps({'prepared':name,'train':len(train),'test':len(raw),'artifacts':len(entry['artifacts'])}),flush=True)
    write_json(OUT/'freeze.json',freeze,exclusive=True)
    print('CLASSIFICATION_INPUT_FREEZE_COMPLETE',flush=True)

if __name__=='__main__':main()
