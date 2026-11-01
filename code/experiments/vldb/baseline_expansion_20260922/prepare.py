"""Freeze unchanged source arrays and the new comparison protocol before scoring."""
import hashlib
import numpy as np
from .common import *

def main():
    if (RUN/'freeze.json').exists():
        print({'entries':len(frozen()['entries']),'resumed':True});return
    from experiments.vldb.main_benchmark import tasks_data
    old=read(MAIN/'freeze.json'); by={s['series_id']:s for s in old['series']}
    all_entries=[]
    def save(entry,values):
        p=RUN/'inputs'/entry['task']/(entry['sid']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,**values)
        entry.update(input=str(p.relative_to(RUN)),input_sha256=sha(p));all_entries.append(entry)
    for ds in tasks_data.DATASETS:
        for b in tasks_data.prepare_dataset(ds):
            for key in ('source','future','val_source','val_future','starts','val_starts'):
                if hashlib.sha256(np.ascontiguousarray(b[key]).tobytes()).hexdigest()!=by[b['series_id']]['arrays'][key]['sha256']:
                    raise RuntimeError('Old forecast input mismatch')
            save(dict(task='forecast',dataset=ds,sid=b['series_id'],channel=b['channel_name'],mean=b['mean'],std=b['std'],true_length=128),
                 {k:b[k] for k in ('source','future','val_source','val_future','starts','val_starts')})
    from experiments.vldb.downstream_expansion.classification import common as uc
    from experiments.vldb.external_utility import ucr_classification as original_uc
    for name in uc.DATASETS:
        z,e,fr=uc.load(name)
        with z:
            vals={k:z[k] for k in ('train','ytrain','ytest','raw','class_labels')}
        physical,_,test,_=original_uc.load_dataset(original_uc.DEFAULT_DATA,name)
        _,_,mean,std=original_uc.train_standardize(physical,test)
        save(dict(task='ucr',dataset=name,sid=name,channel='value',mean=float(mean),std=float(std),true_length=vals['raw'].shape[-1],old_input_sha256=e['prepared_sha256']),
             dict(source=np.ascontiguousarray(vals['raw'][:,None,:]),**vals))
    from experiments.vldb.downstream_expansion.anomaly import run as an
    oldn=read(NASA/'protocol.json');nby={e['file']:e for e in oldn['entries']}
    for name in an.catalog():
        z,y,tr,cut,mean,std,blocks=an.inputs(name)
        if hashlib.sha256(blocks.tobytes()).hexdigest()!=nby[name]['padded_source_sha256']:raise RuntimeError('NASA input differs')
        save(dict(task='anomaly',dataset='MSL' if '_MSL_' in name else 'SMAP',sid=an.series_id(name),file=name,channel='value',mean=mean,std=std,true_length=len(z)-tr,train_end=tr,fit_end=cut),
             dict(source=blocks,original=z,labels=y,train_end=tr,fit_end=cut))
    assert len(all_entries)==172
    record=dict(protocol=PROTOCOL,protocol_sha256=hashlib.sha256(canonical(PROTOCOL)).hexdigest(),created_utc=now(),entries=all_entries,
                upstream={str(p.relative_to(ROOT)):sha(p) for p in (MAIN/'freeze.json',PATCH/'freeze.json',UCR/'freeze.json',NASA/'protocol.json',FOUNDATION/'freeze.json')},
                preparation_code={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(__file__).with_name('common.py'))})
    write(RUN/'freeze.json',record,exclusive=True)
    print({'entries':len(all_entries),'forecast_series':142,'ucr_sets':6,'nasa_files':24,'freeze_sha256':sha(RUN/'freeze.json')},flush=True)

if __name__=='__main__':main()
