"""Bounded post-dependency checks for the isolated official SZ2016 adapter.

Run only after the five-backend experiment, including standalone timing, has
finished. No production scientific rows are written by these checks.
"""
from pathlib import Path
import json
import tempfile

import numpy as np

from . import common as c, codec, reuse


def bits(left, right):
    return left.shape == right.shape and np.array_equal(left.view(np.uint32), right.view(np.uint32))


def check_roundtrip(source, point, folder, framework):
    result=codec.encode('sz2016',source,point['epsilon'],point['tau'],point['rho'],
        framework=framework,workdir=folder)
    restored=codec.decode(result.payload,folder)
    if not bits(restored,result.reconstruction):
        raise RuntimeError('SZ2016 smoke stored-stream replay differs')
    checks=result.metadata['audit']
    if framework and checks['numerical_pass_windows'] != len(source):
        raise RuntimeError('SZ2016 smoke framework ERA failed')
    if not framework and checks['max_abs_error'] > point['epsilon']+64*np.finfo(float).eps:
        raise RuntimeError('SZ2016 smoke native bound failed')
    return result


def run():
    destination=c.RUN/'audit/smoke_sz2016'
    destination.mkdir(parents=True,exist_ok=True)
    point=c.POINTS[0]
    cases={
        'varying128':np.asarray(np.sin(np.arange(128)*.17)+np.arange(128)*.004,dtype=np.float32),
        'constant128':np.full(128,1.25,dtype=np.float32),
        'odd97':np.asarray(np.cos(np.arange(97)*.23)+np.arange(97)*.003,dtype=np.float32),
    }
    synthetic=[]
    for name,values in cases.items():
        source=values.reshape(1,1,-1)
        for framework in (False,True):
            result=check_roundtrip(source,point,destination/(name+'_'+str(framework)),framework)
            synthetic.append(dict(case=name,framework=framework,bytes=len(result.payload),
                audit={k:v for k,v in result.metadata['audit'].items() if k!='rows'}))
    entries=c.entries()
    panels=[next(e for e in entries if e['task']=='forecast' and e['sid']=='ETTm1__000'),
            next(e for e in entries if e['task']=='ucr' and e['sid']=='ECG200'),
            next(e for e in entries if e['task']=='anomaly' and e['dataset']=='MSL')]
    real=[]
    for entry in panels:
        split='val' if entry['task']=='forecast' else 'test'
        point=c.STRESS if entry['task']=='ucr' else c.POINTS[0]
        data=c.arrays(entry)
        source=data['val_source' if split=='val' else 'source']
        historical=reuse.load_native(entry,split,'sz2016',point,source)
        if historical is None:
            raise RuntimeError('SZ2016 smoke expected historical native records')
        cache=codec.WindowCache()
        cache.add_records('sz2016',source,point['epsilon'],point['tau'],point['rho'],16,False,
            historical['records'],None,historical['lineage'],workdir=destination/entry['sid'])
        native=codec.encode('sz2016',source,point['epsilon'],point['tau'],point['rho'],
            framework=False,warm_cache=cache,workdir=destination/(entry['sid']+'_reused'))
        if not bits(native.reconstruction,historical['reconstruction']):
            raise RuntimeError('SZ2016 reframed native reconstruction differs from history')
        if native.metadata['counts']['reused_windows'] != len(source):
            raise RuntimeError('SZ2016 native panel did not fully reuse history')
        # One new official call per task checks that stripping/reinstating the
        # historical uint32 frame preserved the exact native member bytes.
        with tempfile.TemporaryDirectory(prefix='native_member_',dir=destination) as tmp:
            body,restored=codec._native_encode('sz2016',source[0,0],point['epsilon'],Path(tmp))
        if body != historical['records'][0]['base'] or not bits(restored,historical['reconstruction'][0,0]):
            raise RuntimeError('SZ2016 native member or restored values differ from history')
        framework=check_roundtrip(source[:2].copy(),point,destination/(entry['sid']+'_framework'),True)
        real.append(dict(sid=entry['sid'],task=entry['task'],split=split,
            historical_native_windows=len(source),fresh_identity_probe_windows=1,
            native_member_byte_identical=True,native_reconstruction_bitwise_equal=True,
            framework_windows=min(2,len(source)),framework_ERA_pass=framework.metadata['audit']['numerical_pass_windows'],
            framework_probes=framework.metadata['counts']['native_probes'],
            framework_complete_stream_bytes=len(framework.payload)))
    # Framing errors must stop an import, not silently truncate native members.
    invalid=[b'\x01',b'\x04\x00\x00\x00abc',b'\x00\x00\x00\x00']
    for payload in invalid:
        try:
            reuse._native_rows('sz2016',payload,source_shape=(1,1,128),requested_epsilon=.1)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('SZ2016 smoke accepted invalid historical framing')
    result=dict(complete=True,scientific_results=False,identity=codec.identity('sz2016'),
        simple_correction_unchanged=True,search_options=codec.search_options('sz2016'),
        synthetic=synthetic,real=real,invalid_native_framing_rejected=len(invalid))
    c.write(destination/'COMPLETE.json',result,exclusive=True)
    print(json.dumps(result,allow_nan=False),flush=True)
    return result


if __name__=='__main__':
    run()
