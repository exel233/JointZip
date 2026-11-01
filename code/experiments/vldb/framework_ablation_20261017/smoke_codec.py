"""Small real-library preflight, reusable by the frozen pipeline stage.

Only a single real VAL window is used per backend. Completed ERA encodes are
never repeated: their saved cap trace and native selected members are read.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT/'tmp/runtime_framework_ablation_20261017'
READY = RUNTIME/'CODEC_READY.json'


def bindings():
    files = [Path(__file__).with_name(name) for name in
             ('codec.py','native_cache.py','replay.py','smoke_codec.py','test_codec_isolation.py')]
    files += [ROOT/'experiments/vldb/framework_sz2016_20261016/codec.py',
              ROOT/'tools/qpet_correlation_guided_20261013_v1/CorrelationQoI.hpp']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def run():
    from . import codec as c, common, container
    from .test_codec_isolation import run as isolated_test
    from experiments.vldb.framework_sz2016_20261016 import codec as old
    binding = bindings()
    identities = {backend:c.identity(backend) for backend in common.BACKENDS}
    if READY.exists():
        prior = json.loads(READY.read_text())
        if prior.get('complete') and prior.get('bindings') == binding and prior.get('identities') == identities:
            return dict(prior, reused_complete_preflight=True)
        raise RuntimeError('Codec preflight receipt exists but bindings differ; preserve and resolve explicitly')
    started = time.time()
    output = common.RUN/'audit'/'preflight_codec'
    output.mkdir(parents=True,exist_ok=True)
    tests = dict(offline=isolated_test(),backends=[])
    full = common.HISTORY_FULL
    freeze = json.loads((full/'freeze.json').read_text())
    entry = next(v for v in freeze['entries'] if v['task'] == 'forecast')
    with np.load(full/entry['input'],allow_pickle=False) as data:
        source = np.ascontiguousarray(data['val_source'][:1],dtype=np.float32)
    if source.shape != (1,1,128):
        raise RuntimeError('Real preflight VAL window shape changed')
    point = common.POINTS[0]
    for backend in common.BACKENDS:
        folder = output/backend; folder.mkdir(exist_ok=True)
        cache = c.NativeCandidateCache(folder/'native_cache')
        comparisons = []
        for arm in ('E','ER','EA'):
            limits = common.arm_constraints(point,arm)
            args = (backend,source,limits['epsilon'],limits['tau'],limits['rho'],16,True)
            if arm == 'E':
                # The historical repair CLI rejects both inactive statistics.
                # Its only required operation here is already satisfied by the
                # unchanged strict native cap gate; make that no-op explicit.
                def e_only_noop(original,base,epsilon,tau,rho,lag,workdir):
                    if tau >= 0 or rho >= 0 or np.max(np.abs(original.astype(float)-base.astype(float))) > epsilon:
                        raise AssertionError('E-only reference no-op precondition failed')
                    return b'',base.copy(),0,0
                with patch.object(old,'_repair',e_only_noop):
                    reference = old.encode(*args,workdir=folder/(arm+'_reference'))
            else:
                reference = old.encode(*args,workdir=folder/(arm+'_reference'))
            plain = c.encode(*args,workdir=folder/(arm+'_plain'))
            cached = c.encode(*args,workdir=folder/(arm+'_cached'),native_cache=cache)
            repeated = c.encode(*args,workdir=folder/(arm+'_cache_repeat'),native_cache=cache)
            if not reference.payload == plain.payload == cached.payload == repeated.payload:
                raise RuntimeError('New/default/cache stream differs from reference: '+backend+'/'+arm)
            if not reference.reconstruction.tobytes() == plain.reconstruction.tobytes() == cached.reconstruction.tobytes() == repeated.reconstruction.tobytes():
                raise RuntimeError('New/default/cache reconstruction differs: '+backend+'/'+arm)
            if repeated.metadata['counts']['native_probes'] != 0:
                raise RuntimeError('Repeat raw candidate cache made new native calls')
            comparisons.append(dict(arm=arm,reference='old encoder with explicit E-only empty repair' if arm=='E' else 'unchanged old encoder',
                complete_stream_equal=True,decoded_bits_equal=True,
                enabled_pass_windows=cached.metadata['audit']['numerical_pass_windows'],
                cached_counts=cached.metadata['counts'],repeat_counts=repeated.metadata['counts']))
        history = common.history_run(backend)
        history_row = history/'compression'/entry['task']/entry['sid']/'val'/(backend+'_jointzip__'+point['label']+'.json')
        saved = json.loads(history_row.read_text())
        metadata = json.loads((history/saved['metadata']).read_text())
        _,inner = container.unframe((history/saved['artifact']).read_bytes())
        _,records = c.unframe(inner)
        trace = copy.deepcopy(metadata)
        trace['shape'] = list(source.shape)
        trace['windows'] = trace['windows'][:1]
        _,native_inner = container.unframe((history/json.loads((history/'compression'/entry['task']/entry['sid']/'val'/(backend+'_native__'+point['label']+'.json')).read_text())['artifact']).read_bytes())
        _,native_records = c.unframe(native_inner)
        imported = 0
        for record in (records[0],native_records[0]):
            if record['kind'] == 1:
                imported += cache.import_native_payload(backend,source[0,0],record['cap'],record['base'],folder/'import_native',
                                dict(kind='historical_smoke_selected_base',row=str(history_row)))
        captures = {}
        def checkpoint(index,record,evidence,options):
            captures[index] = dict(record=copy.deepcopy(record),evidence=copy.deepcopy(evidence))
        result = c.replay_without_correction(backend,source,point['epsilon'],point['tau'],point['rho'],16,
                 trace,folder/'no_correction',native_cache=cache,window_callback=checkpoint)
        with patch.object(c,'_repair',side_effect=AssertionError('Replay invoked correction')), \
             patch.object(c,'_native_encode',side_effect=AssertionError('Checkpoint replay re-encoded')):
            resumed = c.replay_without_correction(backend,source,point['epsilon'],point['tau'],point['rho'],16,
                      trace,folder/'no_correction_resume',native_cache=cache,resume_records=captures)
        if resumed.payload != result.payload or resumed.reconstruction.tobytes() != result.reconstruction.tobytes():
            raise RuntimeError('No-correction checkpoint resume differs')
        if result.metadata['correction_bytes'] or result.metadata['counts']['correction_values'] or result.metadata['audit']['numerical_pass_windows'] != 1:
            raise RuntimeError('No-correction repair or ERA audit failed')
        expected_caps = [v['cap'] for v in trace['windows'][0]['candidates'] if v['kind']==1]
        actual_caps = [v['cap'] for v in result.metadata['windows'][0]['candidates'] if v['kind']==1]
        if expected_caps != actual_caps:
            raise RuntimeError('No-correction logical candidate order differs')
        tests['backends'].append(dict(backend=backend,comparisons=comparisons,
            historical_ERA_not_encoded=True,old_trace_source=str(history_row),
            imported_native_members=imported,fixed_caps_equal=True,
            no_correction_zero_repairs=True,no_correction_ERA_pass=True,
            no_correction_resume_stream_equal=True,no_correction_counts=result.metadata['counts']))
        cache.close()
        print(json.dumps(dict(event='codec_preflight_backend_complete',backend=backend)),flush=True)
    result = dict(complete=True,bindings=binding,identities=identities,
                  real_source=dict(entry=entry['sid'],input=entry['input'],input_sha256=entry['input_sha256'],
                                   split='val',window=0,shape=list(source.shape)),
                  threshold=point,tests=tests,elapsed_seconds=time.time()-started,
                  completed_utc=common.now(),production_configurations_run=0,
                  scope='six backends, one real VAL window, E/ER/EA reference/default/cache equality; old ERA trace replay without correction')
    temporary = READY.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');temporary.replace(READY)
    (output/'COMPLETE.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('jointzip_ablation_preflight_runtime',RUNTIME/'execute.py')
    runtime = importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    runtime.setup()
    print(json.dumps(run(),sort_keys=True))
