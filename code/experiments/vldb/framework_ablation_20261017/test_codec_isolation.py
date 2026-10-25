"""Focused offline invariants; deterministic fake codec, real framing/cache.

This does not stand in for official-library smoke tests. It tests that cache
reuse cannot use repaired values, that replay considers every recorded cap,
that negative constraints survive unchanged, and that disabled correction
still enforces ERA with the original numerical tolerance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from . import codec as c
from .native_cache import NativeCandidateCache, effective_cap


def run():
    source = (.1*np.sin(np.arange(32)*2*np.pi/8)).astype(np.float32).reshape(1,1,-1)
    row = source[0,0]
    native_values = {}
    calls = []
    config = ['first']

    def identity(backend):
        return dict(binary='fake',binary_sha256='fake-binary',codec=backend,
                    configuration='fake.config',configuration_sha256=config[0],dtype='<f4')

    def native_encode(backend, original, cap, folder):
        calls.append((backend,float(cap)))
        rebuilt = np.zeros_like(original) if cap >= .25 else original.copy()
        payload = b'N'+(b'0' if cap >= .25 else b'1')
        native_values[payload] = rebuilt.copy()
        return payload,rebuilt

    def native_decode(backend,payload,length,folder):
        return native_values[payload].copy()

    def anchors(command):
        count = int(command[3])
        np.full(count,.125,dtype='<f8').tofile(command[2])

    # Only E is enabled for equivalence checks, so the real repair is a no-op.
    def repair(original,base,epsilon,tau,rho,lag,folder):
        assert tau < 0 and rho < 0
        return b'',base.copy(),0,1

    with tempfile.TemporaryDirectory(prefix='jointzip_ablation_unit_') as tmp:
        root = Path(tmp)
        with patch.object(c,'identity',identity), patch.object(c,'_native_encode',native_encode), \
             patch.object(c,'_native_decode',native_decode),patch.object(c,'_run',anchors),patch.object(c,'_repair',repair):
            cache = NativeCandidateCache(root/'cache')
            plain = c.encode('sz3',source,.5,-1.,-1.,16,True,root/'plain')
            cached = c.encode('sz3',source,.5,-1.,-1.,16,True,root/'cached',native_cache=cache)
            repeat = c.encode('sz3',source,.5,-1.,-1.,16,True,root/'repeat',native_cache=cache)
            assert plain.payload == cached.payload == repeat.payload
            assert plain.reconstruction.tobytes() == cached.reconstruction.tobytes() == repeat.reconstruction.tobytes()
            assert repeat.metadata['counts']['native_probes'] == 0
            assert repeat.metadata['counts']['native_cache_hits'] > 0
            assert cache.get('sz3',row,.5)[1].tobytes() == np.zeros_like(row).tobytes()

            # Config identity is part of the cache even if the binary matches.
            assert cache.get('sz2016',row,.5) is None
            cache.put('sz2016',row,.5,b'N0',np.zeros_like(row))
            config[0] = 'second'
            assert cache.get('sz2016',row,.5) is None
            config[0] = 'first'
            assert cache.get('sz2016',row,.5) is not None
            assert effective_cap('zfp',.2) == effective_cap('zfp',.24) == .125
            assert effective_cap('serf_qt',.1) == c._down(.1)

            trace = dict(codec='sz3',framework=True,shape=list(source.shape),identity=identity('sz3'),
                epsilon=.5,tau=.05,rho=.99,max_lag=16,search_options=c.search_options('sz3',True),
                windows=[dict(anchor=.125,candidates=[dict(kind=1,cap=.5,phase='seed'),
                    dict(kind=1,cap=.125,phase='seed'),dict(kind=0,cap=.5,phase='lossless')])])
            captured = {}
            def checkpoint(index,record,evidence,options):
                captured[index] = dict(record=copy.deepcopy(record),evidence=copy.deepcopy(evidence))
            result = c.replay_without_correction('sz3',source,.5,.05,.99,16,trace,root/'replay',
                      native_cache=cache,window_callback=checkpoint)
            assert result.metadata['windows'][0]['selected']['kind'] == 1
            assert result.metadata['windows'][0]['selected']['cap'] == .125
            assert not result.metadata['windows'][0]['candidates'][0]['eligible']
            assert result.metadata['counts']['native_probes'] == 0
            assert result.reconstruction.tobytes() == source.tobytes()
            with patch.object(c,'_native_encode',side_effect=AssertionError('resume recomputed')):
                resumed = c.replay_without_correction('sz3',source,.5,.05,.99,16,trace,
                           root/'resume',native_cache=cache,resume_records=captured)
            assert resumed.payload == result.payload
            assert resumed.metadata['counts']['reused_windows'] == 1

            # Removing the sole feasible cap must fall back losslessly, never
            # select a numerically valid-E but invalid-R/A coarse candidate.
            bad_trace = copy.deepcopy(trace)
            bad_trace['windows'][0]['candidates'] = [bad_trace['windows'][0]['candidates'][0]]
            fallback = c.replay_without_correction('sz3',source,.5,.05,.99,16,bad_trace,
                       root/'fallback',native_cache=cache)
            assert fallback.metadata['counts']['lossless_windows'] == 1
            assert fallback.reconstruction.tobytes() == source.tobytes()

            # The gate must match C++ verify_constraints: a value within its
            # 64-eps tolerance is admitted even if the strict audit says false.
            real_audit = c.audit
            def near_boundary(*args,**kwargs):
                result = real_audit(*args,**kwargs)
                for item in result['rows']:
                    item['strict_pass'] = False
                    item['numerical_pass'] = True
                result['strict_pass_windows'] = 0
                result['numerical_pass_windows'] = len(result['rows'])
                return result
            with patch.object(c,'audit',near_boundary):
                tolerant = c.replay_without_correction('sz3',source,.5,.05,.99,16,bad_trace,
                           root/'tolerance',native_cache=cache)
            assert tolerant.metadata['counts']['lossless_windows'] == 0
            assert tolerant.metadata['audit']['strict_pass_windows'] == 0
            assert tolerant.metadata['audit']['numerical_pass_windows'] == 1
            cache.close()
    return dict(passed=True,scope='offline fake-backend invariants; official smoke still required',
        cached_encode_bytes_equal=True,negative_constraint_flags_preserved=True,
        unrepaired_cache_only=True,config_identity_keyed=True,zfp_effective_key=True,
        serf_rounding_key=True,fixed_trace_all_caps=True,era_gate_enforced=True,
        original_numeric_tolerance=True,checkpoint_replay=True,exact_fallback=True)


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
