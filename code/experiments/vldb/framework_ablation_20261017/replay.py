"""Fixed-trajectory correction ablation; no adaptive search and no repair calls."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from . import codec as c
from .native_cache import backend_fingerprint, effective_cap


def trajectory(metadata):
    return [[dict(cap=float(item['cap']), phase=item.get('phase', 'recorded'))
             for item in window['candidates'] if item['kind'] == 1]
            for window in metadata['windows']]


def trajectory_sha256(metadata):
    return hashlib.sha256(json.dumps(trajectory(metadata), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def replay_without_correction(backend, source, epsilon, tau, rho, max_lag,
                              trace_metadata, workdir=None, *, native_cache=None,
                              window_callback=None, resume_records=None):
    if backend not in c.CODECS or source.dtype != np.float32 or source.ndim != 3 or source.shape[1] != 1:
        raise ValueError('fixed-trajectory replay expects float32 [W,1,T] and a known backend')
    if not np.isfinite(source).all() or not 0 < max_lag < source.shape[-1]-1 or epsilon <= 0 or tau < 0 or rho < 0:
        raise ValueError('correction ablation requires enabled ERA constraints')
    source = np.ascontiguousarray(source)
    if trace_metadata['codec'] != backend or not trace_metadata['framework'] or trace_metadata['shape'] != list(source.shape):
        raise ValueError('fixed ERA trace backend/shape/framework mismatch')
    for field, value in (('epsilon', epsilon), ('tau', tau), ('rho', rho), ('max_lag', max_lag)):
        if trace_metadata[field] != value:
            raise ValueError('fixed ERA trace constraints differ: '+field)
    if backend_fingerprint(trace_metadata['identity']) != backend_fingerprint(c.identity(backend)):
        raise ValueError('fixed ERA trace backend identity differs')
    traces = trajectory(trace_metadata)
    if len(traces) != source.shape[0]:
        raise ValueError('fixed ERA trace has wrong window count')
    for items in traces:
        if any(not math.isfinite(v['cap']) or not 0 < v['cap'] <= epsilon for v in items):
            raise ValueError('fixed ERA trace contains an invalid cap')
    digest = trajectory_sha256(trace_metadata)
    original_options = copy.deepcopy(trace_metadata['search_options'])
    options = dict(original_options, policy='fixed_era_trace_no_correction',
                   trace_sha256=digest, adaptive_refinement=False)
    records, evidence = [], []
    counts = dict(logical_probes=0, native_probes=0, cache_hits=0,
                  native_cache_hits=0, cap_rejections=0,
                  statistical_rejections=0, repair_target_candidates=0,
                  correction_values=0, lossless_windows=0,
                  eligible_lossy_windows=0, fresh_windows=0, reused_windows=0,
                  reused_logical_probes=0, reused_native_probes=0)
    parent = None if workdir is None else str(Path(workdir))
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='no_correction_', dir=parent) as tmp:
        folder = Path(tmp)
        for index, row in enumerate(source[:, 0, :]):
            prior = None if resume_records is None else resume_records.get(index)
            if prior is not None:
                chosen, item = copy.deepcopy(prior['record']), copy.deepcopy(prior['evidence'])
                if item.get('trace_sha256') != digest or item.get('fixed_candidates') != traces[index]:
                    raise RuntimeError('no-correction checkpoint belongs to a different trajectory')
                if chosen['count'] != 0 or chosen['correction']:
                    raise RuntimeError('no-correction checkpoint contains repair')
                if chosen['kind'] == 1:
                    if chosen['cap'] not in [v['cap'] for v in traces[index]]:
                        raise RuntimeError('checkpoint selected a cap outside the fixed trajectory')
                    chosen['rebuilt'] = c._native_decode(backend, chosen['base'], len(row), folder)
                    if float(np.max(np.abs(row.astype(float)-chosen['rebuilt'].astype(float)))) > chosen['cap']:
                        raise RuntimeError('no-correction checkpoint violates its original cap')
                elif chosen['kind'] == 0:
                    chosen['rebuilt'] = np.frombuffer(c.zstd.ZstdDecompressor().decompress(chosen['base']), dtype='<f4').copy()
                    if chosen['rebuilt'].tobytes() != row.tobytes():
                        raise RuntimeError('no-correction checkpoint fallback differs from source')
                else:
                    raise RuntimeError('unknown no-correction checkpoint record kind')
                if not c.audit(row.reshape(1,1,-1), chosen['rebuilt'].reshape(1,1,-1), epsilon,tau,rho,max_lag)['rows'][0]['numerical_pass']:
                    raise RuntimeError('no-correction checkpoint fails ERA')
                item['reused'] = True
                counts['reused_windows'] += 1
                counts['reused_logical_probes'] += len(traces[index])
                counts['eligible_lossy_windows'] += int(item['eligible_lossy_candidates'] > 0)
            else:
                counts['fresh_windows'] += 1
                observed, per_window_cache = [], {}
                for probe in traces[index]:
                    cap = probe['cap']; key = effective_cap(backend, cap)
                    counts['logical_probes'] += 1
                    local_hit = key in per_window_cache
                    if local_hit:
                        counts['cache_hits'] += 1
                        payload, rebuilt = per_window_cache[key]
                        native_hit = False
                    else:
                        if native_cache is None:
                            payload, rebuilt = c._native_encode(backend,row,cap,folder)
                            native_hit = False
                        else:
                            payload, rebuilt, native_hit = native_cache.get_or_compute(
                                backend,row,cap,lambda: c._native_encode(backend,row,cap,folder),
                                dict(kind='fixed_era_trace_no_correction',trace_sha256=digest,window=index))
                        counts['native_probes'] += int(not native_hit)
                        counts['native_cache_hits'] += int(native_hit)
                        per_window_cache[key] = payload, rebuilt.copy()
                    checks = c.audit(row.reshape(1,1,-1), rebuilt.reshape(1,1,-1),epsilon,tau,rho,max_lag)['rows'][0]
                    cap_pass = checks['max_abs_error'] <= cap
                    # verify_constraints() called by the unchanged C++ repair
                    # uses exactly 64*float64-eps tolerance, independently of
                    # the strict native cap filter above. Use that same gate.
                    era_pass = checks['numerical_pass']
                    counts['cap_rejections'] += int(not cap_pass)
                    counts['statistical_rejections'] += int(cap_pass and not era_pass)
                    observed.append(dict(kind=1,cap=cap,phase=probe['phase'],base=payload,
                        correction=b'',count=0,rebuilt=rebuilt.copy(),eligible=bool(cap_pass and era_pass),
                        base_max_error=checks['max_abs_error'],base_pearson=checks['pearson'],
                        base_acf_error=checks['max_acf_error'],base_cap_pass=cap_pass,
                        base_era_pass_strict=checks['strict_pass'],base_era_pass_numerical=checks['numerical_pass'],
                        repair_candidates=0,cache_hit=local_hit,native_cache_hit=native_hit))
                lossy_eligible = sum(v['eligible'] for v in observed)
                counts['eligible_lossy_windows'] += int(lossy_eligible > 0)
                observed.append(dict(kind=0,cap=epsilon,phase='lossless',
                    base=c.zstd.ZstdCompressor(level=3).compress(row.tobytes()),correction=b'',
                    count=0,rebuilt=row.copy(),eligible=True,base_max_error=0.,repair_candidates=0,
                    cache_hit=False,native_cache_hit=False))
                chosen = min((v for v in observed if v['eligible']),
                             key=lambda v:(len(v['base'])+len(v['correction']),-v['cap']))
                def describe(value):
                    return dict({k:v for k,v in value.items() if k not in ('base','correction','rebuilt')},
                                base_bytes=len(value['base']),correction_bytes=len(value['correction']))
                item = dict(window=index,anchor=trace_metadata['windows'][index].get('anchor'),
                            selected=describe(chosen),candidates=[describe(v) for v in observed],
                            trace_sha256=digest,fixed_candidates=traces[index],
                            eligible_lossy_candidates=lossy_eligible,reused=False)
            records.append(chosen); evidence.append(item)
            counts['lossless_windows'] += int(chosen['kind'] == 0)
            if window_callback is not None:
                window_callback(index,chosen,item,options)
        payload, inner_bytes = c._frame(backend,source.shape,epsilon,tau,rho,max_lag,records)
        if parent:
            (Path(parent)/'stream.jzp').write_bytes(payload)
        elapsed = time.perf_counter()-started
        expected = np.stack([v['rebuilt'] for v in records]).reshape(source.shape)
    decode_started = time.perf_counter()
    reconstruction = c.decode(payload,workdir)
    decode_seconds = time.perf_counter()-decode_started
    if reconstruction.tobytes() != expected.tobytes():
        raise RuntimeError('no-correction independent decoder differs from selected values')
    checks = c.audit(source,reconstruction,epsilon,tau,rho,max_lag)
    if checks['numerical_pass_windows'] != source.shape[0]:
        raise RuntimeError('no-correction final ERA contract failed')
    metadata = dict(schema='jointzip.fixed-trajectory-correction-ablation.v1',codec=backend,
        framework=True,identity=c.identity(backend),shape=list(source.shape),epsilon=epsilon,tau=tau,
        rho=rho,max_lag=max_lag,bytes=len(payload),original_bytes=source.nbytes,cr=source.nbytes/len(payload),
        inner_frame_bytes=inner_bytes,base_bytes=sum(len(v['base']) for v in records),correction_bytes=0,
        encode_seconds=elapsed,decode_seconds=decode_seconds,search_options=options,
        original_search_options=original_options,trace_sha256=digest,search_policy=options,
        formula='same recorded full-ERA formula-guided candidate trajectory; no new adaptive search',
        correction='disabled; reject unqualified native candidates; same exact lossless candidate',
        counts=counts,audit=checks,windows=evidence,independent_decode_bitwise_equal=True,
        has_reused_windows=counts['reused_windows']>0,has_candidate_reuse=counts['native_cache_hits']>0,
        timing_scope='fixed-trajectory replay including native calls, internal decode, ERA checks, framing; trace construction and final independent decode excluded',
        timing_reuse_note='cached replay is not standalone no-repair encoder speed; full-ERA trajectory was already computed',
        container='unchanged JZP1 independently decoded native records, whole-container Zstd level3',
        eligibility='original strict native cap AND ERA with original repair verifier 64*float64-eps tolerance; strict audit also reported',
        cap_rejections_scope='logical recorded proposals, with per-cap recheck even for effective ZFP aliases')
    if parent:
        (Path(parent)/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return c.Encoded(payload,reconstruction,metadata)
