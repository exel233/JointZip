"""Fixed-backend framework expansion, retaining frozen formula and simple repair.

Default schedules reproduce the Oct14 pilot for SZ3/ZFP and the Oct13 guided
schedule for SPERR/Serf. Increasing candidate_budget extends the exact default
prefix. WindowCache reuses complete previously computed windows only under
identical source, backend, constraints, policy and candidate budget. Reuse is
reported separately and must never be interpreted as fresh encoding speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time

import numpy as np
import zstandard as zstd

from experiments.vldb.main_benchmark import modern_codecs as modern

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPAIR = Path(os.environ.get('JOINTZIP_PILOT_REPAIR', str(ROOT / 'tools/qpet_correlation_guided_20261013_v1/bin/qpet_correlation_batch')))
ANCHORS = Path(os.environ.get('JOINTZIP_PILOT_ANCHORS', str(ROOT / 'experiments/vldb/framework_pilot_20261014/native/anchors')))
HEAD = struct.Struct('<4sBBIIIdddI')
RECORD = struct.Struct('<BIdII')  # kind, repair count, selected cap, base bytes, correction bytes
BRIDGE = Path(os.environ.get('JOINTZIP_FULL_BRIDGE', str(HERE/'native/fixed_backend')))
BRIDGE_MANIFEST = Path(os.environ.get('JOINTZIP_FULL_BRIDGE_MANIFEST', str(HERE/'native/manifest.json')))
CODECS = {'sz3': 1, 'zfp': 2, 'sperr': 3, 'serf_xor': 4, 'serf_qt': 5}
DEFAULTS = {name: (22,12,512.) if name in ('sz3','zfp','sperr') else (8,5,64.) for name in CODECS}
NAMES = {value: key for key, value in CODECS.items()}


@dataclass
class Encoded:
    payload: bytes
    reconstruction: np.ndarray
    metadata: dict


class WindowCache:
    """In-memory reuse of *complete* searches, never shortened budget runs.

    add(source, metadata, payload, provenance) imports a completed pilot/full
    artifact. add_records() supports a verified historical native adapter.
    Every import replays stored bytes, checks correspondence to saved values,
    and keys source bytes and all options. This is a replay cost, not encoding.
    """
    def __init__(self):
        self.entries={}
        self.imported_windows=0
        self.import_seconds=0.

    @staticmethod
    def _key(codec,row,epsilon,tau,rho,max_lag,framework,options):
        backend=identity(codec)
        implementation=(backend['binary_sha256'],backend.get('upstream_commit'),
                        backend.get('official_library_sha256'))
        return (codec,implementation,row.size,hashlib.sha256(row.tobytes()).digest(),
                float(epsilon),float(tau),float(rho),int(max_lag),bool(framework),
                options['policy'],options['total'],options['initial'],options['divisor'])

    def get(self,codec,row,epsilon,tau,rho,max_lag,framework,options):
        return self.entries.get(self._key(codec,row,epsilon,tau,rho,max_lag,framework,options))

    def put(self,codec,row,epsilon,tau,rho,max_lag,framework,options,record,evidence,provenance):
        key=self._key(codec,row,epsilon,tau,rho,max_lag,framework,options)
        value=dict(record=record,evidence=evidence,provenance=provenance)
        old=self.entries.get(key)
        if old is not None:
            for field in ('kind','count','cap','base','correction'):
                if old['record'][field]!=record[field]:
                    raise RuntimeError('conflicting complete-window reuse artifact')
            if not np.array_equal(old['record']['rebuilt'].view(np.uint32),record['rebuilt'].view(np.uint32)):
                raise RuntimeError('conflicting complete-window reconstruction')
            return
        self.entries[key]=value

    def add(self,source,metadata,payload,provenance,workdir=None):
        """Import a completed pilot/full artifact; source is the exact original."""
        if isinstance(metadata,(str,Path)):
            metadata=json.loads(Path(metadata).read_text())
        if isinstance(payload,(str,Path)):
            payload=Path(payload).read_bytes()
        codec=metadata['codec'];framework=metadata['framework']
        if metadata['schema'] not in ('jointzip.fixed-backend-pilot.v1','jointzip.fixed-backend-full.v1'):
            raise ValueError('unrecognized reuse artifact schema')
        if metadata['identity']['binary_sha256']!=identity(codec)['binary_sha256']:
            raise RuntimeError('reuse backend binary differs')
        if metadata.get('bytes')!=len(payload) or metadata['shape']!=list(source.shape):
            raise RuntimeError('reuse size or shape differs')
        if not metadata.get('independent_decode_bitwise_equal'):
            raise RuntimeError('reuse requires a completed independent decoder check')
        if metadata['schema']=='jointzip.fixed-backend-pilot.v1':
            if codec not in ('sz3','zfp'):
                raise ValueError('unexpected historical pilot backend')
            options=search_options(codec,framework)
        else:
            options=metadata['search_options']
        header,records=unframe(payload)
        if header['codec']!=codec or header['shape']!=tuple(source.shape):
            raise RuntimeError('reuse header mismatch')
        for name in ('epsilon','tau','rho','max_lag'):
            if header[name]!=metadata[name]:
                raise RuntimeError('reuse constraints differ in stream and metadata')
        self.add_records(codec,source,metadata['epsilon'],metadata['tau'],metadata['rho'],
            metadata['max_lag'],framework,records,metadata['windows'],provenance,
            search_policy='formula' if options['policy']=='native' else options['policy'],
            candidate_budget=None if not framework else options['total'],workdir=workdir)

    def add_records(self,codec,source,epsilon,tau,rho,max_lag,framework,records,
                    evidence,provenance,*,search_policy='formula',candidate_budget=None,workdir=None):
        started=time.perf_counter()
        if source.dtype!=np.float32 or source.ndim!=3 or source.shape[1]!=1 or len(records)!=source.shape[0]:
            raise ValueError('reuse source/record shape mismatch')
        options=search_options(codec,framework,search_policy,candidate_budget)
        if evidence is not None and len(evidence)!=len(records):
            raise ValueError('reuse evidence length mismatch')
        # Reframe and independently decode without the original source.
        payload,_=_frame(codec,source.shape,epsilon,tau,rho,max_lag,records)
        rebuilt=decode(payload,workdir)
        if framework and audit(source,rebuilt,epsilon,tau,rho,max_lag)['numerical_pass_windows']!=len(records):
            raise RuntimeError('imported framework window fails final constraints')
        for index,(row,record) in enumerate(zip(source[:,0,:],records)):
            record=dict(record)
            if 'rebuilt' in record and not np.array_equal(record['rebuilt'].view(np.uint32),rebuilt[index,0].view(np.uint32)):
                raise RuntimeError('reuse saved reconstruction disagrees with native decoder')
            record['rebuilt']=rebuilt[index,0].copy()
            record.setdefault('eligible',True)
            record.setdefault('base_max_error',float(np.max(np.abs(row.astype(float)-record['rebuilt'].astype(float)))))
            record.setdefault('repair_candidates',0)
            record.setdefault('phase','reused_native' if not framework else 'reused_complete')
            record.setdefault('cache_hit',False)
            if evidence is None:
                if framework:
                    raise ValueError('framework search reuse needs complete search evidence')
                desc={k:v for k,v in record.items() if k not in ('base','correction','rebuilt')}
                desc.update(base_bytes=len(record['base']),correction_bytes=len(record['correction']))
                item=dict(window=index,anchor=epsilon,selected=desc,candidates=[desc])
            else:
                item=copy.deepcopy(evidence[index])
                selected=item['selected']
                if any(selected[key]!=record[key] for key in ('kind','count','cap')) or selected['base_bytes']!=len(record['base']) or selected['correction_bytes']!=len(record['correction']):
                    raise RuntimeError('reuse selected evidence disagrees with saved stream')
            self.put(codec,row,epsilon,tau,rho,max_lag,framework,options,record,item,provenance)
            self.imported_windows+=1
        self.import_seconds+=time.perf_counter()-started


class CandidateCache:
    """Optional same-candidate reuse for budget-density experiments only.

    Proposal order and observations stay policy-local. A cached value is the
    identical backend call plus the same correction, never a result at another
    cap. Timings with this cache are incremental work, not standalone latency.
    """
    def __init__(self):
        self.entries={}

    @staticmethod
    def _key(codec,row,epsilon,tau,rho,lag,key):
        return (codec,identity(codec)['binary_sha256'],row.size,
                hashlib.sha256(row.tobytes()).digest(),float(epsilon),float(tau),
                float(rho),int(lag),float(key))

    def get(self,codec,row,epsilon,tau,rho,lag,key):
        value=self.entries.get(self._key(codec,row,epsilon,tau,rho,lag,key))
        return None if value is None else dict(value)

    def put(self,codec,row,epsilon,tau,rho,lag,key,value):
        self.entries[self._key(codec,row,epsilon,tau,rho,lag,key)]=dict(value)

    def save(self,path):
        items=[]
        for key,value in self.entries.items():
            key=list(key);key[3]=key[3].hex()
            record={k:v for k,v in value.items() if k not in ('base','correction','rebuilt')}
            for name in ('base','correction'):
                record[name]=base64.b64encode(value[name]).decode('ascii')
            record['rebuilt']=base64.b64encode(value['rebuilt'].astype('<f4',copy=False).tobytes()).decode('ascii')
            items.append([key,record])
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.writing')
        temp.write_text(json.dumps(dict(schema='jointzip.density-candidates.v1',entries=items),allow_nan=False))
        temp.replace(path)

    @classmethod
    def load(cls,path):
        value=json.loads(Path(path).read_text());out=cls()
        if value['schema']!='jointzip.density-candidates.v1':
            raise ValueError('unrecognized density cache')
        for key,record in value['entries']:
            key[3]=bytes.fromhex(key[3]);key=tuple(key)
            if identity(key[0])['binary_sha256']!=key[1]:
                raise RuntimeError('density cache backend binary changed')
            record['base']=base64.b64decode(record['base'],validate=True)
            record['correction']=base64.b64decode(record['correction'],validate=True)
            record['rebuilt']=np.frombuffer(base64.b64decode(record['rebuilt'],validate=True),dtype='<f4').copy()
            if record['rebuilt'].size!=key[2] or not np.isfinite(record['rebuilt']).all():
                raise ValueError('invalid density cached reconstruction')
            out.entries[key]=record
        return out


def load_pilot_cache(source,metadata,payload,provenance=None,workdir=None):
    result=WindowCache()
    result.add(source,metadata,payload,provenance or {'kind':'completed_pilot'},workdir)
    return result


@lru_cache(maxsize=5)
def identity(codec):
    if codec in ('sz3','zfp'):
        return modern._identity(codec)
    if codec not in CODECS:
        raise ValueError(codec)
    manifest=json.loads(BRIDGE_MANIFEST.read_text())
    if hashlib.sha256(BRIDGE.read_bytes()).hexdigest()!=manifest['binary_sha256']:
        raise RuntimeError('fixed backend bridge differs from build manifest')
    return dict(binary=str(BRIDGE),binary_sha256=manifest['binary_sha256'],
        upstream_commit=manifest['upstreams']['SPERR' if codec=='sperr' else 'Serf'],
        official_library_sha256=manifest['libraries']['SPERR' if codec=='sperr' else 'Serf'],
        version='0.8.5' if codec=='sperr' else 'b38450b',codec=codec,
        native_shape='official SPECK1D_FLT, Conditioner+SPECK body; historical native baseline mode' if codec=='sperr' else 'official single-series Serf32',
        dtype='<f4',contract='pointwise_absolute_only',threads=1)


def search_options(codec, framework=True, search_policy='formula', candidate_budget=None):
    if search_policy not in ('formula','noformula'):
        raise ValueError('unknown search policy')
    limit,initial,divisor=DEFAULTS[codec]
    if candidate_budget is not None:
        if isinstance(candidate_budget,bool) or int(candidate_budget)!=candidate_budget or candidate_budget<limit:
            raise ValueError('candidate budget must be integer >= default; extensions preserve the default prefix')
        limit=int(candidate_budget)
    return dict(policy=search_policy if framework else 'native',total=limit if framework else 1,
                initial=initial if framework else 1,divisor=divisor if framework else 0)



def _run(command):
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if os.environ.get('JOINTZIP_PILOT_NATIVE_LIBS'):
        environment['LD_LIBRARY_PATH'] = os.environ['JOINTZIP_PILOT_NATIVE_LIBS']
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True,
                            timeout=300, env=environment)
    if result.returncode:
        raise RuntimeError('native command failed: ' + ' '.join(map(str, command)) + '\n' +
                           result.stderr[-3000:] + result.stdout[-1000:])
    return result.stdout


def _down(value):
    v = np.float32(value)
    if float(v) > value:
        v = np.nextafter(v, np.float32(0))
    return float(v)


def _mid(left, right):
    if not 0 < left < right:
        return 0.
    middle = _down(math.exp((math.log(left) + math.log(right)) / 2))
    if middle <= left:
        middle = float(np.nextafter(np.float32(left), np.float32(right)))
    return middle if left < middle < right else 0.


def _native_encode(codec, row, cap, folder):
    raw, compressed, decoded = folder/'input.f32', folder/'base.native', folder/'base.f32'
    row.tofile(raw)
    binary = identity(codec)['binary']
    if codec == 'sz3':
        argv = [binary, 'c', str(raw), str(compressed), str(len(row)), format(cap, '.17g')]
    elif codec == 'zfp':
        argv = [binary, '-h', '-x', 'serial', '-f', '-1', str(len(row)), '-a',
                format(cap, '.17g'), '-i', str(raw), '-z', str(compressed)]
    else:
        native_cap = _down(cap) if codec.startswith('serf_') else cap
        argv=[binary,'c',codec,str(raw),str(compressed),str(len(row)),format(native_cap,'.17g')]
    (modern._run if codec in ('sz3','zfp') else _run)(argv)
    payload = compressed.read_bytes()
    base = _native_decode(codec, payload, len(row), folder)
    return payload, base


def _native_decode(codec, payload, length, folder):
    compressed, decoded = folder/'base.native', folder/'base.f32'
    compressed.write_bytes(payload)
    binary = identity(codec)['binary']
    if codec == 'sz3':
        argv = [binary, 'd', str(compressed), str(decoded)]
    elif codec == 'zfp':
        argv = [binary, '-h', '-x', 'serial', '-z', str(compressed), '-o', str(decoded)]
    else:
        argv=[binary,'d',codec,str(compressed),str(decoded),str(length),'1']
    (modern._run if codec in ('sz3','zfp') else _run)(argv)
    result = np.fromfile(decoded, dtype='<f4')
    if result.size != length or not np.isfinite(result).all():
        raise RuntimeError('invalid native reconstruction')
    return result


def _repair(row, base, epsilon, tau, rho, lag, folder):
    source, bp = folder/'original.f32', folder/'repair_base.f32'
    cp, rp = folder/'correction.bin', folder/'repaired.f32'
    row.tofile(source)
    base.tofile(bp)
    stdout = _run([REPAIR, 'repair-external', source, bp, cp, rp, 1, len(row),
                   format(epsilon, '.17g'), format(tau, '.17g'), format(rho, '.17g'),
                   lag, 'posthoc_greedy', '7,15,31,63,127'])
    fields = {key: value for key, value in re.findall(r'(\w+)=([^\s]+)', stdout)}
    payload = cp.read_bytes()
    if payload and payload[0] > 6:
        raise RuntimeError('complex correction encoding entered simple pilot')
    rebuilt = np.fromfile(rp, dtype='<f4')
    if rebuilt.shape != row.shape or not np.isfinite(rebuilt).all():
        raise RuntimeError('invalid repaired reconstruction')
    return payload, rebuilt, int(fields['corrections']), int(fields['candidates'])


def _pearson(x, y):
    x, y = np.asarray(x, dtype=np.longdouble), np.asarray(y, dtype=np.longdouble)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/denominator) if denominator > 1e-15 else 0.


def audit(source, reconstruction, epsilon, tau, rho, max_lag):
    rows = []
    for original, restored in zip(source.reshape(-1, source.shape[-1]),
                                  reconstruction.reshape(-1, source.shape[-1])):
        centered = original.astype(np.longdouble)
        centered -= centered.mean()
        corr = (float(np.array_equal(original, restored)) if np.sum(centered*centered) <= 1e-15
                else _pearson(original, restored))
        acf = max((abs(_pearson(original[:-k], original[k:]) -
                       _pearson(restored[:-k], restored[k:])) for k in range(1, max_lag+1)), default=0.)
        error = float(np.max(np.abs(original.astype(np.float64)-restored.astype(np.float64))))
        tolerance = 64*np.finfo(np.float64).eps
        rows.append(dict(max_abs_error=error, pearson=corr, max_acf_error=acf,
                         strict_pass=bool(error <= epsilon and (tau < 0 or acf <= tau) and (rho < 0 or corr >= rho)),
                         numerical_pass=bool(error <= epsilon+tolerance and (tau < 0 or acf <= tau+tolerance)
                         and (rho < 0 or corr+tolerance >= rho))))
    return dict(windows=len(rows), max_abs_error=max(r['max_abs_error'] for r in rows),
                min_pearson=min(r['pearson'] for r in rows),
                max_acf_error=max(r['max_acf_error'] for r in rows),
                strict_pass_windows=sum(r['strict_pass'] for r in rows),
                numerical_pass_windows=sum(r['numerical_pass'] for r in rows), rows=rows)


def _frame(codec, shape, epsilon, tau, rho, max_lag, records):
    inner = bytearray(HEAD.pack(b'JZI1', 1, CODECS[codec], *shape, epsilon, tau, rho, max_lag))
    for record in records:
        base, correction = record['base'], record['correction']
        inner.extend(RECORD.pack(record['kind'], record['count'], record['cap'], len(base), len(correction)))
        inner.extend(base)
        inner.extend(correction)
    return b'JZP1' + zstd.ZstdCompressor(level=3).compress(bytes(inner)), len(inner)


def unframe(payload):
    """Extract independently decodable native/correction records, no search."""
    if not payload.startswith(b'JZP1'):
        raise ValueError('not JZP1')
    inner=zstd.ZstdDecompressor().decompress(payload[4:])
    if len(inner)<HEAD.size:
        raise ValueError('truncated JZP1 header')
    magic,version,codec_id,w,c,t,epsilon,tau,rho,lag=HEAD.unpack_from(inner)
    if magic!=b'JZI1' or version!=1 or codec_id not in NAMES or c!=1 or not w or not t:
        raise ValueError('invalid JZP1 header')
    offset=HEAD.size;records=[]
    for _ in range(w*c):
        if offset+RECORD.size>len(inner):
            raise ValueError('truncated JZP1 record')
        kind,count,cap,nb,nc=RECORD.unpack_from(inner,offset);offset+=RECORD.size
        if kind not in (0,1) or offset+nb+nc>len(inner):
            raise ValueError('invalid JZP1 record')
        base,correction=inner[offset:offset+nb],inner[offset+nb:offset+nb+nc];offset+=nb+nc
        if correction and correction[0]>6:
            raise ValueError('complex correction in reused stream')
        records.append(dict(kind=kind,count=count,cap=cap,base=base,correction=correction))
    if offset!=len(inner):
        raise ValueError('trailing JZP1 bytes')
    return dict(codec=NAMES[codec_id],shape=(w,c,t),epsilon=epsilon,tau=tau,rho=rho,max_lag=lag),records


def decode(payload, workdir=None):
    """Independent source-free JZP1 decode, including persisted simple repairs."""
    if not payload.startswith(b'JZP1'):
        raise ValueError('not JZP1')
    inner = zstd.ZstdDecompressor().decompress(payload[4:])
    magic, version, codec_id, w, c, t, epsilon, tau, rho, lag = HEAD.unpack_from(inner)
    if magic != b'JZI1' or version != 1 or codec_id not in NAMES or c != 1:
        raise ValueError('invalid JZP1 header')
    codec, offset, rows = NAMES[codec_id], HEAD.size, []
    parent = None if workdir is None else str(Path(workdir))
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='decode_', dir=parent) as tmp:
        folder = Path(tmp)
        for _ in range(w*c):
            if offset+RECORD.size > len(inner):
                raise ValueError('truncated record')
            kind, count, cap, nb, nc = RECORD.unpack_from(inner, offset)
            offset += RECORD.size
            if offset+nb+nc > len(inner):
                raise ValueError('truncated payload')
            base_payload, correction = inner[offset:offset+nb], inner[offset+nb:offset+nb+nc]
            offset += nb+nc
            if kind == 0:
                base = np.frombuffer(zstd.ZstdDecompressor().decompress(base_payload), dtype='<f4').copy()
            elif kind == 1:
                base = _native_decode(codec, base_payload, t, folder)
            else:
                raise ValueError('invalid base kind')
            if nc:
                if correction[0] > 6:
                    raise ValueError('complex correction format')
                bp, cp, rp = folder/'apply.f32', folder/'apply.bin', folder/'decoded.f32'
                base.tofile(bp)
                cp.write_bytes(correction)
                _run([REPAIR, 'apply-external', bp, cp, rp, count, c, t])
                base = np.fromfile(rp, dtype='<f4')
            elif count:
                raise ValueError('missing correction')
            if base.size != t or not np.isfinite(base).all():
                raise ValueError('bad decoded row')
            rows.append(base)
    if offset != len(inner):
        raise ValueError('trailing bytes')
    return np.stack(rows).reshape(w,c,t)


def encode(codec, source, epsilon, tau, rho, max_lag=16, framework=True, workdir=None, *,
           search_policy='formula', candidate_budget=None, warm_cache=None,window_callback=None,
           candidate_cache=None):
    """Return persisted bytes, decoded values and full per-window search evidence."""
    if codec not in CODECS:
        raise ValueError(codec)
    if not isinstance(source, np.ndarray) or source.dtype != np.float32 or source.ndim != 3 or source.shape[1] != 1:
        raise ValueError('expected float32 [W,1,T]')
    if not np.isfinite(source).all() or not 0 < max_lag < source.shape[-1]-1 or epsilon <= 0:
        raise ValueError('invalid source/constraints')
    source = np.ascontiguousarray(source)
    options=search_options(codec,framework,search_policy,candidate_budget)
    identity(codec)  # Identity check cached outside reported encoding time.
    parent = None if workdir is None else str(Path(workdir))
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    records, evidence = [], []
    counts = dict(logical_probes=0, native_probes=0, cache_hits=0, cap_rejections=0,
                  repair_target_candidates=0, correction_values=0, lossless_windows=0,
                  reused_windows=0,reused_native_probes=0,reused_logical_probes=0,fresh_windows=0,
                  cross_search_candidate_hits=0)
    reuse_seconds=0.
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='encode_', dir=parent) as tmp:
        folder = Path(tmp)
        anchors = np.full(source.shape[0], epsilon, dtype=np.float64)
        if framework and search_policy=='formula':
            ip, op = folder/'sources.f32', folder/'anchors.f64'
            source.tofile(ip)
            _run([ANCHORS, ip, op, source.shape[0], source.shape[-1], format(epsilon,'.17g'),
                  format(tau,'.17g'), format(rho,'.17g'), max_lag])
            anchors = np.fromfile(op, dtype='<f8')
            if anchors.size != source.shape[0]:
                raise RuntimeError('bad anchor count')
        if framework and search_policy=='noformula':
            anchors.fill(epsilon/math.sqrt(options['divisor']))
        for index, row in enumerate(source[:,0,:]):
            reuse_started=time.perf_counter()
            old=None if warm_cache is None else warm_cache.get(codec,row,epsilon,tau,rho,max_lag,framework,options)
            reuse_seconds+=time.perf_counter()-reuse_started
            if old is not None:
                chosen,evidence_row=old['record'],copy.deepcopy(old['evidence'])
                evidence_row.update(window=index,reused=True,reuse_provenance=old['provenance'])
                records.append(chosen);evidence.append(evidence_row)
                counts['reused_windows']+=1
                counts['reused_logical_probes']+=sum(r['kind']==1 for r in evidence_row.get('candidates',[]))
                counts['reused_native_probes']+=sum(r['kind']==1 and not r.get('cache_hit',False) for r in evidence_row.get('candidates',[]))
                counts['correction_values']+=chosen['count']
                counts['lossless_windows']+=int(chosen['kind']==0)
                if window_callback is not None:
                    window_callback(index,chosen,evidence_row,options)
                continue
            counts['fresh_windows']+=1
            cache, observed, candidates = {}, [], []
            anchor = float(anchors[index])
            def probe(cap, phase):
                counts['logical_probes'] += 1
                # Official fixed-accuracy ZFP quantizes tolerance to a power of two.
                # Identical effective configurations are cached, while the logical
                # scheduler still follows the unchanged 22-proposal policy.
                key = math.ldexp(1., math.floor(math.log2(cap))) if codec == 'zfp' else cap
                if candidate_cache is not None and framework and key not in cache:
                    saved=candidate_cache.get(codec,row,epsilon,tau,rho,max_lag,key)
                    if saved is not None:
                        cache[key]=saved
                        counts['cross_search_candidate_hits']+=1
                if key in cache:
                    counts['cache_hits'] += 1
                    entry = dict(cache[key], cap=cap, phase=phase, cache_hit=True)
                    entry['eligible'] = entry['base_max_error'] <= cap
                    if framework and entry['eligible'] and not cache[key]['eligible']:
                        correction, rebuilt, count, tested = _repair(row, entry['rebuilt'], epsilon, tau, rho, max_lag, folder)
                        counts['repair_target_candidates'] += tested
                        entry.update(correction=correction, rebuilt=rebuilt, count=count, repair_candidates=tested)
                        cache[key] = dict(entry)
                else:
                    counts['native_probes'] += 1
                    base, rebuilt = _native_encode(codec, row, cap, folder)
                    error = float(np.max(np.abs(row.astype(np.float64)-rebuilt.astype(np.float64))))
                    eligible = error <= cap
                    correction, count, tested = b'', 0, 0
                    if framework and eligible:
                        correction, rebuilt, count, tested = _repair(row, rebuilt, epsilon, tau, rho, max_lag, folder)
                    counts['cap_rejections'] += int(not eligible)
                    counts['repair_target_candidates'] += tested
                    entry = dict(kind=1, cap=cap, base=base, correction=correction, count=count,
                                 rebuilt=rebuilt, eligible=eligible, base_max_error=error, repair_candidates=tested)
                    cache[key] = entry
                    entry = dict(entry, phase=phase, cache_hit=False)
                if candidate_cache is not None and framework:
                    candidate_cache.put(codec,row,epsilon,tau,rho,max_lag,key,cache[key])
                candidates.append(entry)
                observed.append((cap, entry))
                return entry
            if not framework:
                chosen = probe(float(epsilon), 'native')
            else:
                if anchor > 0 and _down(anchor) > 0:
                    lower, upper = min(anchor/4., epsilon/options['divisor']), epsilon
                    seeds = []
                    for value in (lower, anchor/2., anchor, anchor*2., upper):
                        cap = _down(min(max(value, lower), upper))
                        if cap > 0 and cap not in seeds:
                            seeds.append(cap)
                    while len(seeds) < options['initial']:
                        ordered = sorted(seeds)
                        gaps = [(math.log(b)-math.log(a), _mid(a,b)) for a,b in zip(ordered[:-1], ordered[1:])]
                        gaps = [(width, midpoint) for width,midpoint in gaps if midpoint > 0]
                        if not gaps:
                            break
                        seeds.append(max(gaps, key=lambda x:x[0])[1])
                    for cap in seeds:
                        probe(cap, 'seed')
                    while len(observed) < options['total']:
                        ordered = sorted(observed, key=lambda x:x[0])
                        def cost(entry):
                            return len(entry['base'])+len(entry['correction']) if entry['eligible'] else float('inf')
                        gaps = [(min(cost(x),cost(y)), -(math.log(b)-math.log(a)), a, _mid(a,b))
                                for (a,x),(b,y) in zip(ordered[:-1],ordered[1:]) if _mid(a,b)>0]
                        if not gaps:
                            break
                        probe(min(gaps, key=lambda x:x[:3])[3], 'refine')
                candidates.append(dict(kind=0, cap=epsilon,
                    base=zstd.ZstdCompressor(level=3).compress(row.tobytes()), correction=b'',
                    count=0, rebuilt=row.copy(), eligible=True, phase='lossless',cache_hit=False,
                    base_max_error=0.,repair_candidates=0))
                chosen = min((r for r in candidates if r['eligible']),
                             key=lambda r:(len(r['base'])+len(r['correction']),-r['cap']))
            records.append(chosen)
            counts['correction_values'] += chosen['count']
            counts['lossless_windows'] += int(chosen['kind']==0)
            def describe(r):
                return {**{k:v for k,v in r.items() if k not in ('base','correction','rebuilt')},
                        'base_bytes':len(r['base']), 'correction_bytes':len(r['correction'])}
            evidence.append(dict(window=index,anchor=anchor, selected=describe(chosen),
                                 candidates=[describe(r) for r in candidates],reused=False))
            if warm_cache is not None:
                warm_cache.put(codec,row,epsilon,tau,rho,max_lag,framework,options,chosen,evidence[-1],
                               {'kind':'earlier_window_in_current_process'})
            if window_callback is not None:
                window_callback(index,chosen,evidence[-1],options)
        payload, inner_bytes = _frame(codec, source.shape, epsilon, tau, rho, max_lag, records)
        if parent:
            (Path(parent)/'stream.jzp').write_bytes(payload)
        encode_seconds = time.perf_counter()-started
        expected = np.stack([r['rebuilt'] for r in records]).reshape(source.shape)
    decode_started = time.perf_counter()
    reconstruction = decode(payload, workdir)
    decode_seconds = time.perf_counter()-decode_started
    if not np.array_equal(expected.view(np.uint32), reconstruction.view(np.uint32)):
        raise RuntimeError('persisted decoder does not reproduce encoder reconstruction')
    checks = audit(source,reconstruction,epsilon,tau,rho,max_lag)
    if framework and checks['numerical_pass_windows'] != source.shape[0]:
        raise RuntimeError('framework failed final constraints')
    metadata = dict(schema='jointzip.fixed-backend-full.v1',codec=codec,framework=framework,
                    identity=identity(codec), shape=list(source.shape),epsilon=epsilon,tau=tau,rho=rho,max_lag=max_lag,
                    bytes=len(payload),original_bytes=source.nbytes,cr=source.nbytes/len(payload),
                    inner_frame_bytes=inner_bytes,base_bytes=sum(len(r['base']) for r in records),
                    correction_bytes=sum(len(r['correction']) for r in records),
                    encode_seconds=encode_seconds,decode_seconds=decode_seconds,reuse_lookup_seconds=reuse_seconds,
                    search_options=options,has_reused_windows=counts['reused_windows']>0,
                    has_candidate_reuse=counts['cross_search_candidate_hits']>0,
                    timing_reuse_note='encoding includes replay/framing but reused windows omit historical search; never compare this as fresh encoder speed',
                    timing_scope='formula, candidate codec processes+I/O, internal decode, simple repair, complete framing and stream write; independent decode/audit excluded',
                    container='JZP1, independent native row payloads + simple repair, whole-container Zstd level 3',
                    search_policy=options,formula='unchanged Oct13 CorrelationQoI anchors; posthoc acceptance, not nonlinear ACF priori guarantee',
                    correction='frozen repair-external legacy; greedy, original + denominators7,15,31,63,127; formats0--6',
                    cap_rejections_scope='unique physical backend calls; logical cache hits are individually rechecked',
                    counts=counts,audit=checks,windows=evidence,independent_decode_bitwise_equal=True)
    if parent:
        (Path(parent)/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return Encoded(payload,reconstruction,metadata)
