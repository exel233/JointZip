"""Read-only imports of completed baseline and pilot window records.

Native payloads and decoded float32 values are reused, never historical CR or
timings.  The caller must serialize these records in its current complete
container, independently decode it, and calculate the new complete-file bytes.
This module only verifies files actually requested, with stat-keyed digest
caching; it does not walk or hash entire historical result trees.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXT4 = Path('/opt/jointzip-home/.local/cth_expansion_runs')
SIMPLE_NAME = 'vldb_simple_correction_20260929_v1'
PILOT_NAME = 'vldb_framework_pilot_20261014_v1'
NATIVE_NAMES = {'sz3': 'sz3', 'zfp': 'zfp', 'sperr': 'sperr', 'serf_xor': 'serf', 'sz2016': 'sz2016'}
TSL = struct.Struct('<4sIQ')
QMC = struct.Struct('<4sBBBBIIId')
XBC = struct.Struct('<4sBBHIIId')
SPR = struct.Struct('<4sIIId')
JZ_HEAD = struct.Struct('<4sBBIIIdddI')
JZ_RECORD = struct.Struct('<BIdII')


def _root(name, override=None):
    if override is not None:
        return Path(override)
    environment = os.environ.get('JOINTZIP_FULL_' + ('SIMPLE' if name == SIMPLE_NAME else 'PILOT') + '_ROOT')
    if environment:
        return Path(environment)
    candidate = EXT4 / name
    return candidate if candidate.is_dir() else ROOT / 'experiments/results' / name


@lru_cache(maxsize=65536)
def _digest_cached(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha(path):
    path = Path(path)
    st = path.stat()
    return _digest_cached(str(path.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=4096)
def _json_cached(path, size, mtime_ns):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _read(path):
    path = Path(path)
    st = path.stat()
    return _json_cached(str(path.resolve()), st.st_size, st.st_mtime_ns)


def _checked_file(root, relative, digest=None):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError('Historical artifact path escaped its result root')
    if not path.is_file():
        raise RuntimeError('Committed historical artifact missing: ' + str(path))
    if digest is not None and _sha(path) != digest:
        raise RuntimeError('Historical artifact changed: ' + str(path))
    return path


def _bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype == np.float32
            and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


@lru_cache(maxsize=256)
def _frozen_array(path, size, mtime_ns, key):
    with np.load(path, allow_pickle=False) as values:
        return np.ascontiguousarray(values[key])


def _input_array(path, key):
    st = path.stat()
    return _frozen_array(str(path.resolve()), st.st_size, st.st_mtime_ns, key)


def _source_check(root, entry, split, source):
    freeze = _read(root / 'freeze.json')
    old = next((v for v in freeze['entries']
                if (v['task'], v['sid']) == (entry['task'], entry['sid'])), None)
    if old is None or old['input_sha256'] != entry['input_sha256']:
        raise RuntimeError('Historical input binding differs: ' + entry['sid'])
    for name in ('mean', 'std', 'channel', 'true_length'):
        if old.get(name) != entry.get(name):
            raise RuntimeError('Historical input metadata differs: ' + name)
    path = _checked_file(root, old['input'], old['input_sha256'])
    key = 'val_source' if split == 'val' else 'source'
    original = _input_array(path, key)
    if not _bits(source, original):
        raise RuntimeError('Requested source does not equal the frozen ' + split + ' input')


def _length_streams(payload, offset, count, length_type, crc=False):
    result = []
    header = struct.Struct(length_type)
    for _ in range(count):
        if offset + header.size > len(payload):
            raise RuntimeError('Truncated historical row header')
        fields = header.unpack_from(payload, offset)
        size = fields[0]
        offset += header.size
        if size <= 0 or offset + size > len(payload):
            raise RuntimeError('Truncated historical native payload')
        value = payload[offset:offset + size]
        offset += size
        if crc and zlib.crc32(value) != fields[1]:
            raise RuntimeError('Historical native row CRC mismatch')
        result.append(value)
    if offset != len(payload):
        raise RuntimeError('Historical payload has unparsed trailing bytes')
    return result


def _native_rows(backend, payload, *, source_shape=None, requested_epsilon=None):
    """Return original independent native bodies, shape, epsilon and mode.

    SPERR's historical C=1 backend is SPECK1D_FLT, not the 2D C API on an
    n-by-1 tensor.  Its bodies may only be imported into the same 1D backend.
    """
    if backend == 'sz2016':
        # SZ1.1 has no outer batch/shape header: each historical record is
        # uint32_le native_length || unchanged official native member. The
        # shape and cap are supplied by the independently bound historical
        # input and condition row, then every body is decoded by the current
        # official decoder before it enters the window cache.
        if source_shape is None or len(source_shape) != 3 or source_shape[1] != 1:
            raise RuntimeError('SZ2016 import needs bound single-channel source shape')
        if requested_epsilon is None or not np.isfinite(requested_epsilon) or requested_epsilon <= 0:
            raise RuntimeError('SZ2016 import needs bound positive absolute epsilon')
        offset, streams = 0, []
        while offset < len(payload):
            if offset+4 > len(payload):
                raise RuntimeError('Truncated historical SZ2016 member length')
            size, = struct.unpack_from('<I',payload,offset)
            offset += 4
            if size <= 0 or offset+size > len(payload):
                raise RuntimeError('Truncated historical SZ2016 native member')
            streams.append(payload[offset:offset+size])
            offset += size
        w,c,t,epsilon = len(streams),1,source_shape[-1],requested_epsilon
        if w > source_shape[0]:
            raise RuntimeError('Historical SZ2016 payload exceeds source window count')
        mode = 'sz2016_bestfit_abs_1d'
    elif backend in ('sz3', 'zfp'):
        if len(payload) < QMC.size:
            raise RuntimeError('Truncated historical QMC1')
        magic, version, identity, dtype, reserved, w, c, t, epsilon = QMC.unpack_from(payload)
        if (magic, version, identity, dtype, reserved) != (b'QMC1', 1, 1 if backend == 'sz3' else 2, 1, 0):
            raise RuntimeError('Historical modern codec identity mismatch')
        streams = _length_streams(payload, QMC.size, w*c, '<Q')
        mode = 'sz3_abs_1d' if backend == 'sz3' else 'zfp_fixed_accuracy_1d'
    elif backend == 'serf_xor':
        if len(payload) < XBC.size:
            raise RuntimeError('Truncated historical XBC1')
        magic, version, identity, reserved, w, c, t, epsilon = XBC.unpack_from(payload)
        if (magic, version, identity, reserved) != (b'XBC1', 1, 1, 0):
            raise RuntimeError('Historical Serf-XOR identity mismatch')
        streams = _length_streams(payload, XBC.size, w*c, '<QI', crc=True)
        mode = 'serf_xor_32'
    elif backend == 'sperr':
        if len(payload) < SPR.size:
            raise RuntimeError('Truncated historical SPR1')
        magic, w, c, t, epsilon = SPR.unpack_from(payload)
        if magic != b'SPR1' or c != 1:
            raise RuntimeError('Only historical 1D SPERR rows are eligible')
        streams = _length_streams(payload, SPR.size, w, '<I')
        mode = 'sperr_speck1d_flt'
    else:
        raise ValueError('Unsupported historical backend: ' + backend)
    if min(w, c, t) <= 0 or c != 1:
        raise RuntimeError('Invalid historical dimensions')
    return streams, (w, c, t), float(epsilon), mode


def load_native(entry, split, backend, point, source, *, source_old_root=None):
    """Import all rows of one completed original native condition, or None.

    Missing results for an otherwise supported backend are an error: silently
    recomputing would conceal an incomplete reuse plan.  Serf-QT is new and
    returns None.  Return fields are records/reconstruction/lineage/native_mode.
    """
    if backend not in NATIVE_NAMES:
        return None
    root = _root(SIMPLE_NAME, source_old_root)
    old_method = NATIVE_NAMES[backend]
    row_path = root / 'compression' / entry['task'] / entry['sid'] / split / (old_method + '__' + point['label'] + '.json')
    row = _read(row_path)
    expected = (entry['task'], entry['sid'], entry['dataset'], split, old_method, point)
    actual = tuple(row[k] for k in ('task', 'sid', 'dataset', 'split', 'method', 'point'))
    if actual != expected or row.get('status') != 'complete':
        raise RuntimeError('Historical native condition mismatch/incomplete: ' + str(row_path))
    if row['input_sha256'] != entry['input_sha256']:
        raise RuntimeError('Historical native row input binding differs')
    _source_check(root, entry, split, source)
    if row['freeze_sha256'] != _sha(root / 'freeze.json'):
        raise RuntimeError('Historical native freeze binding differs')
    manifest = root / 'codec_manifest' / (old_method + '.json')
    if row['codec_manifest_sha256'] != _sha(manifest):
        raise RuntimeError('Historical native codec manifest binding differs')
    if backend == 'sz2016':
        from .codec import identity
        current = identity(backend)
        files = _read(manifest)['codec']['files']
        if (files.get('tools/sz2016/bin/sz2016_batch') != current['binary_sha256'] or
                files.get('tools/sz2016/sz2016.config') != current['configuration_sha256']):
            raise RuntimeError('Historical SZ2016 executable/config differs from current adapter')
    artifact = _checked_file(root, row['artifact'], row['artifact_sha256'])
    reconstructed = _checked_file(root, row['reconstruction'], row['reconstruction_sha256'])
    payload = artifact.read_bytes()
    if len(payload) != row['artifact_bytes']:
        raise RuntimeError('Historical complete-byte accounting changed')
    if payload.startswith(b'TSL1'):
        _, block, true_length = TSL.unpack_from(payload)
        if entry['task'] != 'anomaly' or block != 128 or true_length != entry['true_length']:
            raise RuntimeError('Historical true-length header differs')
        payload = payload[TSL.size:]
    from experiments.vldb.utility.artifacts import parse_scs2
    parsed = parse_scs2(payload)
    if parsed.codec_id != old_method:
        raise RuntimeError('Historical SCS2 codec differs')
    streams, modes = [], set()
    for packed in parsed.payloads:
        rows, shape, epsilon, mode = _native_rows(backend, packed,
            source_shape=source.shape, requested_epsilon=point['epsilon'])
        if shape[1:] != source.shape[1:] or epsilon != point['epsilon']:
            raise RuntimeError('Historical native shape/epsilon differs')
        streams.extend(rows)
        modes.add(mode)
    y = np.load(reconstructed, allow_pickle=False)
    if y.dtype != np.float32 or y.shape != source.shape or not np.isfinite(y).all():
        raise RuntimeError('Historical reconstruction shape/type differs')
    if len(streams) != len(source) or len(modes) != 1:
        raise RuntimeError('Historical native row coverage differs')
    records = []
    for base, original, restored in zip(streams, source[:, 0, :], y[:, 0, :]):
        error = float(np.max(np.abs(original.astype(np.float64) - restored.astype(np.float64))))
        records.append(dict(kind=1, cap=float(point['epsilon']), base=base, correction=b'', count=0,
                            rebuilt=restored.copy(), eligible=error <= point['epsilon'],
                            base_max_error=error, repair_candidates=0,
                            phase='historical_native', cache_hit=True))
    lineage = dict(kind='historical_native_payloads_reframed', backend=backend,
        original_method=old_method, original_root=str(root), original_row=str(row_path),
        original_row_sha256=_sha(row_path), original_artifact=str(artifact),
        original_artifact_sha256=row['artifact_sha256'], original_complete_bytes=row['artifact_bytes'],
        original_reconstruction=str(reconstructed), original_reconstruction_sha256=row['reconstruction_sha256'],
        original_codec_manifest=str(manifest), original_codec_manifest_sha256=row['codec_manifest_sha256'],
        original_input_sha256=entry['input_sha256'], windows=len(records),
        native_mode=next(iter(modes)), native_payload_bytes=sum(map(len, streams)),
        billing='Original SCS2 complete bytes are historical only; calculate current complete JZP bytes after reframing.',
        timing='No historical encoding time is charged as fresh encoding; separately time representative standalone calls.',
        independent_current_decode_required=True)
    return dict(records=records, reconstruction=y, lineage=lineage, native_mode=next(iter(modes)))


def _pilot_records(payload, reconstruction):
    import zstandard as zstd
    if not payload.startswith(b'JZP1'):
        raise RuntimeError('Historical pilot is not JZP1')
    inner = zstd.ZstdDecompressor().decompress(payload[4:])
    magic, version, codec_id, w, c, t, epsilon, tau, rho, lag = JZ_HEAD.unpack_from(inner)
    if magic != b'JZI1' or version != 1 or codec_id not in (1, 2) or c != 1:
        raise RuntimeError('Historical pilot header differs')
    if reconstruction.shape != (w, c, t) or reconstruction.dtype != np.float32:
        raise RuntimeError('Historical pilot reconstruction dimensions differ')
    offset, records = JZ_HEAD.size, []
    for index in range(w):
        kind, count, cap, nb, nc = JZ_RECORD.unpack_from(inner, offset)
        offset += JZ_RECORD.size
        if offset + nb + nc > len(inner) or kind not in (0, 1):
            raise RuntimeError('Truncated historical pilot record')
        base, correction = inner[offset:offset+nb], inner[offset+nb:offset+nb+nc]
        offset += nb + nc
        if (nc and correction[0] > 6) or (count and not nc):
            raise RuntimeError('Historical pilot correction contract differs')
        records.append(dict(kind=kind, count=count, cap=cap, base=base, correction=correction,
                            rebuilt=reconstruction[index, 0].copy(), eligible=True,
                            phase='historical_pilot', cache_hit=True))
    if offset != len(inner):
        raise RuntimeError('Historical pilot has trailing bytes')
    return records, dict(backend={1:'sz3', 2:'zfp'}[codec_id], shape=(w,c,t),
                         epsilon=epsilon, tau=tau, rho=rho, max_lag=lag)


def load_pilot(entry, split, backend, point, source, framework=True, *, pilot_root=None):
    """Import matching pilot windows, retaining official full-TEST positions.

    A full forecast panel yields complete=True and may reuse its entire file.
    UCR yields a mapping of 32 existing query records, not a full-TEST result;
    callers encode only missing indices and repack all records in full order.
    Pilot CPU model scores are not returned as current full-run CUDA results.
    """
    if split != 'test' or backend not in ('sz3', 'zfp'):
        return None
    root = _root(PILOT_NAME, pilot_root)
    if not (root / 'COMPLETE.json').is_file():
        raise RuntimeError('Historical pilot has no COMPLETE record')
    completion = _read(root / 'COMPLETE.json')
    if completion.get('complete') is not True:
        raise RuntimeError('Historical pilot was not completed')
    if completion.get('plan_sha256') != _sha(root / 'PLAN.json'):
        raise RuntimeError('Historical pilot plan changed after completion')
    plan = _read(root / 'PLAN.json')
    panel = next((p for p in plan['panels'] if p['sid'] == entry['sid']), None)
    if panel is None or point not in panel['points']:
        return None
    name = '__'.join((entry['sid'], point['label'], backend, 'jointzip' if framework else 'native'))
    row_path = root / 'rows' / (name + '.json')
    row = _read(row_path)
    if (row.get('status') != 'complete' or row['point'] != point or row['framework'] != framework
            or row['backend'] != backend or row['sid'] != entry['sid']):
        raise RuntimeError('Historical pilot row differs or is incomplete')
    if panel['source_sha256'] != entry['input_sha256']:
        raise RuntimeError('Historical pilot input binding differs')
    indices = np.asarray(panel['selected_test_indices'], dtype=np.int64)
    if len(indices) != len(set(indices.tolist())) or np.any(indices < 0) or np.any(indices >= len(source)):
        raise RuntimeError('Historical pilot indices are invalid')
    selected_source = np.load(root / ('source_' + entry['sid'] + '.npy'), allow_pickle=False)
    if not _bits(source[indices], selected_source):
        raise RuntimeError('Historical pilot selected input changed')
    artifact = _checked_file(root, row['artifact'])
    reconstruction_path = _checked_file(root, row['reconstruction'])
    reconstruction = np.load(reconstruction_path, allow_pickle=False)
    payload = artifact.read_bytes()
    if len(payload) != row['compressed_bytes']:
        raise RuntimeError('Historical pilot byte accounting differs')
    records, header = _pilot_records(payload, reconstruction)
    if (header['backend'], header['epsilon'], header['tau'], header['rho'], header['max_lag']) != (
            backend, point['epsilon'], point['tau'], point['rho'], 16):
        raise RuntimeError('Historical pilot constraints differ')
    evidence = row['codec_metadata']['windows']
    if len(evidence) != len(records) or len(records) != len(indices):
        raise RuntimeError('Historical pilot window evidence is incomplete')
    for record, item in zip(records, evidence):
        chosen = item['selected']
        for key in ('base_max_error', 'repair_candidates'):
            record[key] = chosen[key]
        if (chosen['base_bytes'], chosen['correction_bytes'], chosen['count'], chosen['kind']) != (
                len(record['base']), len(record['correction']), record['count'], record['kind']):
            raise RuntimeError('Historical pilot chosen-window evidence differs')
    complete = len(indices) == len(source) and np.array_equal(indices, np.arange(len(source)))
    lineage = dict(kind='pilot_full_condition' if complete else 'pilot_subset_windows',
        original_root=str(root), original_row=str(row_path), original_row_sha256=_sha(row_path),
        original_artifact=str(artifact), original_artifact_sha256=_sha(artifact),
        original_complete_bytes=len(payload), original_reconstruction=str(reconstruction_path),
        original_reconstruction_sha256=_sha(reconstruction_path), original_input_sha256=entry['input_sha256'],
        original_plan_sha256=_sha(root/'PLAN.json'), full_test_indices=indices.tolist(), windows=len(records),
        full_condition=complete, billing='Subset CR is not full-TEST CR; reframe all selected and newly computed records.',
        consumer_reuse='Pilot CPU neural metrics are not full-run CUDA results; separate identity-based prediction reuse is required.',
        independent_current_decode_required=True)
    return dict(records=records, indices=indices.tolist(), records_by_index=dict(zip(indices.tolist(), records)),
                reconstruction=reconstruction, evidence=evidence, metadata=row['codec_metadata'],
                complete=complete, payload=payload if complete else None, lineage=lineage)
