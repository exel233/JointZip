"""Common, source-independent metadata framing for saved codec streams.

JCF1 uses the same JSON data schema as JZF1. Its payload tag dispatches
between unchanged JZP1 JointZip bytes and ZLC1 + a complete NCS collection
of lossless ZSL1 batches. No scientific encoder is called here.
"""
from __future__ import annotations
import json
import struct
from pathlib import Path
import numpy as np
import zstandard as zstd

HEADER = struct.Struct('<4sI')
FIELDS = {'version', 'dtype', 'shape', 'true_count', 'mean', 'std', 'channel'}

def frame(metadata_json: bytes, payload: bytes) -> bytes:
    data = HEADER.pack(b'JCF1', len(metadata_json)) + metadata_json + payload
    parse(data)
    return data

def parse(data: bytes):
    if len(data) < 13:
        raise ValueError('Truncated common file')
    magic, n = HEADER.unpack_from(data)
    if magic != b'JCF1' or not 0 < n <= 65536 or 8+n+4 > len(data):
        raise ValueError('Invalid common file header')
    header, payload = data[8:8+n], data[8+n:]
    meta = json.loads(header)
    if set(meta) != FIELDS or meta['version'] != 1 or meta['dtype'] != '<f4':
        raise ValueError('Unsupported metadata schema')
    if len(meta['shape']) != 3 or any(not isinstance(v,int) or v<=0 for v in meta['shape']):
        raise ValueError('Invalid dimensions')
    if not 0 < meta['true_count'] <= int(np.prod(meta['shape'])):
        raise ValueError('Invalid true sample count')
    if not np.isfinite(meta['mean']) or not np.isfinite(meta['std']) or meta['std'] <= 0:
        raise ValueError('Invalid normalization')
    if not isinstance(meta['channel'], str) or not meta['channel']:
        raise ValueError('Missing channel')
    if payload[:4] not in (b'JZP1',b'ZLC1'):
        raise ValueError('Unknown payload codec')
    return meta, header, payload

def decompress_exact(frame: bytes, expected_size: int | None = None) -> bytes:
    decoder = zstd.ZstdDecompressor().decompressobj()
    raw = decoder.decompress(frame)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Zstd frame not consumed exactly')
    if expected_size is not None and len(raw) != expected_size:
        raise ValueError('Zstd size mismatch')
    return raw

def unpack_collection(collection: bytes):
    magic, n = struct.unpack_from('<4sI',collection)
    if magic not in (b'NCS1',b'NCS2') or n < 1:
        raise ValueError('Invalid lossless collection')
    offset, batches = 8, []
    for _ in range(n):
        if magic == b'NCS2':
            mode, raw_n, stored_n = struct.unpack_from('<BQQ',collection,offset)
            offset += 17
        else:
            raw_n, = struct.unpack_from('<Q',collection,offset)
            mode, stored_n = 0, raw_n
            offset += 8
        if mode not in range(5) or stored_n <= 0 or offset+stored_n > len(collection):
            raise ValueError('Invalid collection record')
        stored = collection[offset:offset+stored_n]
        offset += stored_n
        raw = decompress_exact(stored,raw_n) if mode else stored
        if len(raw) != raw_n:
            raise ValueError('Collection raw size mismatch')
        batches.append(raw)
    if offset != len(collection):
        raise ValueError('Collection trailing bytes')
    return batches

def decode_zsl1(batch: bytes):
    magic, w, c, t = struct.unpack_from('<4sIII',batch)
    if magic != b'ZSL1' or min(w,c,t) <= 0:
        raise ValueError('Invalid ZSL1 batch')
    offset, rows = 16, []
    for _ in range(w):
        n, = struct.unpack_from('<I',batch,offset)
        offset += 4
        if n <= 0 or offset+n > len(batch):
            raise ValueError('Invalid ZSL1 frame boundary')
        raw = decompress_exact(batch[offset:offset+n],c*t*4)
        offset += n
        rows.append(np.frombuffer(raw,dtype='<f4').copy().reshape(c,t))
    if offset != len(batch):
        raise ValueError('ZSL1 trailing bytes')
    return np.stack(rows)

def decode_lossless(data: bytes):
    meta, _, payload = parse(data)
    if not payload.startswith(b'ZLC1'):
        raise ValueError('Not the lossless payload type')
    values = np.concatenate([decode_zsl1(v) for v in unpack_collection(payload[4:])],axis=0)
    if list(values.shape) != meta['shape'] or not np.isfinite(values).all():
        raise ValueError('Decoded data contradict metadata')
    return values, meta

def as_original_jointzip(data: bytes) -> bytes:
    """Return an ordinary JZF1 file for the unchanged JointZip decoder."""
    _, header, payload = parse(data)
    if not payload.startswith(b'JZP1'):
        raise ValueError('Not the JointZip payload type')
    return HEADER.pack(b'JZF1',len(header))+header+payload

def decode(data: bytes, jointzip_decoder=None):
    """Decode without the source. JointZip uses its existing codec runtime."""
    meta, _, payload = parse(data)
    if payload.startswith(b'ZLC1'):
        return decode_lossless(data)
    if jointzip_decoder is None:
        from experiments.vldb.framework_ablation_20261017.codec import decode as jointzip_decoder
    values = jointzip_decoder(payload)
    if list(values.shape) != meta['shape']:
        raise ValueError('JointZip dimensions contradict metadata')
    return values,meta

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('file',type=Path)
    p.add_argument('--extract-jzf',type=Path)
    p.add_argument('--lossless-npy',type=Path)
    args=p.parse_args()
    data=args.file.read_bytes()
    if args.extract_jzf:
        args.extract_jzf.write_bytes(as_original_jointzip(data))
    elif args.lossless_npy:
        values,metadata=decode_lossless(data)
        np.save(args.lossless_npy,values)
        print(json.dumps(metadata))
    else:
        metadata,_,payload=parse(data)
        print(json.dumps(dict(metadata=metadata,payload_type=payload[:4].decode(),bytes=len(data))))
