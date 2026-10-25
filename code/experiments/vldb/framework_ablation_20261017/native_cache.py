"""Persistent cache of native, *unrepaired* candidates shared across ablation arms.

No statistical thresholds or repair output enter a native candidate. SQLite
commits each value atomically; short per-shard OS locks prevent duplicate codec
calls in concurrent workers. Locks are released by the OS after interruption.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np


def effective_cap(backend, cap):
    cap = float(cap)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('native cap must be finite and positive')
    if backend == 'zfp':
        return math.ldexp(1., math.floor(math.log2(cap)))
    if backend.startswith('serf_'):
        rounded = np.float32(cap)
        if float(rounded) > cap:
            rounded = np.nextafter(rounded, np.float32(0))
        return float(rounded)
    return cap


def backend_fingerprint(identity):
    # Host paths do not change the executable or configuration semantics. All
    # remaining declared identity fields, including SZ2016 config hash, do.
    return {k: v for k, v in identity.items() if k not in ('binary', 'configuration')}


class NativeCandidateCache:
    schema = 'jointzip.native-unrepaired-cache.v1'

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'locks').mkdir(exist_ok=True)
        self._db = None
        self._pid = None

    def _connection(self):
        if self._db is None or self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db = sqlite3.connect(self.root / 'native.sqlite3', timeout=120)
            self._pid = os.getpid()
            self._db.execute('PRAGMA busy_timeout=120000')
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            self._db.execute('CREATE TABLE IF NOT EXISTS candidates '
                             '(key TEXT PRIMARY KEY, binding TEXT NOT NULL, '
                             'payload BLOB NOT NULL, rebuilt BLOB NOT NULL, provenance TEXT NOT NULL)')
            self._db.commit()
        return self._db

    def binding(self, backend, row, cap):
        from . import codec
        row = np.asarray(row)
        if row.dtype != np.float32 or row.ndim != 1 or not np.isfinite(row).all():
            raise ValueError('native cache expects a finite float32 row')
        value = dict(schema=self.schema, backend=backend,
                     identity=backend_fingerprint(codec.identity(backend)),
                     dtype='<f4', length=int(row.size),
                     source_sha256=hashlib.sha256(row.astype('<f4', copy=False).tobytes()).hexdigest(),
                     effective_cap_hex=effective_cap(backend, cap).hex())
        binding = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(binding.encode()).hexdigest(), binding

    @contextmanager
    def _lock(self, key):
        # A bounded number of lock files; unrelated collisions only serialize
        # briefly and never cause scientific candidate reuse across bindings.
        with (self.root / 'locks' / (key[:3] + '.lock')).open('a+b') as stream:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b'0'); stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def get(self, backend, row, cap):
        key, binding = self.binding(backend, row, cap)
        saved = self._connection().execute(
            'SELECT binding,payload,rebuilt FROM candidates WHERE key=?', (key,)).fetchone()
        if saved is None:
            return None
        if saved[0] != binding or len(saved[2]) != row.size * 4 or not saved[1]:
            raise RuntimeError('native cache binding or record length differs')
        rebuilt = np.frombuffer(saved[2], dtype='<f4').copy()
        if not np.isfinite(rebuilt).all():
            raise RuntimeError('native cache has nonfinite unrepaired values')
        return bytes(saved[1]), rebuilt

    def put(self, backend, row, cap, payload, rebuilt, provenance=None):
        key, binding = self.binding(backend, row, cap)
        rebuilt = np.asarray(rebuilt)
        if rebuilt.dtype != np.float32 or rebuilt.shape != row.shape or not np.isfinite(rebuilt).all() or not payload:
            raise ValueError('invalid unrepaired native cache value')
        native_bytes = bytes(payload)
        restored_bytes = rebuilt.astype('<f4', copy=False).tobytes()
        db = self._connection()
        with db:
            db.execute('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?)',
                       (key, binding, native_bytes, restored_bytes,
                        json.dumps(provenance or {}, sort_keys=True, allow_nan=False)))
        saved = db.execute('SELECT binding,payload,rebuilt FROM candidates WHERE key=?', (key,)).fetchone()
        if saved != (binding, native_bytes, restored_bytes):
            raise RuntimeError('same native candidate binding produced different bytes')

    def get_or_compute(self, backend, row, cap, compute, provenance=None):
        """Return (native bytes, unrepaired values, cache_hit)."""
        key, _ = self.binding(backend, row, cap)
        with self._lock(key):
            saved = self.get(backend, row, cap)
            if saved is not None:
                return *saved, True
            payload, rebuilt = compute()
            self.put(backend, row, cap, payload, rebuilt, provenance)
            return payload, rebuilt.copy(), False

    def import_native_payload(self, backend, row, cap, payload, workdir, provenance=None):
        """Seed an old native member by decoding its BASE, never its repair."""
        from . import codec
        folder = Path(workdir); folder.mkdir(parents=True, exist_ok=True)
        key, _ = self.binding(backend, row, cap)
        with self._lock(key):
            saved = self.get(backend, row, cap)
            if saved is not None:
                if saved[0] != payload:
                    raise RuntimeError('historical native payload conflicts with candidate cache')
                return False
            rebuilt = codec._native_decode(backend, payload, len(row), folder)
            self.put(backend, row, cap, payload, rebuilt, provenance)
            return True

    def import_stream(self, backend, source, metadata, payload, workdir, provenance=None):
        """Import selected native BASE members from a verified old JZP1 stream.

        Caller supplies the exact old source and strips the JZF1 outer frame.
        Decoding does not apply correction. Lossless selected records are skipped.
        """
        from . import codec
        if backend_fingerprint(metadata['identity']) != backend_fingerprint(codec.identity(backend)):
            raise RuntimeError('historical backend identity differs from current native cache')
        header, records = codec.unframe(payload)
        if header['codec'] != backend or header['shape'] != tuple(source.shape):
            raise RuntimeError('historical native member source shape/backend differs')
        count = 0
        for i, (row, record) in enumerate(zip(source[:, 0, :], records)):
            if record['kind'] == 1:
                count += self.import_native_payload(backend, row, record['cap'], record['base'], workdir,
                            dict(provenance or {}, window=i, role='unrepaired_base_only'))
        return count

    def close(self):
        if self._db is not None:
            self._db.close(); self._db = None; self._pid = None
