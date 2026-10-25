"""Frozen official SZ3/ZFP implementations with self-contained QMC1 framing.

Each [window, channel] series is an independent native 1D float32 stream.
No cross-window information, padding, repair, or joint-constraint claim is used.
SZ3 uses a thin CLI over its allocating public API because its release CLI's
fixed allocation is below the API's required bound on small windows. ZFP uses
the untouched official CLI. Both retain their official compressed formats.
Run build_modern_codecs.sh in Linux/WSL first. Timings include process startup
and CLI file I/O; they are not in-process codec kernel throughput measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFIX = ROOT / "tmp" / "modern-codecs"
EXPECTED = {
    "sz3": (1, "3.3.2", "0ebe6fa26b0aa2e274d13b78df81a81e2db1b268"),
    "zfp": (2, "1.0.1", "f40868a6a1c190c802e7d8b5987064f044bf7812"),
}
# magic, format version, codec id, dtype (=1: little-endian f32), reserved,
# W, C, T, requested epsilon. Each row follows as uint64 length + native bytes.
HEADER = struct.Struct("<4sBBBBIIId")
LENGTH = struct.Struct("<Q")
MAX_VALUES = 1 << 28


@dataclass
class Encoded:
    payload: bytes
    reconstruction: np.ndarray
    metadata: dict[str, Any]


def _prefix() -> Path:
    return Path(os.environ.get("MODERN_CODECS_PREFIX", DEFAULT_PREFIX)).resolve()


@lru_cache(maxsize=16)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    # stat arguments invalidate the cache when a binary is replaced.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(codec: str) -> dict[str, Any]:
    if codec not in EXPECTED:
        raise ValueError(f"unsupported modern codec {codec!r}")
    if sys.platform != "linux" or sys.byteorder != "little":
        raise RuntimeError("Frozen modern CLI adapters require little-endian Linux/WSL")
    prefix = _prefix()
    manifest = json.loads((prefix / "manifest.json").read_text())
    value = dict(manifest["codecs"][codec])
    _, version, commit = EXPECTED[codec]
    if value["version"] != version or value["upstream_commit"] != commit:
        raise RuntimeError(f"{codec} manifest does not match the frozen official release")
    binary = prefix / "bin" / codec
    stat = binary.stat()
    digest = _sha256(str(binary), stat.st_size, stat.st_mtime_ns)
    if digest != value["binary_sha256"]:
        raise RuntimeError(f"{codec} binary SHA256 differs from the build manifest")
    for dependency in value.get("dependencies", []):
        dep_path = prefix / dependency["relative_path"]
        dep_stat = dep_path.stat()
        if _sha256(str(dep_path), dep_stat.st_size, dep_stat.st_mtime_ns) != dependency["binary_sha256"]:
            raise RuntimeError(f"{codec} dependency differs from the build manifest")
    value.update(binary=str(binary), threads=1,
                 build_configuration=manifest["configuration"],
                 compiler=manifest["compiler"], cmake=manifest["cmake"],
                 build_script_sha256=manifest["build_script_sha256"],
                 adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 process_environment={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                      "MKL_NUM_THREADS": "1", "LC_ALL": "C"},
                 native_shape="one 1D T-vector per window/channel",
                 contract="pointwise_absolute_only", dtype="<f4", frame="QMC1/v1")
    return value


def available_codecs() -> dict[str, dict[str, Any]]:
    """Return identities only for installed, hash-verified frozen binaries."""
    result = {}
    for codec in EXPECTED:
        try:
            result[codec] = _identity(codec)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    return result


def _run(command: list[str]) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LC_ALL="C")
    env.pop("LD_PRELOAD", None)
    env["LD_LIBRARY_PATH"] = str(_prefix() / "build/SZ3/tools/zstd")
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    measurement = {"command": command, "wall_seconds": time.perf_counter() - started,
                   "returncode": result.returncode, "stdout": result.stdout,
                   "stderr": result.stderr,
                   "timing_scope": "native CLI process including startup and file I/O"}
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}: "
                           f"{result.stderr[-3000:]} {result.stdout[-3000:]}")
    return measurement


def _parse(codec: str, payload: bytes) -> tuple[tuple[int, int, int], float, list[bytes]]:
    if codec not in EXPECTED or len(payload) < HEADER.size:
        raise ValueError("unknown codec or truncated QMC1 header")
    magic, version, codec_id, dtype, reserved, w, c, t, epsilon = HEADER.unpack_from(payload)
    if (magic, version, codec_id, dtype, reserved) != (b"QMC1", 1, EXPECTED[codec][0], 1, 0):
        raise ValueError("QMC1 codec/version/type mismatch")
    if not (w and c and t) or w * c * t > MAX_VALUES:
        raise ValueError("invalid or unsupported QMC1 dimensions")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("QMC1 epsilon must be finite and positive")
    if w * c > (len(payload) - HEADER.size) // (LENGTH.size + 1):
        raise ValueError("QMC1 row count exceeds available frame bytes")
    position = HEADER.size
    streams = []
    for _ in range(w * c):
        if position + LENGTH.size > len(payload):
            raise ValueError("truncated QMC1 row length")
        length, = LENGTH.unpack_from(payload, position)
        position += LENGTH.size
        if length == 0 or length > len(payload) - position:
            raise ValueError("truncated or empty QMC1 native stream")
        streams.append(payload[position:position + length])
        position += length
    if position != len(payload):
        raise ValueError("QMC1 trailing bytes")
    return (w, c, t), epsilon, streams


def _decode(codec: str, payload: bytes, identity: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    shape, _, streams = _parse(codec, payload)
    rows = []
    measurements = []
    with tempfile.TemporaryDirectory(prefix=f"vldb_{codec}_decode_") as directory:
        folder = Path(directory)
        for index, value in enumerate(streams):
            compressed = folder / f"{index}.native"
            decoded = folder / f"{index}.f32"
            compressed.write_bytes(value)
            if codec == "sz3":
                command = [identity["binary"], "d", str(compressed), str(decoded)]
            else:
                command = [identity["binary"], "-h", "-x", "serial", "-z", str(compressed),
                           "-o", str(decoded)]
            measurements.append(_run(command))
            if not decoded.is_file() or decoded.stat().st_size != shape[2] * 4:
                raise RuntimeError(f"{codec} did not decode the expected complete float32 row")
            row = np.fromfile(decoded, dtype="<f4")
            if not np.isfinite(row).all():
                raise RuntimeError(f"{codec} produced non-finite reconstruction")
            rows.append(row)
    return np.stack(rows).reshape(shape), measurements


def decode(codec: str, payload: bytes) -> np.ndarray:
    """Decode only persisted bytes; source data/encoder reconstruction are unused."""
    return _decode(codec, payload, _identity(codec))[0]


def encode(codec: str, source: np.ndarray, epsilon: float) -> Encoded:
    """Encode finite native float32 [W,C,T]; report strict point-error feasibility.

    A numerical error-bound violation is recorded rather than silently repaired,
    suppressed or declared a successful joint-contract output.
    """
    identity = _identity(codec)
    if not isinstance(source, np.ndarray) or source.dtype != np.dtype("float32"):
        raise TypeError("source must already be canonical float32; implicit precision conversion is forbidden")
    if source.ndim != 3 or any(d == 0 or d >= 2**32 for d in source.shape) or source.size > MAX_VALUES:
        raise ValueError("source must have supported nonempty [W,C,T] dimensions")
    if not np.isfinite(source).all():
        raise ValueError("source must contain finite float32 values")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    source = np.ascontiguousarray(source)
    parts = [HEADER.pack(b"QMC1", 1, EXPECTED[codec][0], 1, 0, *source.shape, epsilon)]
    calls = []
    native_sizes = []
    encode_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"vldb_{codec}_encode_") as directory:
        folder = Path(directory)
        for index, row in enumerate(source.reshape(-1, source.shape[-1])):
            raw = folder / f"{index}.f32"
            compressed = folder / f"{index}.native"
            row.tofile(raw)
            if codec == "sz3":
                command = [identity["binary"], "c", str(raw), str(compressed),
                           str(row.size), format(epsilon, ".17g")]
            else:
                command = [identity["binary"], "-h", "-x", "serial", "-f", "-1", str(row.size),
                           "-a", format(epsilon, ".17g"), "-i", str(raw), "-z", str(compressed)]
            calls.append(_run(command))
            value = compressed.read_bytes()
            if not value:
                raise RuntimeError(f"{codec} produced an empty compressed file")
            native_sizes.append(len(value))
            parts.extend((LENGTH.pack(len(value)), value))
    payload = b"".join(parts)
    encode_seconds = time.perf_counter() - encode_started
    decode_started = time.perf_counter()
    reconstruction, decode_calls = _decode(codec, payload, identity)
    decode_seconds = time.perf_counter() - decode_started
    max_error = float(np.max(np.abs(source.astype(np.float64) - reconstruction.astype(np.float64))))
    metadata = {
        "schema": "vldb.modern-codecs.v1", "identity": identity,
        "shape": list(source.shape), "epsilon": epsilon,
        "native_stream_sizes": native_sizes, "native_stream_bytes": sum(native_sizes),
        "framing_bytes": len(payload) - sum(native_sizes), "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "native_encode_calls": calls, "native_decode_calls": decode_calls,
        "encode_seconds": encode_seconds, "immediate_decode_seconds": decode_seconds,
        "encode_timing_scope": "QMC1 framing, input/output I/O and serial native CLI calls; excludes validation decode",
        "point_check": {"max_absolute_error": max_error, "epsilon": epsilon,
                        "tolerance": 0.0, "passed": max_error <= epsilon},
        "joint_contract_guaranteed": False,
    }
    return Encoded(payload, reconstruction, metadata)
