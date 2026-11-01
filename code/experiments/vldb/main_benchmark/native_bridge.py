"""Linux transport for existing, unchanged project codecs and complete SCS2.

The historical adapters build ``wsl --exec`` commands. This module replaces only
their process transport and Windows path conversion within this evaluator. It
does not change the encoder/decoder algorithms or any persisted format.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from neural_compression_lab import classical as codecs
from neural_compression_lab.metrics import acf_preservation_metrics, reconstruction_metrics
from neural_compression_lab.runner import (
    _canonical_sha256, _encode_native_collection_stream, _normalization_stream,
    _provenance_stream, _self_contained_native_stream,
)
from experiments.vldb.utility import artifacts

ROOT = Path(__file__).resolve().parents[3]
TRANSPORT = Path(__file__).with_name("linux_exec.sh")
_NATIVE = threading.local()
QPET_SETTINGS = dict(
    max_lag=16, qoz_level=1, correction_mode="posthoc_greedy",
    estimator_scales=[1.75, 2.0, 2.25, 2.5], qpet_confidence=0.999,
    qpet_k=2.0, group_size=1, repair_denominators=[7, 15, 31, 63, 127],
    candidate_profile="generic", repair_optimizer="ssbb",
    correction_format_ceiling=10, base_frontier="incumbent_only",
    relaxed_bound_ratio_ceiling="none", surrogate_frontier="none",
    cfsr_bound_ratio_ceiling="none",
)


def _run_linux(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    # No retry on codec/data failures; a Linux process needs no WSL startup retry.
    with tempfile.TemporaryDirectory(prefix="vldb_native_time_") as folder:
        report = Path(folder) / "time.json"
        wrapped = ["/usr/bin/time", "-f", '{"elapsed_seconds":%e,"peak_rss_kib":%M}',
                   "-o", str(report), *command]
        started = time.perf_counter()
        with subprocess.Popen(wrapped, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as child:
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                raise
            process = subprocess.CompletedProcess(wrapped, child.returncode, stdout, stderr)
        measurement: dict[str, Any] = {
            "wall_seconds": time.perf_counter() - started,
            "returncode": process.returncode,
        }
        if report.exists():
            try:
                measurement.update(json.loads(report.read_text().splitlines()[-1]))
            except (ValueError, IndexError):
                measurement["time_report"] = report.read_text()
        if hasattr(_NATIVE, "measurements"):
            _NATIVE.measurements.append(measurement)
        return process


def install_linux_transport() -> None:
    if not Path("/proc/version").exists():
        raise RuntimeError("This evaluation runtime must run in Linux/WSL")
    TRANSPORT.chmod(0o755)
    codecs._wsl_path = lambda path: str(Path(path).resolve())
    codecs._run_native = _run_linux
    artifacts._wsl_path = codecs._wsl_path
    artifacts._run_native = _run_linux

    def validate(instance: Any) -> str:
        if hasattr(instance, "binary") and not instance.binary.is_file():
            raise FileNotFoundError(instance.binary)
        if hasattr(instance, "adapter") and not instance.adapter.is_file():
            raise FileNotFoundError(instance.adapter)
        return str(TRANSPORT)

    for cls in (codecs.QPETCorrelationCodec, codecs.SPERRCodec, codecs.SZ2016Codec,
                codecs.CAMEOCodec, codecs.CAMEOPointGuardCodec, codecs.CAMEOJointGuardCodec):
        cls._validate_backend = validate


@dataclass
class Encoded:
    codec_id: str
    payload: bytes
    reconstruction: np.ndarray
    metadata: dict[str, Any]
    encode_seconds: float
    native_measurements: list[dict[str, Any]]
    point: dict[str, float]


def effective_point(method: str, epsilon: float, tau: float, rho: float) -> dict[str, float]:
    return {"epsilon": float(epsilon),
            "acf_epsilon": 2.0 if method in ("qpet_point", "qpet_pearson") else float(tau),
            "min_correlation": -1.0 if method in ("qpet_point", "qpet_acf") else float(rho)}


def encode(method: str, source: np.ndarray, epsilon: float, tau: float, rho: float) -> Encoded:
    tensor = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
    point = effective_point(method, epsilon, tau, rho)
    _NATIVE.measurements = []
    started = time.perf_counter()
    if method.startswith("qpet_"):
        settings = dict(QPET_SETTINGS)
        if method == "qpet_cf9":
            settings["correction_format_ceiling"] = 9
        elif method == "qpet_legacy":
            settings.update(repair_optimizer="legacy", correction_format_ceiling=6)
        model = codecs.QPETCorrelationCodec(**settings)
        decoded, payload = model.compress_reconstruct_point(tensor, point)
        codec_id = "qpet_correlation"
    elif method in ("sperr", "sz2016"):
        model = codecs.SPERRCodec() if method == "sperr" else codecs.SZ2016Codec()
        decoded, payload = model.compress_reconstruct(tensor, epsilon)
        codec_id = method
    elif method == "zstd":
        model = codecs.ZstdLosslessCodec(level=19)
        decoded, payload = model.compress_reconstruct(tensor, None)
        codec_id = "zstd_lossless"
    elif method in ("cameo", "cameo_joint"):
        model = (codecs.CAMEOCodec(max_lag=16) if method == "cameo"
                 else codecs.CAMEOJointGuardCodec(max_lag=16))
        decoded, payload = (model.compress_reconstruct(tensor, tau) if method == "cameo"
                            else model.compress_reconstruct_point(tensor, point))
        codec_id = "cameo" if method == "cameo" else "cameo_jointguard"
    elif method in ("sz3", "zfp"):
        from . import modern_codecs
        result = modern_codecs.encode(method, source, epsilon)
        return Encoded(method, result.payload, result.reconstruction, result.metadata,
                       time.perf_counter() - started, list(_NATIVE.measurements), point)
    else:
        raise ValueError(f"unknown method {method}")
    return Encoded(codec_id, payload, decoded.detach().cpu().numpy(),
                   dict(getattr(model, "last_metadata", {})),
                   time.perf_counter() - started, list(_NATIVE.measurements), point)


def decode_payload(codec_id: str, payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if codec_id == "qpet_correlation":
        return codecs.QPETCorrelationCodec(**QPET_SETTINGS).decompress_payload(payload)
    if codec_id in ("sperr", "sz2016"):
        return artifacts.replay_official_baseline_payload(codec_id, payload, shape)
    if codec_id == "zstd_lossless":
        return codecs._decode_zstd_lossless_payload(payload)
    if codec_id == "cameo_jointguard":
        return codecs.CAMEOJointGuardCodec(max_lag=16).decompress_payload(payload)
    if codec_id == "cameo":
        from tools.cameo.cameo_batch import HEADER, _decode_series
        if len(payload) < HEADER.size:
            raise ValueError("truncated CAM1")
        magic, batch, channels, length, _, _, _ = HEADER.unpack_from(payload)
        if magic != b"CAM1" or (batch, channels, length) != tuple(shape):
            raise ValueError("CAM1 shape/header mismatch")
        position = HEADER.size
        rows = []
        for _ in range(batch * channels):
            series, position = _decode_series(payload, position, length)
            rows.append(series)
        if position != len(payload):
            raise ValueError("CAM1 trailing bytes")
        return np.stack(rows).reshape(shape)
    if codec_id in ("sz3", "zfp"):
        from . import modern_codecs
        return modern_codecs.decode(codec_id, payload)
    raise ValueError(codec_id)


def wrap(encoded: Encoded, *, channel: str, mean: float, std: float,
         protocol_sha: str, freeze_sha: str, method_index: int, point_index: int) -> bytes:
    bundle = SimpleNamespace(channels=[channel], standardizer=SimpleNamespace(
        mean=np.asarray([mean], dtype=np.float32), std=np.asarray([std], dtype=np.float32)))
    identity = {"schema": "nclab.native-provenance.v1",
                "frozen_config_sha256": protocol_sha, "freeze_manifest_sha256": freeze_sha,
                "spec_index": method_index, "point_index": point_index,
                "spec_sha256": _canonical_sha256({"codec": encoded.codec_id, "arm_index": method_index,
                                                  "protocol_sha256": protocol_sha}),
                "point_sha256": _canonical_sha256(encoded.point)}
    collection, _ = _encode_native_collection_stream([encoded.payload], "stream_closure_v1")
    value = _self_contained_native_stream(encoded.codec_id, _normalization_stream(bundle),
                                         _provenance_stream(identity), collection)
    parsed = artifacts.parse_scs2(value)
    if parsed.payloads != (encoded.payload,):
        raise RuntimeError("SCS2 did not reproduce original native payload")
    return value


def persist_replay(path: Path, encoded: Encoded, source: np.ndarray,
                   **wrap_kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    closure_started = time.perf_counter()
    value = wrap(encoded, **wrap_kwargs)
    closure_seconds = time.perf_counter() - closure_started
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
    times = []
    decoded = None
    for _ in range(3):
        started = time.perf_counter()
        parsed = artifacts.parse_scs2(path)
        decoded = decode_payload(encoded.codec_id, parsed.payloads[0], source.shape)
        times.append(time.perf_counter() - started)
        if decoded.shape != source.shape or not np.array_equal(
                decoded.view(np.uint32), encoded.reconstruction.view(np.uint32)):
            raise RuntimeError("persisted decode differs from immediate float32 reconstruction")
    assert decoded is not None
    metrics = reconstruction_metrics(source.astype(np.float64), decoded.astype(np.float64))
    metrics.update(acf_preservation_metrics(source.astype(np.float64),
                                           decoded.astype(np.float64), max_lag=16))
    return decoded, {"artifact_bytes": len(value), "native_payload_bytes": len(encoded.payload),
                     "raw_bytes": int(source.nbytes), "complete_cr": source.nbytes / len(value),
                     "artifact_sha256": hashlib.sha256(value).hexdigest(),
                     "encode_seconds": encoded.encode_seconds,
                     "outer_closure_seconds": closure_seconds,
                     "encode_and_outer_closure_seconds": encoded.encode_seconds + closure_seconds,
                     "encode_timing_scope": "adapter encode plus immediate reconstruction; excludes SCS2 wrapping",
                     "native_encode_only_seconds": encoded.metadata.get("encode_seconds"),
                     "decode_timing_scope": "read SCS2, unpack closure, native decode; warm OS caches",
                     "decode_seconds_median": statistics.median(times),
                     "decode_seconds_repetitions": times,
                     "native_calls": encoded.native_measurements, "metrics": metrics,
                     "three_replays_bit_identical": True}
