"""Strict SCS2 parsing and official native-codec replay helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from neural_compression_lab.classical import (
    QPETCorrelationCodec,
    SPERRCodec,
    SZ2016Codec,
    _run_native,
    _wsl_path,
)
from neural_compression_lab.models import build_model
from neural_compression_lab.runner import (
    NativeCollectionDiagnostics,
    _canonical_sha256,
    _decode_native_collection_stream,
    _decode_normalization_stream,
    _decode_provenance_stream,
    _decode_self_contained_native_stream,
    _encode_native_collection_stream,
    _normalization_stream,
    _provenance_stream,
    _self_contained_native_stream,
)


@dataclass(frozen=True)
class ParsedSCS2:
    codec_id: str
    channels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    provenance: dict[str, Any] | None
    collection_format: str
    payloads: tuple[bytes, ...]
    byte_size: int
    sha256: str


def parse_scs2(value: bytes | Path) -> ParsedSCS2:
    stream = value if isinstance(value, bytes) else value.read_bytes()
    codec_id, normalization, provenance_stream, collection = (
        _decode_self_contained_native_stream(stream)
    )
    channels, mean, std = _decode_normalization_stream(normalization)
    # Development SCS2 is allowed to omit the optional model/provenance field.
    # An actually present field is never treated as optional: it must be a
    # complete canonical PRV1 stream or the strict decoder raises ValueError.
    provenance = (
        _decode_provenance_stream(provenance_stream)
        if provenance_stream
        else None
    )
    payloads = tuple(_decode_native_collection_stream(collection))
    magic = collection[:4].decode("ascii")
    return ParsedSCS2(
        codec_id=codec_id,
        channels=tuple(channels),
        mean=mean,
        std=std,
        provenance=provenance,
        collection_format=magic,
        payloads=payloads,
        byte_size=len(stream),
        sha256=hashlib.sha256(stream).hexdigest(),
    )


def replay_qpet_scs2(
    artifact: Path,
    spec: dict[str, Any],
    channels: int,
    window: int,
) -> tuple[np.ndarray, ParsedSCS2]:
    """Replay QPC framing locally and its pinned official base decoders natively."""
    parsed = parse_scs2(artifact)
    if parsed.codec_id != "qpet_correlation":
        raise ValueError(f"not a QPET SCS2 artifact: {parsed.codec_id}")
    model = build_model(spec, channels=channels, window=window, groups={})
    if not isinstance(model, QPETCorrelationCodec):
        raise TypeError("configured model is not QPETCorrelationCodec")
    windows = [model.decompress_payload(payload) for payload in parsed.payloads]
    if not windows:
        raise ValueError("QPET artifact contains no QPC payload")
    reconstructed = np.concatenate(windows, axis=0)
    if reconstructed.dtype != np.float32 or not np.isfinite(reconstructed).all():
        raise RuntimeError("official QPET replay returned an invalid tensor")
    return reconstructed, parsed


def run_official_baseline(
    codec_name: str, source: np.ndarray, epsilon: float
) -> tuple[np.ndarray, bytes, str, dict[str, Any]]:
    """Run the pinned encoder and its immediate persisted-stream decode."""
    tensor = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
    if codec_name == "sperr":
        model = SPERRCodec()
    elif codec_name == "sz2016":
        model = SZ2016Codec()
    else:
        raise ValueError(f"unsupported official baseline: {codec_name}")
    decoded, payload = model.compress_reconstruct(tensor, float(epsilon))
    reconstruction = decoded.detach().cpu().numpy()
    if not payload or reconstruction.shape != source.shape:
        raise RuntimeError(f"{codec_name} returned an invalid result")
    return (
        reconstruction,
        payload,
        str(model.upstream_commit),
        dict(getattr(model, "last_metadata", {})),
    )


def replay_official_baseline_payload(
    codec_name: str,
    payload: bytes,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Decode a re-read payload via the pinned SPERR/SZ native decoder."""
    if codec_name == "sperr":
        model = SPERRCodec()
    elif codec_name == "sz2016":
        model = SZ2016Codec()
    else:
        raise ValueError(f"unsupported official baseline: {codec_name}")
    wsl = model._validate_backend()
    with tempfile.TemporaryDirectory(prefix=f"vldb_utility_{codec_name}_replay_") as name:
        directory = Path(name)
        stream_path = directory / "stream.bin"
        reconstruction_path = directory / "reconstruction.f32"
        stream_path.write_bytes(payload)
        if codec_name == "sperr":
            command = [
                wsl,
                "--exec",
                _wsl_path(model.binary),
                "--decode",
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
            ]
            timeout = 180
        else:
            assert isinstance(model, SZ2016Codec)
            command = [
                wsl,
                "--exec",
                _wsl_path(model.binary),
                "--decode",
                _wsl_path(model.config),
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
                *(str(value) for value in shape),
            ]
            timeout = 120
        process = _run_native(command, timeout=timeout)
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
            raise RuntimeError(f"persisted {codec_name} replay failed: {detail}")
        expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
        if (
            not reconstruction_path.is_file()
            or reconstruction_path.stat().st_size != expected_bytes
        ):
            raise RuntimeError(f"persisted {codec_name} replay returned wrong byte count")
        decoded = np.fromfile(reconstruction_path, dtype="<f4").reshape(shape)
    if not np.isfinite(decoded).all():
        raise RuntimeError(f"persisted {codec_name} replay returned non-finite values")
    return decoded


def make_baseline_scs2(
    codec_name: str,
    payload: bytes,
    bundle: Any,
    protocol_sha256: str,
    freeze_sha256: str,
    spec_index: int,
    point_index: int,
    epsilon: float,
) -> tuple[bytes, NativeCollectionDiagnostics]:
    """Wrap one official baseline stream using the experiment's SCS2 accounting."""
    spec = {"name": codec_name, "collection_frontier": "stream_closure_v1"}
    point = {"epsilon": float(epsilon)}
    identity = {
        "schema": "nclab.native-provenance.v1",
        "frozen_config_sha256": protocol_sha256,
        "freeze_manifest_sha256": freeze_sha256,
        "spec_index": int(spec_index),
        "point_index": int(point_index),
        "spec_sha256": _canonical_sha256(spec),
        "point_sha256": _canonical_sha256(point),
    }
    collection, diagnostics = _encode_native_collection_stream(
        [payload], "stream_closure_v1"
    )
    stream = _self_contained_native_stream(
        codec_name,
        _normalization_stream(bundle),
        _provenance_stream(identity),
        collection,
    )
    # Parse our own bytes before the caller is allowed to persist or charge them.
    parsed = parse_scs2(stream)
    if parsed.payloads != (payload,) or parsed.codec_id != codec_name:
        raise RuntimeError("self-produced baseline SCS2 failed exact replay")
    return stream, diagnostics
