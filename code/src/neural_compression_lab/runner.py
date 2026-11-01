from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import yaml
import zstandard

from .bitstream import bsc_compressed_size, compressed_size, packed_binary, packed_unsigned, signed_varints
from .classical import NativeClassicalCodec
from .data import DataBundle, build_data
from .losses import compression_loss
from .metrics import acf_preservation_metrics, binary_metrics, reconstruction_metrics
from .models import (
    DeepDictCodec,
    EdgeCodec,
    LFZipCodec,
    UniEdgeCodec,
    TaskProbe,
    build_model,
    parameter_count,
)


CJRF_METADATA_FIELDS = (
    "cjrf_relaxed_base_attempts",
    "cjrf_relaxed_base_failures",
    "cjrf_relaxed_base_pruned",
    "cjrf_relaxed_base_duplicates",
    "cjrf_relaxed_base_records",
    "cjrf_relaxed_base_outside_pointwise",
    "cjrf_relaxed_legacy_repaired",
    "cjrf_relaxed_legacy_repair_candidates",
    "cjrf_selected_groups",
    "cjrf_saved_bytes",
    "cjrf_legacy_winner_groups",
    "cjrf_fvro_winner_groups",
    "cjrf_zrcar_winner_groups",
    "cjrf_ssbb_winner_groups",
)

CFSR_METADATA_FIELDS = (
    "cfsr_surrogate_sources",
    "cfsr_surrogate_base_attempts",
    "cfsr_surrogate_base_failures",
    "cfsr_surrogate_base_pruned",
    "cfsr_surrogate_base_duplicates",
    "cfsr_surrogate_base_records",
    "cfsr_surrogate_base_outside_original_pointwise",
    "cfsr_surrogate_legacy_repaired",
    "cfsr_surrogate_legacy_repair_candidates",
    "cfsr_selected_groups",
    "cfsr_saved_bytes",
    "cfsr_legacy_winner_groups",
    "cfsr_fvro_winner_groups",
    "cfsr_zrcar_winner_groups",
    "cfsr_ssbb_winner_groups",
    "cfsr_mode1_winner_groups",
    "cfsr_mode3_winner_groups",
    "cfsr_mode4_winner_groups",
    "cfsr_mode5_winner_groups",
    "cfsr_mode6_winner_groups",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum().clamp_min(1)
    negative = (1 - labels).sum().clamp_min(1)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=negative / positive)


@torch.no_grad()
def probe_metrics(probe: TaskProbe, loader, device: torch.device, transform=None) -> dict[str, float]:
    probe.eval()
    labels_all, probabilities_all = [], []
    for x, labels in loader:
        x = x.to(device)
        if transform is not None:
            x = transform(x)
        probabilities = torch.sigmoid(probe(x)).cpu().numpy()
        labels_all.append(labels.numpy())
        probabilities_all.append(probabilities)
    return binary_metrics(np.concatenate(labels_all), np.concatenate(probabilities_all))


def train_probe(
    channels: int,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> TaskProbe:
    probe = TaskProbe(channels).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(probe.state_dict())
    best_loss = float("inf")
    for epoch in range(epochs):
        probe.train()
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            loss = _weighted_bce(probe(x), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        losses = []
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                losses.append(float(_weighted_bce(probe(x), labels)))
        value = float(np.mean(losses))
        print(f"  task probe epoch {epoch + 1:02d}: val_bce={value:.5f}")
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    return probe


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    name: str,
    spec: dict,
    loader,
    device: torch.device,
    bypass_quantizer: bool = False,
) -> float:
    model.eval()
    losses = []
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        output = model(x, bypass_quantizer=bypass_quantizer)
        loss, _ = compression_loss(name, output, x, labels, spec)
        losses.append(float(loss))
    return float(np.mean(losses))


def train_codec(
    model: nn.Module,
    name: str,
    spec: dict,
    train_loader,
    val_loader,
    training: dict,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        return model, [], 0.0
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(spec.get("learning_rate", training["learning_rate"])),
        betas=tuple(float(value) for value in spec.get("optimizer_betas", [0.9, 0.999])),
        weight_decay=float(spec.get("weight_decay", 0.0)),
    )
    scheduler = None
    if spec.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["epochs"]),
            eta_min=float(spec.get("eta_min", 1e-6)),
        )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        model.train()
        train_losses = []
        bypass_quantizer = isinstance(model, (EdgeCodec, UniEdgeCodec)) and epoch < int(spec.get("warmup_epochs", 0))
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            output = model(x, bypass_quantizer=bypass_quantizer)
            loss, _ = compression_loss(name, output, x, labels, spec)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        val = validation_loss(model, name, spec, val_loader, device, bypass_quantizer=bypass_quantizer)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": val}
        history.append(row)
        phase = " warmup" if bypass_quantizer else ""
        print(f"  {name} epoch {epoch + 1:02d}{phase}: train={row['train_loss']:.5f} val={val:.5f}")
        # A bypassed autoencoder and its quantized codec optimize different paths.
        # Never select an uninitialized warm-up checkpoint as the best RVQ model.
        if bypass_quantizer:
            continue
        if val < best_val - 1e-6:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(training.get("early_stopping_patience", 7)):
                print(f"  {name}: early stopping")
                break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    return model, history, elapsed


def operating_points(model: nn.Module, spec: dict) -> list[dict[str, Any]]:
    if isinstance(model, (EdgeCodec, UniEdgeCodec)):
        return [{"label": f"q={q}", "n_quantizers": q} for q in range(1, model.num_quantizers + 1)]
    if isinstance(model, NativeClassicalCodec) and not model.requires_epsilon:
        return model.operating_points(spec)
    if isinstance(model, NativeClassicalCodec):
        return model.operating_points(spec)
    if isinstance(model, (DeepDictCodec, LFZipCodec)):
        return [{"label": f"epsilon={epsilon:g}", "epsilon": float(epsilon)} for epsilon in spec["residual_epsilons"]]
    return [{"label": "continuous-fp32"}]


def _inverse(bundle: DataBundle, data: np.ndarray) -> np.ndarray:
    time_major = np.transpose(data, (0, 2, 1))
    return np.transpose(bundle.standardizer.inverse(time_major), (0, 2, 1))


def _standardized_joint_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_lag: int,
) -> dict[str, float]:
    """Independently evaluate the joint constraints in standardized space.

    Inputs retain the batch/channel/time layout used by the native codec.  The
    float64 conversion is deliberate: the QPET adapter and native verifier also
    accumulate Pearson and ACF statistics in double precision.
    """
    source = np.asarray(original, dtype=np.float64)
    decoded = np.asarray(reconstructed, dtype=np.float64)
    if source.shape != decoded.shape or source.ndim != 3 or source.size == 0:
        raise ValueError("standardized joint metrics require equal nonempty BCT tensors")
    if not np.isfinite(source).all() or not np.isfinite(decoded).all():
        raise ValueError("standardized joint metrics require finite tensors")

    point_error = float(np.max(np.abs(source - decoded)))
    acf = acf_preservation_metrics(source, decoded, max_lag)
    correlation = reconstruction_metrics(source, decoded)["correlation_series_min"]
    metrics = {
        "persisted_standardized_max_abs_error": point_error,
        "persisted_standardized_acf_max_abs": float(acf["acf_max_abs"]),
        "persisted_standardized_correlation_series_min": float(correlation),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("persisted standardized constraint metric is non-finite")
    if metrics["persisted_standardized_max_abs_error"] < 0.0:
        raise RuntimeError("persisted standardized point error is negative")
    if metrics["persisted_standardized_acf_max_abs"] < 0.0:
        raise RuntimeError("persisted standardized ACF error is negative")
    if not -1.0 <= metrics["persisted_standardized_correlation_series_min"] <= 1.0:
        raise RuntimeError("persisted standardized Pearson metric is outside [-1,1]")
    return metrics


def _enforce_persisted_joint_constraints(
    metrics: dict[str, float], point: dict[str, Any]
) -> None:
    """Fail closed when the persisted QPET reconstruction violates its point."""
    epsilon = float(point["epsilon"])
    acf_epsilon = float(point.get("acf_epsilon", -1.0))
    minimum_correlation = float(point.get("min_correlation", -1.0))
    if metrics["persisted_standardized_max_abs_error"] > epsilon + 1e-7:
        raise RuntimeError(
            "persisted QPET reconstruction violated the standardized point-wise "
            f"bound: {metrics['persisted_standardized_max_abs_error']} > {epsilon}"
        )
    if (
        acf_epsilon >= 0.0
        and metrics["persisted_standardized_acf_max_abs"] > acf_epsilon + 1e-7
    ):
        raise RuntimeError(
            "persisted QPET reconstruction violated the standardized worst-lag "
            f"ACF bound: {metrics['persisted_standardized_acf_max_abs']} > "
            f"{acf_epsilon}"
        )
    if (
        minimum_correlation >= 0.0
        and metrics["persisted_standardized_correlation_series_min"]
        < minimum_correlation - 1e-7
    ):
        raise RuntimeError(
            "persisted QPET reconstruction violated the standardized per-series "
            "Pearson lower bound: "
            f"{metrics['persisted_standardized_correlation_series_min']} < "
            f"{minimum_correlation}"
        )


def _model_bytes(model: nn.Module) -> int:
    if isinstance(model, NativeClassicalCodec):
        return 0
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def _symbols_payload(output: dict, model: nn.Module) -> tuple[bytes, int]:
    symbols = output["symbols"]
    array = symbols.detach().cpu().numpy()
    if isinstance(model, (EdgeCodec, UniEdgeCodec)):
        return packed_unsigned(array, model.codebook_size), array.size * math.ceil(math.log2(model.codebook_size))
    if isinstance(model, DeepDictCodec):
        return packed_binary(array), array.size
    # TCN-RNN uses the continuous float32 latent assumed by the paper.
    array = array.astype(np.float32, copy=False)
    return array.tobytes(), array.size * 32


_NATIVE_COLLECTION_FRONTIERS = {"none", "stream_closure_v1"}
_STREAM_CLOSURE_ZSTD_LEVELS = (3, 9, 15, 19)
_NCS1_HEADER = struct.Struct("<4sI")
_NCS1_RECORD = struct.Struct("<Q")
_NCS2_HEADER = struct.Struct("<4sI")
_NCS2_RECORD = struct.Struct("<BQQ")
_MAX_NATIVE_PAYLOAD_BYTES = 256 * 1024 * 1024
_MAX_NATIVE_COLLECTION_RAW_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class NativeCollectionDiagnostics:
    frontier: str
    collection_format: str
    ncs1_bytes: int
    candidate_bytes: int
    saved_bytes: int
    compressed_payloads: int
    zstd_level_min: int
    zstd_level_max: int


def _validate_native_payloads(payloads: list[bytes]) -> None:
    if not payloads or len(payloads) > np.iinfo(np.uint32).max:
        raise ValueError("native collection requires 1..uint32 payloads")
    total = 0
    for payload in payloads:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("native collection payloads must be nonempty bytes")
        if len(payload) > _MAX_NATIVE_PAYLOAD_BYTES:
            raise ValueError("native collection payload exceeds the safety cap")
        total += len(payload)
        if total > _MAX_NATIVE_COLLECTION_RAW_BYTES:
            raise ValueError("native collection raw bytes exceed the safety cap")


def _ncs1_stream(payloads: list[bytes]) -> bytes:
    """Frame independently decodable native batch streams with exact lengths."""
    _validate_native_payloads(payloads)
    output = bytearray(_NCS1_HEADER.pack(b"NCS1", len(payloads)))
    for payload in payloads:
        output.extend(_NCS1_RECORD.pack(len(payload)))
        output.extend(payload)
    return bytes(output)


def _stream_closure_record(payload: bytes) -> tuple[int, bytes, int]:
    frames = tuple(
        zstandard.ZstdCompressor(
            level=level,
            threads=0,
            write_content_size=True,
        ).compress(payload)
        for level in _STREAM_CLOSURE_ZSTD_LEVELS
    )
    selected_index = min(
        range(len(frames)),
        key=lambda index: (len(frames[index]), index),
    )
    selected = frames[selected_index]
    if len(selected) >= len(payload):
        return 0, payload, 0
    return selected_index + 1, selected, _STREAM_CLOSURE_ZSTD_LEVELS[selected_index]


def _ncs2_stream(
    payloads: list[bytes],
) -> tuple[bytes, int, int, int]:
    _validate_native_payloads(payloads)
    output = bytearray(_NCS2_HEADER.pack(b"NCS2", len(payloads)))
    selected_levels: list[int] = []
    for payload in payloads:
        mode, stored, level = _stream_closure_record(payload)
        output.extend(_NCS2_RECORD.pack(mode, len(payload), len(stored)))
        output.extend(stored)
        if mode != 0:
            selected_levels.append(level)
    return (
        bytes(output),
        len(selected_levels),
        min(selected_levels, default=0),
        max(selected_levels, default=0),
    )


def _encode_native_collection_stream(
    payloads: list[bytes],
    collection_frontier: str = "none",
) -> tuple[bytes, NativeCollectionDiagnostics]:
    if (
        type(collection_frontier) is not str
        or collection_frontier not in _NATIVE_COLLECTION_FRONTIERS
    ):
        raise ValueError(
            "collection_frontier must be none or stream_closure_v1"
        )
    ncs1 = _ncs1_stream(payloads)
    if collection_frontier == "none":
        return ncs1, NativeCollectionDiagnostics(
            frontier="none",
            collection_format="NCS1",
            ncs1_bytes=len(ncs1),
            candidate_bytes=len(ncs1),
            saved_bytes=0,
            compressed_payloads=0,
            zstd_level_min=0,
            zstd_level_max=0,
        )
    ncs2, compressed_payloads, level_min, level_max = _ncs2_stream(payloads)
    try:
        decoded_candidate = _decode_native_collection_stream(ncs2)
    except ValueError as error:
        raise RuntimeError("self-produced NCS2 candidate is not decodable") from error
    if decoded_candidate != payloads:
        raise RuntimeError("self-produced NCS2 candidate changed a codec payload")
    use_ncs2 = len(ncs2) < len(ncs1)
    selected = ncs2 if use_ncs2 else ncs1
    return selected, NativeCollectionDiagnostics(
        frontier="stream_closure_v1",
        collection_format="NCS2" if use_ncs2 else "NCS1",
        ncs1_bytes=len(ncs1),
        candidate_bytes=len(ncs2),
        saved_bytes=len(ncs1) - len(selected),
        compressed_payloads=compressed_payloads,
        zstd_level_min=level_min,
        zstd_level_max=level_max,
    )


def _native_collection_stream(
    payloads: list[bytes],
    collection_frontier: str = "none",
) -> bytes:
    return _encode_native_collection_stream(payloads, collection_frontier)[0]


def _decode_native_collection_stream(stream: bytes) -> list[bytes]:
    if not isinstance(stream, bytes):
        raise ValueError("native collection stream must be bytes")
    if len(stream) < _NCS1_HEADER.size:
        raise ValueError("truncated native collection header")
    magic, count = _NCS1_HEADER.unpack_from(stream)
    if magic not in {b"NCS1", b"NCS2"} or count == 0:
        raise ValueError("invalid native collection magic")
    minimum_record_bytes = (
        _NCS1_RECORD.size + 1
        if magic == b"NCS1"
        else _NCS2_RECORD.size + 1
    )
    if count > (len(stream) - _NCS1_HEADER.size) // minimum_record_bytes:
        raise ValueError("native collection count exceeds remaining bytes")
    position = _NCS1_HEADER.size
    payloads: list[bytes] = []
    total_raw_bytes = 0
    for _ in range(count):
        if magic == b"NCS1":
            if position + _NCS1_RECORD.size > len(stream):
                raise ValueError("truncated native collection length")
            raw_size = _NCS1_RECORD.unpack_from(stream, position)[0]
            stored_size = raw_size
            mode = 0
            position += _NCS1_RECORD.size
        else:
            if position + _NCS2_RECORD.size > len(stream):
                raise ValueError("truncated NCS2 record")
            mode, raw_size, stored_size = _NCS2_RECORD.unpack_from(stream, position)
            position += _NCS2_RECORD.size
            if mode > len(_STREAM_CLOSURE_ZSTD_LEVELS):
                raise ValueError("invalid NCS2 payload mode")
        if (
            raw_size == 0
            or raw_size > _MAX_NATIVE_PAYLOAD_BYTES
            or stored_size == 0
            or stored_size > len(stream) - position
        ):
            raise ValueError("invalid native collection payload length")
        total_raw_bytes += raw_size
        if total_raw_bytes > _MAX_NATIVE_COLLECTION_RAW_BYTES:
            raise ValueError("native collection raw bytes exceed the safety cap")
        if mode == 0 and stored_size != raw_size:
            raise ValueError("raw NCS2 payload sizes are inconsistent")
        if mode != 0 and stored_size >= raw_size:
            raise ValueError("compressed NCS2 payload is not a strict byte win")
        stored = stream[position:position + stored_size]
        position += stored_size
        if mode == 0:
            payload = stored
        else:
            try:
                if zstandard.frame_content_size(stored) != raw_size:
                    raise ValueError("NCS2 frame content size is inconsistent")
                payload = zstandard.ZstdDecompressor().decompress(
                    stored,
                    max_output_size=raw_size,
                    allow_extra_data=False,
                )
            except zstandard.ZstdError as error:
                raise ValueError("invalid NCS2 Zstd frame") from error
            if len(payload) != raw_size:
                raise ValueError("NCS2 decoded payload size is inconsistent")
        payloads.append(payload)
    if position != len(stream):
        raise ValueError("native collection has trailing bytes")
    return payloads


def _normalization_stream(bundle: DataBundle) -> bytes:
    output = bytearray(struct.pack("<4sI", b"NOR2", len(bundle.channels)))
    for channel in bundle.channels:
        encoded = channel.encode("utf-8")
        if not encoded or len(encoded) > np.iinfo(np.uint16).max:
            raise ValueError("channel name cannot be represented in NOR2")
        output.extend(struct.pack("<H", len(encoded)))
        output.extend(encoded)
    output.extend(np.asarray(bundle.standardizer.mean, dtype="<f4").tobytes())
    output.extend(np.asarray(bundle.standardizer.std, dtype="<f4").tobytes())
    return bytes(output)


def _decode_normalization_stream(
    stream: bytes,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if len(stream) < 8:
        raise ValueError("truncated normalization stream")
    magic, channels = struct.unpack_from("<4sI", stream)
    if magic != b"NOR2" or channels == 0:
        raise ValueError("invalid normalization stream")
    position = 8
    names: list[str] = []
    for _ in range(channels):
        if position + 2 > len(stream):
            raise ValueError("truncated normalization channel name length")
        size = struct.unpack_from("<H", stream, position)[0]
        position += 2
        if size == 0 or size > len(stream) - position:
            raise ValueError("truncated normalization channel name")
        try:
            names.append(stream[position:position + size].decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("normalization channel name is not UTF-8") from error
        position += size
    expected_tail = 2 * channels * np.dtype("<f4").itemsize
    if len(stream) - position != expected_tail:
        raise ValueError("invalid normalization parameter length")
    values = np.frombuffer(stream, dtype="<f4", offset=position)
    mean = values[:channels].copy()
    std = values[channels:].copy()
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization stream contains invalid parameters")
    return names, mean, std


def _self_contained_native_stream(
    codec_id: str,
    normalization: bytes,
    model_artifact: bytes,
    codec_collection: bytes,
) -> bytes:
    encoded_codec_id = codec_id.encode("utf-8")
    if not encoded_codec_id:
        raise ValueError("self-contained native stream requires a codec identifier")
    header = struct.pack(
        "<4sIQQQQ",
        b"SCS2",
        2,
        len(encoded_codec_id),
        len(normalization),
        len(model_artifact),
        len(codec_collection),
    )
    return header + encoded_codec_id + normalization + model_artifact + codec_collection


def _decode_self_contained_native_stream(
    stream: bytes,
) -> tuple[str, bytes, bytes, bytes]:
    header = struct.Struct("<4sIQQQQ")
    if len(stream) < header.size:
        raise ValueError("truncated self-contained stream header")
    (
        magic, version, codec_id_size, normalization_size, model_size, collection_size,
    ) = header.unpack_from(stream)
    if magic != b"SCS2" or version != 2 or codec_id_size == 0:
        raise ValueError("invalid self-contained stream header")
    sizes = (codec_id_size, normalization_size, model_size, collection_size)
    if any(size > len(stream) for size in sizes) or sum(sizes) != len(stream) - header.size:
        raise ValueError("invalid self-contained stream sizes")
    position = header.size
    fields: list[bytes] = []
    for size in sizes:
        fields.append(stream[position:position + size])
        position += size
    try:
        codec_id = fields[0].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("self-contained codec identifier is not UTF-8") from error
    return codec_id, fields[1], fields[2], fields[3]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance_stream(identity: dict[str, Any]) -> bytes:
    required = {
        "schema",
        "frozen_config_sha256",
        "freeze_manifest_sha256",
        "spec_index",
        "point_index",
        "spec_sha256",
        "point_sha256",
    }
    if set(identity) != required or identity["schema"] != "nclab.native-provenance.v1":
        raise ValueError("invalid native provenance identity")
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<4sI", b"PRV1", len(payload)) + payload


def _decode_provenance_stream(stream: bytes) -> dict[str, Any]:
    if len(stream) < 8:
        raise ValueError("truncated native provenance stream")
    magic, payload_size = struct.unpack_from("<4sI", stream)
    if magic != b"PRV1" or payload_size != len(stream) - 8:
        raise ValueError("invalid native provenance stream")
    try:
        value = json.loads(stream[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native provenance is not canonical JSON") from error
    if not isinstance(value, dict) or _provenance_stream(value) != stream:
        raise ValueError("native provenance is not canonical")
    return value


def _load_freeze_verifier(workspace: Path):
    tool = workspace / "tools" / "qpet_correlation" / "freeze_manifest.py"
    spec = importlib.util.spec_from_file_location("nclab_freeze_manifest", tool)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load freeze verifier: {tool}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    verifier = getattr(module, "verify_manifest", None)
    if verifier is None:
        raise RuntimeError("freeze tool does not expose verify_manifest")
    return verifier


@torch.no_grad()
def evaluate_codec(
    model: nn.Module,
    spec: dict,
    point: dict[str, Any],
    loader,
    bundle: DataBundle,
    probe: TaskProbe,
    raw_probe_f1: float,
    device: torch.device,
    training_seconds: float,
    artifact_path: Path | None = None,
    provenance_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    probe.eval()
    collection_frontier = spec.get("collection_frontier", "none")
    if (
        type(collection_frontier) is not str
        or collection_frontier not in _NATIVE_COLLECTION_FRONTIERS
    ):
        raise ValueError(
            "collection_frontier must be none or stream_closure_v1"
        )
    if (
        collection_frontier != "none"
        and not isinstance(model, NativeClassicalCodec)
    ):
        raise ValueError(
            "collection_frontier is supported only for native classical codecs"
        )
    native_collection_diagnostics: NativeCollectionDiagnostics | None = None
    originals, reconstructions, labels_all, probabilities_all = [], [], [], []
    payloads: list[bytes] = []
    payload_bits = 0
    values = 0
    elapsed = 0.0
    used_symbols: set[tuple[int, int, int]] = set()
    symbol_capacity = 0
    residual_symbol_arrays: list[np.ndarray] = []
    native_standardized_originals: list[np.ndarray] = []
    native_standardized_reconstructions: list[np.ndarray] = []
    native_retained_points = 0
    native_logical_values = 0
    native_corrections = 0
    native_online_corrections = 0
    native_candidates_tested = 0
    native_candidate_failures = 0
    has_qpet_metadata = False
    has_native_constraint_metadata = False
    native_base_payload_bytes = 0
    native_correction_payload_bytes = 0
    native_variable_eb_groups = 0
    native_global_eb_groups = 0
    native_double_correction_groups = 0
    native_float_exact_correction_groups = 0
    native_sparse_exact_correction_groups = 0
    native_sparse_xor_correction_groups = 0
    native_quantized_correction_groups = 0
    native_packed_quantized_correction_groups = 0
    native_adaptive_packed_correction_groups = 0
    native_factorized_packed_correction_groups = 0
    native_subset_rank_packed_correction_groups = 0
    native_predictive_packed_correction_groups = 0
    native_constant_channel_correction_groups = 0
    native_gradient_balanced_groups = 0
    native_sperr_groups = 0
    native_sperr_selected_bound_min = float("inf")
    native_sperr_selected_bound_max = 0.0
    native_lossless_zstd_groups = 0
    native_serf_xor_groups = 0
    native_serf_qt_groups = 0
    native_serf_smooth_groups = 0
    native_semantic_independent_groups = 0
    native_semantic_shared_groups = 0
    native_semantic_strict_constant_groups = 0
    native_shared_base_search_groups = 0
    native_lfzip_order0_windows = 0
    native_lfzip_order32_windows = 0
    native_lfzip_qbyte1_windows = 0
    native_lfzip_qbyte2_windows = 0
    native_lfzip_base_scale_min = float("inf")
    native_lfzip_base_scale_max = 0.0
    native_standardized_max_abs_error = 0.0
    native_cameo_mean_acf_max = 0.0
    native_strict_acf_max_abs = 0.0
    native_correlation_series_min = 1.0
    cpp_standardized_max_abs_error = 0.0
    cpp_strict_acf_max_abs = 0.0
    cpp_correlation_series_min = 1.0
    native_stream_version: int | None = None
    native_stream_version_min: int | None = None
    native_candidate_profile: str | None = None
    native_repair_optimizer: str | None = None
    native_correction_format_ceiling: int | None = None
    native_base_frontier: str | None = None
    native_relaxed_bound_ratio_ceiling: float | str | None = None
    native_cjrf_ratio_mask: int | None = None
    native_cjrf = {field: 0 for field in CJRF_METADATA_FIELDS}
    native_surrogate_frontier: str | None = None
    native_cfsr_bound_ratio_ceiling: float | str | None = None
    native_cfsr_ratio_mask: int | None = None
    native_cfsr = {field: 0 for field in CFSR_METADATA_FIELDS}
    native_zrcar_base_records = 0
    native_zrcar_legacy_seeds = 0
    native_zrcar_fvro_seeds = 0
    native_zrcar_states_generated = 0
    native_zrcar_infeasible_states = 0
    native_zrcar_target_states_pruned = 0
    native_zrcar_infeasible_frontier_expansions = 0
    native_zrcar_infeasible_frontier_recoveries = 0
    native_zrcar_candidates = 0
    native_zrcar_prefix_candidates = 0
    native_zrcar_local_candidates = 0
    native_zrcar_selected_groups = 0
    native_zrcar_saved_bytes = 0
    native_zrcar_from_fvro_seed_groups = 0
    native_fvro_base_records = 0
    native_fvro_global_generated = 0
    native_fvro_global_infeasible = 0
    native_fvro_global_serialized = 0
    native_fvro_channel_generated = 0
    native_fvro_channel_infeasible = 0
    native_fvro_channel_serialized = 0
    native_fvro_local_generated = 0
    native_fvro_local_infeasible = 0
    native_fvro_selected_channels = 0
    native_fvro_strict_channel_steps = 0
    native_fvro_seeds_retained = 0
    native_fvro_reference_quantized_groups = 0
    native_fvro_final_quantized_groups = 0
    native_fvro_reference_format4_groups = 0
    native_fvro_reference_format6_groups = 0
    native_fvro_final_format4_groups = 0
    native_fvro_final_format6_groups = 0
    native_fvro_final_format7_groups = 0
    native_fvro_final_format8_groups = 0
    native_fvro_final_format9_groups = 0
    native_fvro_final_format10_groups = 0
    native_fvro_reference_exact_escapes = 0
    native_fvro_final_exact_escapes = 0
    native_fvro_reference_format4_exact_escapes = 0
    native_fvro_reference_format6_exact_escapes = 0
    native_fvro_final_format4_exact_escapes = 0
    native_fvro_final_format6_exact_escapes = 0
    native_fvro_final_format7_exact_escapes = 0
    native_fvro_final_format8_exact_escapes = 0
    native_fvro_final_format9_exact_escapes = 0
    native_fvro_final_format10_exact_escapes = 0
    native_fvro_ref6_to_final6_groups = 0
    native_fvro_ref6_to_non6_groups = 0
    native_fvro_ref6_to_final6_reference_exact_escapes = 0
    native_fvro_ref6_to_final6_final_exact_escapes = 0
    native_fvro_ref6_to_non6_reference_exact_escapes = 0
    native_fvro_selected_groups = 0
    native_fvro_saved_bytes = 0
    ssbb_metadata_fields = (
        "ssbb_base_records",
        "ssbb_channels_searched",
        "ssbb_positions_processed",
        "ssbb_actions_generated",
        "ssbb_states_generated",
        "ssbb_states_deduplicated",
        "ssbb_statistics_feasible_states",
        "ssbb_statistics_infeasible_states",
        "ssbb_feasible_states_retained",
        "ssbb_infeasible_states_retained",
        "ssbb_states_pruned_by_beam",
        "ssbb_terminal_states",
        "ssbb_terminal_direct_feasible",
        "ssbb_terminal_direct_infeasible",
        "ssbb_outputs_returned",
        "ssbb_candidates_serialized",
        "ssbb_seeds_retained",
        "ssbb_selected_channels",
        "ssbb_representation_candidates_serialized",
        "ssbb_representation_improved_starts",
        "ssbb_representation_gross_saved_bytes",
        "ssbb_composition_starts",
        "ssbb_composition_local_options",
        "ssbb_composition_transitions_generated",
        "ssbb_composition_transitions_deduplicated",
        "ssbb_composition_direct_feasible",
        "ssbb_composition_direct_infeasible",
        "ssbb_composition_candidates_serialized",
        "ssbb_composition_singletons_serialized",
        "ssbb_composition_pairs_serialized",
        "ssbb_composition_frontier_carries",
        "ssbb_composition_anchor_retention_events",
        "ssbb_composition_format_retention_events",
        "ssbb_composition_rate_retention_events",
        "ssbb_composition_states_pruned",
        "ssbb_composition_max_frontier",
        "ssbb_composition_selected_channels",
        "ssbb_composition_selected_groups",
        "ssbb_composition_saved_bytes",
        "ssbb_selected_groups",
        "ssbb_saved_bytes",
        "ssbb_incremental_saved_bytes",
        "ssbb_representation_selected_groups",
        "ssbb_selected_representation_saved_bytes",
        "ssbb_search_selected_groups",
        "ssbb_selected_search_saved_bytes",
    )
    native_ssbb = {field: 0 for field in ssbb_metadata_fields}
    native_second_stage_selected_groups = 0
    native_second_stage_saved_bytes = 0
    evaluated_standardized_max_abs_error = 0.0
    persisted_standardized_metrics: dict[str, float] | None = None

    for x, labels in loader:
        x = x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if isinstance(model, NativeClassicalCodec):
            reconstructed, payload = model.compress_reconstruct_point(x, point)
            bits = len(payload) * 8
            metadata = getattr(model, "last_metadata", {})
            if any(
                key in metadata
                for key in (
                    "standardized_max_abs_error",
                    "strict_acf_max_abs",
                    "correlation_series_min",
                )
            ):
                has_native_constraint_metadata = True
            if metadata.get("version") is not None:
                has_qpet_metadata = True
                batch_stream_version = int(metadata["version"])
                if native_stream_version is None:
                    native_stream_version = batch_stream_version
                    native_stream_version_min = batch_stream_version
                else:
                    native_stream_version = max(
                        native_stream_version, batch_stream_version
                    )
                    if native_stream_version_min is None:
                        raise RuntimeError("native stream-version range is incomplete")
                    native_stream_version_min = min(
                        native_stream_version_min, batch_stream_version
                    )
                batch_candidate_profile = metadata.get("candidate_profile")
                if not isinstance(batch_candidate_profile, str):
                    raise RuntimeError("QPET metadata omitted its candidate profile")
                if native_candidate_profile is None:
                    native_candidate_profile = batch_candidate_profile
                elif native_candidate_profile != batch_candidate_profile:
                    raise RuntimeError("QPET emitted mixed candidate profiles")
                batch_repair_optimizer = metadata.get("repair_optimizer")
                if not isinstance(batch_repair_optimizer, str):
                    raise RuntimeError("QPET metadata omitted its repair optimizer")
                if native_repair_optimizer is None:
                    native_repair_optimizer = batch_repair_optimizer
                elif native_repair_optimizer != batch_repair_optimizer:
                    raise RuntimeError("QPET emitted mixed repair optimizers")
                batch_format_ceiling = metadata.get("correction_format_ceiling")
                if type(batch_format_ceiling) is not int:
                    raise RuntimeError(
                        "QPET metadata omitted its correction-format ceiling"
                    )
                if native_correction_format_ceiling is None:
                    native_correction_format_ceiling = batch_format_ceiling
                elif native_correction_format_ceiling != batch_format_ceiling:
                    raise RuntimeError("QPET emitted mixed correction-format ceilings")
                batch_base_frontier = metadata.get("base_frontier")
                if not isinstance(batch_base_frontier, str):
                    raise RuntimeError("QPET metadata omitted its base frontier")
                if native_base_frontier is None:
                    native_base_frontier = batch_base_frontier
                elif native_base_frontier != batch_base_frontier:
                    raise RuntimeError("QPET emitted mixed base frontiers")
                batch_relaxed_ceiling = metadata.get(
                    "relaxed_bound_ratio_ceiling"
                )
                if not (
                    batch_relaxed_ceiling == "none"
                    or (
                        type(batch_relaxed_ceiling) is float
                        and batch_relaxed_ceiling in {1.25, 1.5, 2.0}
                    )
                ):
                    raise RuntimeError(
                        "QPET metadata omitted its relaxed-bound ratio ceiling"
                    )
                if native_relaxed_bound_ratio_ceiling is None:
                    native_relaxed_bound_ratio_ceiling = batch_relaxed_ceiling
                elif native_relaxed_bound_ratio_ceiling != batch_relaxed_ceiling:
                    raise RuntimeError(
                        "QPET emitted mixed relaxed-bound ratio ceilings"
                    )
                batch_cjrf_ratio_mask = metadata.get("cjrf_ratio_mask")
                if type(batch_cjrf_ratio_mask) is not int:
                    raise RuntimeError("QPET metadata omitted its CJRF ratio mask")
                if native_cjrf_ratio_mask is None:
                    native_cjrf_ratio_mask = batch_cjrf_ratio_mask
                elif native_cjrf_ratio_mask != batch_cjrf_ratio_mask:
                    raise RuntimeError("QPET emitted mixed CJRF ratio masks")
                batch_surrogate_frontier = metadata.get("surrogate_frontier")
                if not isinstance(batch_surrogate_frontier, str):
                    raise RuntimeError(
                        "QPET metadata omitted its surrogate frontier"
                    )
                if native_surrogate_frontier is None:
                    native_surrogate_frontier = batch_surrogate_frontier
                elif native_surrogate_frontier != batch_surrogate_frontier:
                    raise RuntimeError("QPET emitted mixed surrogate frontiers")
                batch_cfsr_ceiling = metadata.get(
                    "cfsr_bound_ratio_ceiling"
                )
                if not (
                    batch_cfsr_ceiling == "none"
                    or (
                        type(batch_cfsr_ceiling) is float
                        and batch_cfsr_ceiling in {0.25, 0.5, 1.0}
                    )
                ):
                    raise RuntimeError(
                        "QPET metadata omitted its CFSR bound-ratio ceiling"
                    )
                if native_cfsr_bound_ratio_ceiling is None:
                    native_cfsr_bound_ratio_ceiling = batch_cfsr_ceiling
                elif native_cfsr_bound_ratio_ceiling != batch_cfsr_ceiling:
                    raise RuntimeError(
                        "QPET emitted mixed CFSR bound-ratio ceilings"
                    )
                batch_cfsr_ratio_mask = metadata.get("cfsr_ratio_mask")
                if type(batch_cfsr_ratio_mask) is not int:
                    raise RuntimeError("QPET metadata omitted its CFSR ratio mask")
                if native_cfsr_ratio_mask is None:
                    native_cfsr_ratio_mask = batch_cfsr_ratio_mask
                elif native_cfsr_ratio_mask != batch_cfsr_ratio_mask:
                    raise RuntimeError("QPET emitted mixed CFSR ratio masks")
            native_retained_points += int(metadata.get("retained_points", 0))
            native_logical_values += int(metadata.get("logical_values", 0))
            native_corrections += int(metadata.get("corrections", 0))
            native_online_corrections += int(metadata.get("online_corrections", 0))
            native_candidates_tested += int(metadata.get("candidates_tested", 0))
            native_candidate_failures += int(metadata.get("candidate_failures", 0))
            for field in CJRF_METADATA_FIELDS:
                native_cjrf[field] += int(metadata.get(field, 0))
            for field in CFSR_METADATA_FIELDS:
                native_cfsr[field] += int(metadata.get(field, 0))
            native_base_payload_bytes += int(metadata.get("base_payload_bytes", 0))
            native_correction_payload_bytes += int(
                metadata.get("correction_payload_bytes", 0)
            )
            native_variable_eb_groups += int(metadata.get("variable_eb_groups", 0))
            native_global_eb_groups += int(metadata.get("global_eb_groups", 0))
            native_double_correction_groups += int(
                metadata.get("double_correction_groups", 0)
            )
            native_float_exact_correction_groups += int(
                metadata.get("float_exact_correction_groups", 0)
            )
            native_sparse_exact_correction_groups += int(
                metadata.get("sparse_exact_correction_groups", 0)
            )
            native_sparse_xor_correction_groups += int(
                metadata.get("sparse_xor_correction_groups", 0)
            )
            native_quantized_correction_groups += int(
                metadata.get("quantized_correction_groups", 0)
            )
            native_packed_quantized_correction_groups += int(
                metadata.get("packed_quantized_correction_groups", 0)
            )
            native_adaptive_packed_correction_groups += int(
                metadata.get("adaptive_packed_correction_groups", 0)
            )
            native_factorized_packed_correction_groups += int(
                metadata.get("factorized_packed_correction_groups", 0)
            )
            native_subset_rank_packed_correction_groups += int(
                metadata.get("subset_rank_packed_correction_groups", 0)
            )
            native_predictive_packed_correction_groups += int(
                metadata.get("predictive_packed_correction_groups", 0)
            )
            native_constant_channel_correction_groups += int(
                metadata.get("constant_channel_correction_groups", 0)
            )
            native_gradient_balanced_groups += int(
                metadata.get("gradient_balanced_groups", 0)
            )
            native_sperr_groups += int(metadata.get("sperr_groups", 0))
            native_lossless_zstd_groups += int(
                metadata.get("lossless_zstd_groups", 0)
            )
            native_serf_xor_groups += int(metadata.get("serf_xor_groups", 0))
            native_serf_qt_groups += int(metadata.get("serf_qt_groups", 0))
            native_serf_smooth_groups += int(metadata.get("serf_smooth_groups", 0))
            native_semantic_independent_groups += int(
                metadata.get("semantic_independent_groups", 0)
            )
            native_semantic_shared_groups += int(
                metadata.get("semantic_shared_groups", 0)
            )
            native_semantic_strict_constant_groups += int(
                metadata.get("semantic_strict_constant_groups", 0)
            )
            native_shared_base_search_groups += int(
                metadata.get("shared_base_search_groups", 0)
            )
            native_zrcar_base_records += int(metadata.get("zrcar_base_records", 0))
            native_zrcar_legacy_seeds += int(
                metadata.get("zrcar_legacy_seeds", 0)
            )
            native_zrcar_fvro_seeds += int(
                metadata.get("zrcar_fvro_seeds", 0)
            )
            native_zrcar_states_generated += int(
                metadata.get("zrcar_states_generated", 0)
            )
            native_zrcar_infeasible_states += int(
                metadata.get("zrcar_infeasible_states", 0)
            )
            native_zrcar_target_states_pruned += int(
                metadata.get("zrcar_target_states_pruned", 0)
            )
            native_zrcar_infeasible_frontier_expansions += int(
                metadata.get("zrcar_infeasible_frontier_expansions", 0)
            )
            native_zrcar_infeasible_frontier_recoveries += int(
                metadata.get("zrcar_infeasible_frontier_recoveries", 0)
            )
            native_zrcar_candidates += int(metadata.get("zrcar_candidates", 0))
            native_zrcar_prefix_candidates += int(
                metadata.get("zrcar_prefix_candidates", 0)
            )
            native_zrcar_local_candidates += int(
                metadata.get("zrcar_local_candidates", 0)
            )
            native_zrcar_selected_groups += int(
                metadata.get("zrcar_selected_groups", 0)
            )
            native_zrcar_saved_bytes += int(metadata.get("zrcar_saved_bytes", 0))
            native_zrcar_from_fvro_seed_groups += int(
                metadata.get("zrcar_from_fvro_seed_groups", 0)
            )
            native_fvro_base_records += int(metadata.get("fvro_base_records", 0))
            native_fvro_global_generated += int(
                metadata.get("fvro_global_generated", 0)
            )
            native_fvro_global_infeasible += int(
                metadata.get("fvro_global_infeasible", 0)
            )
            native_fvro_global_serialized += int(
                metadata.get("fvro_global_serialized", 0)
            )
            native_fvro_channel_generated += int(
                metadata.get("fvro_channel_generated", 0)
            )
            native_fvro_channel_infeasible += int(
                metadata.get("fvro_channel_infeasible", 0)
            )
            native_fvro_channel_serialized += int(
                metadata.get("fvro_channel_serialized", 0)
            )
            native_fvro_local_generated += int(
                metadata.get("fvro_local_generated", 0)
            )
            native_fvro_local_infeasible += int(
                metadata.get("fvro_local_infeasible", 0)
            )
            native_fvro_selected_channels += int(
                metadata.get("fvro_selected_channels", 0)
            )
            native_fvro_strict_channel_steps += int(
                metadata.get("fvro_strict_channel_steps", 0)
            )
            native_fvro_seeds_retained += int(
                metadata.get("fvro_seeds_retained", 0)
            )
            native_fvro_reference_quantized_groups += int(
                metadata.get("fvro_reference_quantized_groups", 0)
            )
            native_fvro_final_quantized_groups += int(
                metadata.get("fvro_final_quantized_groups", 0)
            )
            native_fvro_reference_format4_groups += int(
                metadata.get("fvro_reference_format4_groups", 0)
            )
            native_fvro_reference_format6_groups += int(
                metadata.get("fvro_reference_format6_groups", 0)
            )
            native_fvro_final_format4_groups += int(
                metadata.get("fvro_final_format4_groups", 0)
            )
            native_fvro_final_format6_groups += int(
                metadata.get("fvro_final_format6_groups", 0)
            )
            native_fvro_final_format7_groups += int(
                metadata.get("fvro_final_format7_groups", 0)
            )
            native_fvro_final_format8_groups += int(
                metadata.get("fvro_final_format8_groups", 0)
            )
            native_fvro_final_format9_groups += int(
                metadata.get("fvro_final_format9_groups", 0)
            )
            native_fvro_final_format10_groups += int(
                metadata.get("fvro_final_format10_groups", 0)
            )
            native_fvro_reference_exact_escapes += int(
                metadata.get("fvro_reference_exact_escapes", 0)
            )
            native_fvro_final_exact_escapes += int(
                metadata.get("fvro_final_exact_escapes", 0)
            )
            native_fvro_reference_format4_exact_escapes += int(
                metadata.get("fvro_reference_format4_exact_escapes", 0)
            )
            native_fvro_reference_format6_exact_escapes += int(
                metadata.get("fvro_reference_format6_exact_escapes", 0)
            )
            native_fvro_final_format4_exact_escapes += int(
                metadata.get("fvro_final_format4_exact_escapes", 0)
            )
            native_fvro_final_format6_exact_escapes += int(
                metadata.get("fvro_final_format6_exact_escapes", 0)
            )
            native_fvro_final_format7_exact_escapes += int(
                metadata.get("fvro_final_format7_exact_escapes", 0)
            )
            native_fvro_final_format8_exact_escapes += int(
                metadata.get("fvro_final_format8_exact_escapes", 0)
            )
            native_fvro_final_format9_exact_escapes += int(
                metadata.get("fvro_final_format9_exact_escapes", 0)
            )
            native_fvro_final_format10_exact_escapes += int(
                metadata.get("fvro_final_format10_exact_escapes", 0)
            )
            native_fvro_ref6_to_final6_groups += int(
                metadata.get("fvro_ref6_to_final6_groups", 0)
            )
            native_fvro_ref6_to_non6_groups += int(
                metadata.get("fvro_ref6_to_non6_groups", 0)
            )
            native_fvro_ref6_to_final6_reference_exact_escapes += int(
                metadata.get("fvro_ref6_to_final6_reference_exact_escapes", 0)
            )
            native_fvro_ref6_to_final6_final_exact_escapes += int(
                metadata.get("fvro_ref6_to_final6_final_exact_escapes", 0)
            )
            native_fvro_ref6_to_non6_reference_exact_escapes += int(
                metadata.get("fvro_ref6_to_non6_reference_exact_escapes", 0)
            )
            native_fvro_selected_groups += int(
                metadata.get("fvro_selected_groups", 0)
            )
            native_fvro_saved_bytes += int(metadata.get("fvro_saved_bytes", 0))
            for field in ssbb_metadata_fields:
                value = int(metadata.get(field, 0))
                if field == "ssbb_composition_max_frontier":
                    native_ssbb[field] = max(native_ssbb[field], value)
                else:
                    native_ssbb[field] += value
            native_second_stage_selected_groups += int(
                metadata.get("second_stage_selected_groups", 0)
            )
            native_second_stage_saved_bytes += int(
                metadata.get("second_stage_saved_bytes", 0)
            )
            native_lfzip_order0_windows += int(metadata.get("lfzip_order0_windows", 0))
            native_lfzip_order32_windows += int(metadata.get("lfzip_order32_windows", 0))
            native_lfzip_qbyte1_windows += int(metadata.get("lfzip_qbyte1_windows", 0))
            native_lfzip_qbyte2_windows += int(metadata.get("lfzip_qbyte2_windows", 0))
            if metadata.get("lfzip_base_scale_min") is not None:
                native_lfzip_base_scale_min = min(
                    native_lfzip_base_scale_min,
                    float(metadata["lfzip_base_scale_min"]),
                )
                native_lfzip_base_scale_max = max(
                    native_lfzip_base_scale_max,
                    float(metadata["lfzip_base_scale_max"]),
                )
            if metadata.get("sperr_selected_bound_min") is not None:
                native_sperr_selected_bound_min = min(
                    native_sperr_selected_bound_min,
                    float(metadata["sperr_selected_bound_min"]),
                )
                native_sperr_selected_bound_max = max(
                    native_sperr_selected_bound_max,
                    float(metadata["sperr_selected_bound_max"]),
                )
            native_standardized_max_abs_error = max(
                native_standardized_max_abs_error,
                float(metadata.get("standardized_max_abs_error", 0.0)),
            )
            native_cameo_mean_acf_max = max(
                native_cameo_mean_acf_max,
                float(metadata.get("cameo_mean_acf_max", 0.0)),
            )
            native_strict_acf_max_abs = max(
                native_strict_acf_max_abs,
                float(metadata.get("strict_acf_max_abs", 0.0)),
            )
            native_correlation_series_min = min(
                native_correlation_series_min,
                float(metadata.get("correlation_series_min", 1.0)),
            )
            cpp_standardized_max_abs_error = max(
                cpp_standardized_max_abs_error,
                float(metadata.get("cpp_standardized_max_abs_error", 0.0)),
            )
            cpp_strict_acf_max_abs = max(
                cpp_strict_acf_max_abs,
                float(metadata.get("cpp_strict_acf_max_abs", 0.0)),
            )
            cpp_correlation_series_min = min(
                cpp_correlation_series_min,
                float(metadata.get("cpp_correlation_series_min", 1.0)),
            )
            native_standardized_reconstructions.append(
                np.ascontiguousarray(
                    reconstructed.detach().cpu().numpy(), dtype="<f4"
                )
            )
            native_standardized_originals.append(
                np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4")
            )
        elif isinstance(model, LFZipCodec):
            reconstructed, residual_symbols = model.error_bounded_reconstruct(x, point["epsilon"])
            residual_array = residual_symbols.cpu().numpy()
            if residual_array.min() < np.iinfo(np.int16).min or residual_array.max() > np.iinfo(np.int16).max:
                raise OverflowError("LFZip residual symbol exceeds the paper's signed int16 representation")
            payload = residual_array.astype("<i2", copy=False).tobytes()
            residual_symbol_arrays.append(residual_array)
            bits = len(payload) * 8
        else:
            kwargs = {k: v for k, v in point.items() if k != "label"}
            output = model(x, **kwargs)
            reconstructed = output["reconstruction"]
            payload, bits = _symbols_payload(output, model)
            if isinstance(model, (EdgeCodec, UniEdgeCodec)):
                array = output["symbols"].detach().cpu().numpy()
                for level in range(array.shape[-1]):
                    used_symbols.update((0, level, int(value)) for value in np.unique(array[..., level]))
                symbol_capacity = array.shape[-1] * model.codebook_size
            if isinstance(model, DeepDictCodec):
                epsilon = point["epsilon"]
                residual_symbols = torch.round((x - reconstructed) / (2.0 * epsilon)).to(torch.int32)
                residual_symbol_arrays.append(residual_symbols.cpu().numpy())
                reconstructed = reconstructed + residual_symbols * (2.0 * epsilon)
                residual_payload = signed_varints(residual_symbols.cpu().numpy())
                payload = struct.pack("<I", len(payload)) + payload + residual_payload
                bits += len(residual_payload) * 8 + 32
        if not torch.isfinite(reconstructed).all():
            raise RuntimeError(f"{spec['name']} produced a non-finite reconstruction")
        evaluated_standardized_max_abs_error = max(
            evaluated_standardized_max_abs_error,
            float(torch.max(torch.abs(x - reconstructed)).item()),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - started

        probabilities = torch.sigmoid(probe(reconstructed)).cpu().numpy()
        originals.append(_inverse(bundle, x.cpu().numpy()))
        reconstructions.append(_inverse(bundle, reconstructed.cpu().numpy()))
        labels_all.append(labels.numpy())
        probabilities_all.append(probabilities)
        payloads.append(payload)
        payload_bits += bits
        values += x.numel()

    if has_qpet_metadata and native_repair_optimizer in {"ssbb_value", "ssbb"}:
        if (
            native_ssbb["ssbb_representation_candidates_serialized"]
            != native_ssbb["ssbb_composition_starts"]
            or native_ssbb["ssbb_representation_improved_starts"]
            > native_ssbb["ssbb_representation_candidates_serialized"]
            or (native_ssbb["ssbb_representation_improved_starts"] == 0)
            != (native_ssbb["ssbb_representation_gross_saved_bytes"] == 0)
            or native_ssbb["ssbb_representation_selected_groups"]
            > native_ssbb["ssbb_selected_groups"]
            or native_ssbb["ssbb_representation_selected_groups"]
            > native_ssbb["ssbb_representation_improved_starts"]
            or native_ssbb["ssbb_search_selected_groups"]
            > native_ssbb["ssbb_selected_groups"]
            or (native_ssbb["ssbb_representation_selected_groups"] == 0)
            != (native_ssbb["ssbb_selected_representation_saved_bytes"] == 0)
            or (native_ssbb["ssbb_search_selected_groups"] == 0)
            != (native_ssbb["ssbb_selected_search_saved_bytes"] == 0)
            or native_ssbb["ssbb_incremental_saved_bytes"]
            != native_ssbb["ssbb_selected_representation_saved_bytes"]
            + native_ssbb["ssbb_selected_search_saved_bytes"]
            or (native_ssbb["ssbb_selected_groups"] == 0)
            != (native_ssbb["ssbb_incremental_saved_bytes"] == 0)
            or native_ssbb["ssbb_saved_bytes"]
            < native_ssbb["ssbb_incremental_saved_bytes"]
            or native_ssbb["ssbb_selected_representation_saved_bytes"]
            > native_ssbb["ssbb_representation_gross_saved_bytes"]
        ):
            raise RuntimeError(
                "aggregated SSBB representation/search attribution diverged"
            )

    original = np.concatenate(originals)
    reconstructed = np.concatenate(reconstructions)
    reconstruction = reconstruction_metrics(original, reconstructed)
    acf_metrics = acf_preservation_metrics(
        original,
        reconstructed,
        int(spec.get("acf_metric_lags", spec.get("max_lag", 16))),
    )
    task = binary_metrics(np.concatenate(labels_all), np.concatenate(probabilities_all))
    if isinstance(model, NativeClassicalCodec):
        # Preserve boundaries between independently decodable batch streams in
        # an actual collection container, then count that exact byte string.
        native_stream, native_collection_diagnostics = (
            _encode_native_collection_stream(payloads, collection_frontier)
        )
        if _decode_native_collection_stream(native_stream) != payloads:
            raise RuntimeError("native collection round trip changed a codec payload")
        stream_bits = len(native_stream) * 8
        collection_framing_bytes = (
            _NCS2_HEADER.size + len(payloads) * _NCS2_RECORD.size
            if native_collection_diagnostics.collection_format == "NCS2"
            else _NCS1_HEADER.size + len(payloads) * _NCS1_RECORD.size
        )
        entropy_coder = getattr(model, "entropy_coder", "native")
    elif isinstance(model, LFZipCodec):
        # Match the official LFZip command line: bsc -b64p -e2.  A compact
        # stream header makes epsilon and tensor shape self-describing.
        header = struct.pack(
            "<4sfQHH",
            b"LFZ1",
            float(point["epsilon"]),
            int(values),
            len(bundle.channels),
            int(bundle.test.window),
        )
        stream_bits = bsc_compressed_size([header, *payloads]) * 8
        collection_framing_bytes = len(header)
        entropy_coder = "libbsc-3.3.12-e2"
    else:
        stream_bits = compressed_size(payloads) * 8
        collection_framing_bytes = 0
        entropy_coder = "zlib-9"
    if residual_symbol_arrays:
        residual_array = np.concatenate([array.reshape(-1) for array in residual_symbol_arrays])
        _, counts = np.unique(residual_array, return_counts=True)
        probabilities = counts / counts.sum()
        hard_symbol_entropy = float(-(probabilities * np.log2(probabilities)).sum())
        zero_symbol_fraction = float(np.mean(residual_array == 0))
    else:
        hard_symbol_entropy = None
        zero_symbol_fraction = None
    model_size = _model_bytes(model)
    # This is the exact side-information blob required to map standardized
    # values back to physical units, charged once per collection for all methods.
    normalization_blob = _normalization_stream(bundle)
    decoded_channels, decoded_mean, decoded_std = _decode_normalization_stream(
        normalization_blob
    )
    if decoded_channels != bundle.channels or not np.array_equal(
        decoded_mean.view(np.uint32), np.asarray(bundle.standardizer.mean).view(np.uint32)
    ) or not np.array_equal(
        decoded_std.view(np.uint32), np.asarray(bundle.standardizer.std).view(np.uint32)
    ):
        raise RuntimeError("normalization stream round trip changed float32 parameters")
    normalization_bytes = len(normalization_blob)
    normalization_parameter_bytes = 2 * len(bundle.channels) * np.dtype("<f4").itemsize
    normalization_framing_bytes = 8
    channel_schema_bytes = (
        normalization_bytes - normalization_parameter_bytes - normalization_framing_bytes
    )
    self_contained_framing_bytes = 0
    artifact_file: str | None = None
    artifact_sha256: str | None = None
    artifact_byte_roundtrip_verified: bool | None = None
    artifact_numeric_roundtrip_verified: bool | None = None
    if isinstance(model, NativeClassicalCodec):
        model_artifact = (
            _provenance_stream(provenance_identity)
            if provenance_identity is not None
            else b""
        )
        model_size = len(model_artifact)
        self_contained_stream = _self_contained_native_stream(
            str(spec["name"]), normalization_blob, model_artifact, native_stream
        )
        decoded_codec, decoded_normalization, decoded_model, decoded_collection = (
            _decode_self_contained_native_stream(self_contained_stream)
        )
        if (
            decoded_codec != spec["name"]
            or decoded_normalization != normalization_blob
            or decoded_model != model_artifact
            or decoded_collection != native_stream
        ):
            raise RuntimeError("self-contained native stream failed byte-exact round trip")
        self_contained_framing_bytes = 40 + len(str(spec["name"]).encode("utf-8"))
        self_contained_stream_bits = len(self_contained_stream) * 8
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact_path.with_name(f".{artifact_path.name}.tmp")
            temporary.write_bytes(self_contained_stream)
            temporary.replace(artifact_path)
            persisted = artifact_path.read_bytes()
            if persisted != self_contained_stream:
                raise RuntimeError("persisted native stream differs from the measured bytes")
            (
                persisted_codec,
                persisted_normalization,
                persisted_model,
                persisted_collection,
            ) = _decode_self_contained_native_stream(persisted)
            if (
                persisted_codec != spec["name"]
                or persisted_normalization != normalization_blob
                or persisted_model != model_artifact
            ):
                raise RuntimeError("persisted native stream metadata changed on disk")
            if provenance_identity is not None and (
                _decode_provenance_stream(persisted_model) != provenance_identity
            ):
                raise RuntimeError("persisted native provenance changed on disk")
            artifact_byte_roundtrip_verified = True
            payload_decoder = getattr(model, "decompress_payload", None)
            if callable(payload_decoder):
                persisted_payloads = _decode_native_collection_stream(
                    persisted_collection
                )
                persisted_reconstruction = np.concatenate(
                    [payload_decoder(payload) for payload in persisted_payloads]
                )
                expected_reconstruction = np.concatenate(
                    native_standardized_reconstructions
                )
                if not np.array_equal(
                    persisted_reconstruction.view(np.uint32),
                    expected_reconstruction.view(np.uint32),
                ):
                    raise RuntimeError(
                        f"persisted {spec['name']} artifact changed the verified reconstruction"
                    )
                artifact_numeric_roundtrip_verified = True
                if has_qpet_metadata:
                    persisted_original = np.concatenate(
                        native_standardized_originals
                    )
                    persisted_standardized_metrics = _standardized_joint_metrics(
                        persisted_original,
                        persisted_reconstruction,
                        int(getattr(model, "max_lag", spec.get("max_lag", 16))),
                    )
                    _enforce_persisted_joint_constraints(
                        persisted_standardized_metrics, point
                    )
            artifact_file = (Path("bitstreams") / artifact_path.name).as_posix()
            artifact_sha256 = hashlib.sha256(persisted).hexdigest()
    else:
        self_contained_stream_bits = stream_bits + 8 * (model_size + normalization_bytes)
    if has_qpet_metadata and persisted_standardized_metrics is None:
        raise RuntimeError(
            "QPET evaluation did not complete persisted SCS2/QPC standardized replay"
        )
    windows = len(loader.dataset)
    qpet_frontier_suffix = ""
    if spec["name"] == "qpet_correlation" and (
        "base_frontier" in spec or "relaxed_bound_ratio_ceiling" in spec
    ):
        qpet_frontier_suffix = (
            f":bf{native_base_frontier}:rm{native_cjrf_ratio_mask}"
        )
    qpet_surrogate_suffix = ""
    if spec["name"] == "qpet_correlation" and (
        "surrogate_frontier" in spec
        or "cfsr_bound_ratio_ceiling" in spec
    ):
        qpet_surrogate_suffix = (
            f":sf{native_surrogate_frontier}:sm{native_cfsr_ratio_mask}"
        )
    result: dict[str, Any] = {
        "method": spec["name"],
        "method_variant": (
            f"{spec['name']}:{spec.get('repair_optimizer', 'legacy')}:"
            f"cf{native_correction_format_ceiling}{qpet_frontier_suffix}"
            f"{qpet_surrogate_suffix}"
            if spec["name"] == "qpet_correlation"
            else spec["name"]
        ),
        "operating_point": point["label"],
        "values": values,
        "payload_bits": payload_bits,
        "stream_bits": stream_bits,
        "payload_bits_per_value": payload_bits / values,
        "bits_per_value": stream_bits / values,
        "payload_compression_ratio": (values * 32) / payload_bits,
        "compression_ratio": (values * 32) / stream_bits,
        "normalization_bytes": normalization_bytes,
        "normalization_parameter_bytes": normalization_parameter_bytes,
        "normalization_framing_bytes": normalization_framing_bytes,
        "channel_schema_bytes": channel_schema_bytes,
        "self_contained_framing_bytes": self_contained_framing_bytes,
        "collection_framing_bytes": collection_framing_bytes,
        "native_collection_frontier": (
            native_collection_diagnostics.frontier
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_format": (
            native_collection_diagnostics.collection_format
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_ncs1_bytes": (
            native_collection_diagnostics.ncs1_bytes
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_candidate_bytes": (
            native_collection_diagnostics.candidate_bytes
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_saved_bytes": (
            native_collection_diagnostics.saved_bytes
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_compressed_payloads": (
            native_collection_diagnostics.compressed_payloads
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_zstd_level_min": (
            native_collection_diagnostics.zstd_level_min
            if native_collection_diagnostics is not None else None
        ),
        "native_collection_zstd_level_max": (
            native_collection_diagnostics.zstd_level_max
            if native_collection_diagnostics is not None else None
        ),
        "artifact_file": artifact_file,
        "artifact_sha256": artifact_sha256,
        "artifact_byte_roundtrip_verified": artifact_byte_roundtrip_verified,
        "artifact_numeric_roundtrip_verified": artifact_numeric_roundtrip_verified,
        "persisted_standardized_metrics_source": (
            f"SCS2/{native_collection_diagnostics.collection_format}/QPC-replay"
            if persisted_standardized_metrics is not None else None
        ),
        "persisted_standardized_max_abs_error": (
            persisted_standardized_metrics[
                "persisted_standardized_max_abs_error"
            ]
            if persisted_standardized_metrics is not None else None
        ),
        "persisted_standardized_acf_max_abs": (
            persisted_standardized_metrics[
                "persisted_standardized_acf_max_abs"
            ]
            if persisted_standardized_metrics is not None else None
        ),
        "persisted_standardized_correlation_series_min": (
            persisted_standardized_metrics[
                "persisted_standardized_correlation_series_min"
            ]
            if persisted_standardized_metrics is not None else None
        ),
        "provenance_bytes": len(model_artifact) if isinstance(model, NativeClassicalCodec) else 0,
        "stream_accounting": (
            "persisted-native-exact"
            if isinstance(model, NativeClassicalCodec)
            else "legacy-estimate"
        ),
        "self_contained_stream_bits": self_contained_stream_bits,
        "self_contained_bits_per_value": self_contained_stream_bits / values,
        "self_contained_compression_ratio": (
            (values * 32) / self_contained_stream_bits
        ),
        "entropy_coder": entropy_coder,
        "hard_symbol_entropy": hard_symbol_entropy,
        "zero_symbol_fraction": zero_symbol_fraction,
        "parameters": parameter_count(model),
        "model_bytes": model_size,
        "latency_ms_per_window": elapsed * 1000 / windows,
        "training_seconds": training_seconds,
        **reconstruction,
        "task_accuracy": task["accuracy"],
        "task_precision": task["precision"],
        "task_recall": task["recall"],
        "task_f1": task["f1"],
        "utility_retention": task["f1"] / raw_probe_f1 if raw_probe_f1 > 0 else 0.0,
        "codebook_usage": len(used_symbols) / symbol_capacity if symbol_capacity else None,
        "logical_compression_ratio": (
            native_logical_values / native_retained_points if native_retained_points else None
        ),
        "constraint_corrections": native_corrections if native_corrections else None,
        "online_constraint_rejections": (
            native_online_corrections if native_online_corrections else None
        ),
        "constraint_candidates_tested": (
            native_candidates_tested if has_qpet_metadata else None
        ),
        "native_candidate_failures": (
            native_candidate_failures if has_qpet_metadata else None
        ),
        "native_repair_optimizer": (
            native_repair_optimizer if has_qpet_metadata else None
        ),
        "native_correction_format_ceiling": (
            native_correction_format_ceiling if has_qpet_metadata else None
        ),
        "native_base_frontier": (
            native_base_frontier if has_qpet_metadata else None
        ),
        "native_relaxed_bound_ratio_ceiling": (
            native_relaxed_bound_ratio_ceiling if has_qpet_metadata else None
        ),
        "native_cjrf_ratio_mask": (
            native_cjrf_ratio_mask if has_qpet_metadata else None
        ),
        **{
            f"native_{field}": value if has_qpet_metadata else None
            for field, value in native_cjrf.items()
        },
        "native_surrogate_frontier": (
            native_surrogate_frontier if has_qpet_metadata else None
        ),
        "native_cfsr_bound_ratio_ceiling": (
            native_cfsr_bound_ratio_ceiling if has_qpet_metadata else None
        ),
        "native_cfsr_ratio_mask": (
            native_cfsr_ratio_mask if has_qpet_metadata else None
        ),
        **{
            f"native_{field}": value if has_qpet_metadata else None
            for field, value in native_cfsr.items()
        },
        "native_zrcar_base_records": (
            native_zrcar_base_records if has_qpet_metadata else None
        ),
        "native_zrcar_legacy_seeds": (
            native_zrcar_legacy_seeds if has_qpet_metadata else None
        ),
        "native_zrcar_fvro_seeds": (
            native_zrcar_fvro_seeds if has_qpet_metadata else None
        ),
        "native_zrcar_states_generated": (
            native_zrcar_states_generated if has_qpet_metadata else None
        ),
        "native_zrcar_infeasible_states": (
            native_zrcar_infeasible_states if has_qpet_metadata else None
        ),
        "native_zrcar_target_states_pruned": (
            native_zrcar_target_states_pruned if has_qpet_metadata else None
        ),
        "native_zrcar_infeasible_frontier_expansions": (
            native_zrcar_infeasible_frontier_expansions
            if has_qpet_metadata
            else None
        ),
        "native_zrcar_infeasible_frontier_recoveries": (
            native_zrcar_infeasible_frontier_recoveries
            if has_qpet_metadata
            else None
        ),
        "native_zrcar_candidates": (
            native_zrcar_candidates if has_qpet_metadata else None
        ),
        "native_zrcar_prefix_candidates": (
            native_zrcar_prefix_candidates if has_qpet_metadata else None
        ),
        "native_zrcar_local_candidates": (
            native_zrcar_local_candidates if has_qpet_metadata else None
        ),
        "native_zrcar_selected_groups": (
            native_zrcar_selected_groups if has_qpet_metadata else None
        ),
        "native_zrcar_saved_bytes": (
            native_zrcar_saved_bytes if has_qpet_metadata else None
        ),
        "native_zrcar_from_fvro_seed_groups": (
            native_zrcar_from_fvro_seed_groups if has_qpet_metadata else None
        ),
        "native_fvro_base_records": (
            native_fvro_base_records if has_qpet_metadata else None
        ),
        "native_fvro_global_generated": (
            native_fvro_global_generated if has_qpet_metadata else None
        ),
        "native_fvro_global_infeasible": (
            native_fvro_global_infeasible if has_qpet_metadata else None
        ),
        "native_fvro_global_serialized": (
            native_fvro_global_serialized if has_qpet_metadata else None
        ),
        "native_fvro_channel_generated": (
            native_fvro_channel_generated if has_qpet_metadata else None
        ),
        "native_fvro_channel_infeasible": (
            native_fvro_channel_infeasible if has_qpet_metadata else None
        ),
        "native_fvro_channel_serialized": (
            native_fvro_channel_serialized if has_qpet_metadata else None
        ),
        "native_fvro_local_generated": (
            native_fvro_local_generated if has_qpet_metadata else None
        ),
        "native_fvro_local_infeasible": (
            native_fvro_local_infeasible if has_qpet_metadata else None
        ),
        "native_fvro_selected_channels": (
            native_fvro_selected_channels if has_qpet_metadata else None
        ),
        "native_fvro_strict_channel_steps": (
            native_fvro_strict_channel_steps if has_qpet_metadata else None
        ),
        "native_fvro_seeds_retained": (
            native_fvro_seeds_retained if has_qpet_metadata else None
        ),
        "native_fvro_reference_quantized_groups": (
            native_fvro_reference_quantized_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_quantized_groups": (
            native_fvro_final_quantized_groups if has_qpet_metadata else None
        ),
        "native_fvro_reference_format4_groups": (
            native_fvro_reference_format4_groups if has_qpet_metadata else None
        ),
        "native_fvro_reference_format6_groups": (
            native_fvro_reference_format6_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format4_groups": (
            native_fvro_final_format4_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format6_groups": (
            native_fvro_final_format6_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format7_groups": (
            native_fvro_final_format7_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format8_groups": (
            native_fvro_final_format8_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format9_groups": (
            native_fvro_final_format9_groups if has_qpet_metadata else None
        ),
        "native_fvro_final_format10_groups": (
            native_fvro_final_format10_groups if has_qpet_metadata else None
        ),
        "native_fvro_reference_exact_escapes": (
            native_fvro_reference_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_exact_escapes": (
            native_fvro_final_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_reference_format4_exact_escapes": (
            native_fvro_reference_format4_exact_escapes
            if has_qpet_metadata
            else None
        ),
        "native_fvro_reference_format6_exact_escapes": (
            native_fvro_reference_format6_exact_escapes
            if has_qpet_metadata
            else None
        ),
        "native_fvro_final_format4_exact_escapes": (
            native_fvro_final_format4_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_format6_exact_escapes": (
            native_fvro_final_format6_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_format7_exact_escapes": (
            native_fvro_final_format7_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_format8_exact_escapes": (
            native_fvro_final_format8_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_format9_exact_escapes": (
            native_fvro_final_format9_exact_escapes if has_qpet_metadata else None
        ),
        "native_fvro_final_format10_exact_escapes": (
            native_fvro_final_format10_exact_escapes
            if has_qpet_metadata else None
        ),
        "native_fvro_ref6_to_final6_groups": (
            native_fvro_ref6_to_final6_groups if has_qpet_metadata else None
        ),
        "native_fvro_ref6_to_non6_groups": (
            native_fvro_ref6_to_non6_groups if has_qpet_metadata else None
        ),
        "native_fvro_ref6_to_final6_reference_exact_escapes": (
            native_fvro_ref6_to_final6_reference_exact_escapes
            if has_qpet_metadata
            else None
        ),
        "native_fvro_ref6_to_final6_final_exact_escapes": (
            native_fvro_ref6_to_final6_final_exact_escapes
            if has_qpet_metadata
            else None
        ),
        "native_fvro_ref6_to_non6_reference_exact_escapes": (
            native_fvro_ref6_to_non6_reference_exact_escapes
            if has_qpet_metadata
            else None
        ),
        "native_fvro_selected_groups": (
            native_fvro_selected_groups if has_qpet_metadata else None
        ),
        "native_fvro_saved_bytes": (
            native_fvro_saved_bytes if has_qpet_metadata else None
        ),
        **{
            f"native_{field}": native_ssbb[field] if has_qpet_metadata else None
            for field in ssbb_metadata_fields
        },
        "native_second_stage_selected_groups": (
            native_second_stage_selected_groups if has_qpet_metadata else None
        ),
        "native_second_stage_saved_bytes": (
            native_second_stage_saved_bytes if has_qpet_metadata else None
        ),
        "native_base_payload_bytes": (
            native_base_payload_bytes if has_qpet_metadata else None
        ),
        "native_correction_payload_bytes": (
            native_correction_payload_bytes if has_qpet_metadata else None
        ),
        "native_variable_eb_groups": (
            native_variable_eb_groups if has_qpet_metadata else None
        ),
        "native_global_eb_groups": (
            native_global_eb_groups if has_qpet_metadata else None
        ),
        "native_double_correction_groups": (
            native_double_correction_groups if has_qpet_metadata else None
        ),
        "native_float_exact_correction_groups": (
            native_float_exact_correction_groups if has_qpet_metadata else None
        ),
        "native_sparse_exact_correction_groups": (
            native_sparse_exact_correction_groups if has_qpet_metadata else None
        ),
        "native_sparse_xor_correction_groups": (
            native_sparse_xor_correction_groups if has_qpet_metadata else None
        ),
        "native_quantized_correction_groups": (
            native_quantized_correction_groups if has_qpet_metadata else None
        ),
        "native_packed_quantized_correction_groups": (
            native_packed_quantized_correction_groups
            if has_qpet_metadata else None
        ),
        "native_adaptive_packed_correction_groups": (
            native_adaptive_packed_correction_groups
            if has_qpet_metadata else None
        ),
        "native_factorized_packed_correction_groups": (
            native_factorized_packed_correction_groups
            if has_qpet_metadata else None
        ),
        "native_subset_rank_packed_correction_groups": (
            native_subset_rank_packed_correction_groups
            if has_qpet_metadata else None
        ),
        "native_predictive_packed_correction_groups": (
            native_predictive_packed_correction_groups
            if has_qpet_metadata else None
        ),
        "native_constant_channel_correction_groups": (
            native_constant_channel_correction_groups if has_qpet_metadata else None
        ),
        "native_gradient_balanced_groups": (
            native_gradient_balanced_groups if has_qpet_metadata else None
        ),
        "native_sperr_groups": (
            native_sperr_groups if has_qpet_metadata else None
        ),
        "native_sperr_selected_bound_min": (
            native_sperr_selected_bound_min if native_sperr_groups else None
        ),
        "native_sperr_selected_bound_max": (
            native_sperr_selected_bound_max if native_sperr_groups else None
        ),
        "native_lossless_zstd_groups": (
            native_lossless_zstd_groups if has_qpet_metadata else None
        ),
        "native_serf_xor_groups": (
            native_serf_xor_groups if has_qpet_metadata else None
        ),
        "native_serf_qt_groups": (
            native_serf_qt_groups if has_qpet_metadata else None
        ),
        "native_serf_smooth_groups": (
            native_serf_smooth_groups if has_qpet_metadata else None
        ),
        "native_semantic_independent_groups": (
            native_semantic_independent_groups if has_qpet_metadata else None
        ),
        "native_semantic_shared_groups": (
            native_semantic_shared_groups if has_qpet_metadata else None
        ),
        "native_semantic_strict_constant_groups": (
            native_semantic_strict_constant_groups if has_qpet_metadata else None
        ),
        "native_shared_base_search_groups": (
            native_shared_base_search_groups if has_qpet_metadata else None
        ),
        "native_lfzip_order0_windows": (
            native_lfzip_order0_windows if native_lfzip_order0_windows + native_lfzip_order32_windows else None
        ),
        "native_lfzip_order32_windows": (
            native_lfzip_order32_windows if native_lfzip_order0_windows + native_lfzip_order32_windows else None
        ),
        "native_lfzip_qbyte1_windows": (
            native_lfzip_qbyte1_windows if native_lfzip_qbyte1_windows + native_lfzip_qbyte2_windows else None
        ),
        "native_lfzip_qbyte2_windows": (
            native_lfzip_qbyte2_windows if native_lfzip_qbyte1_windows + native_lfzip_qbyte2_windows else None
        ),
        "native_lfzip_base_scale_min": (
            native_lfzip_base_scale_min if np.isfinite(native_lfzip_base_scale_min) else None
        ),
        "native_lfzip_base_scale_max": (
            native_lfzip_base_scale_max if np.isfinite(native_lfzip_base_scale_min) else None
        ),
        "standardized_max_abs_error": (
            native_standardized_max_abs_error
            if has_native_constraint_metadata else None
        ),
        "native_cameo_mean_acf_max": (
            native_cameo_mean_acf_max
            if has_native_constraint_metadata and "cameo" in str(spec["name"])
            else None
        ),
        "native_strict_acf_max_abs": (
            native_strict_acf_max_abs
            if has_native_constraint_metadata else None
        ),
        "native_correlation_series_min": (
            native_correlation_series_min
            if has_native_constraint_metadata else None
        ),
        "cpp_standardized_max_abs_error": (
            cpp_standardized_max_abs_error if has_qpet_metadata else None
        ),
        "cpp_strict_acf_max_abs": (
            cpp_strict_acf_max_abs if has_qpet_metadata else None
        ),
        "cpp_correlation_series_min": (
            cpp_correlation_series_min if has_qpet_metadata else None
        ),
        "native_stream_version": native_stream_version,
        "native_stream_version_min": native_stream_version_min,
        "native_candidate_profile": (
            native_candidate_profile if has_qpet_metadata else None
        ),
        "evaluated_standardized_max_abs_error": evaluated_standardized_max_abs_error,
        "physical_diagnostic_max_abs_error": reconstruction["max_abs_error"],
        "physical_diagnostic_acf_max_abs": acf_metrics["acf_max_abs"],
        "physical_diagnostic_correlation_series_min": reconstruction[
            "correlation_series_min"
        ],
        **acf_metrics,
    }
    return result


def save_outputs(
    output_dir: Path,
    config: dict,
    results: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float]]],
    raw_probe: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"raw_probe": raw_probe, "results": results}, handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)

    plt.figure(figsize=(8, 5))
    for method_variant in sorted({row["method_variant"] for row in results}):
        rows = sorted(
            (row for row in results if row["method_variant"] == method_variant),
            key=lambda row: row["bits_per_value"],
        )
        plt.plot(
            [row["bits_per_value"] for row in rows],
            [row["nrmse"] for row in rows],
            marker="o",
            label=method_variant,
        )
    plt.xlabel("Bits per input value (actual compressed stream)")
    plt.ylabel("Mean per-channel NRMSE")
    plt.xscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "rate_distortion.png", dpi=180)
    plt.close()


def run(
    config_path: Path,
    seed_override: int | None = None,
    run_name_override: str | None = None,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> Path:
    workspace = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve(strict=True)
    config_bytes = config_path.read_bytes()
    frozen_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    require_freeze_manifest = bool(config.get("require_freeze_manifest", False))
    configured_expected_windows = config.get("data", {}).get(
        "expected_test_windows"
    )
    if configured_expected_windows is not None and (
        type(configured_expected_windows) is not int
        or configured_expected_windows <= 0
    ):
        raise ValueError("data.expected_test_windows must be a positive integer")
    if require_freeze_manifest and configured_expected_windows is None:
        raise ValueError("frozen external run requires data.expected_test_windows")
    if require_freeze_manifest and config["data"].get("max_test_windows") is not None:
        raise ValueError("frozen external run forbids data.max_test_windows")
    if require_freeze_manifest and (
        seed_override is not None or run_name_override is not None
    ):
        raise ValueError("frozen external runs forbid seed and run-name overrides")
    if (manifest_path is None) != (manifest_sha256 is None):
        raise ValueError("--manifest and --manifest-sha256 must be supplied together")
    if require_freeze_manifest and manifest_path is None:
        raise ValueError("frozen external run requires a freeze manifest and its SHA-256")
    verified_manifest_sha256: str | None = None
    manifest_bytes: bytes | None = None
    if manifest_path is not None:
        manifest_path = manifest_path.resolve(strict=True)
        verifier = _load_freeze_verifier(workspace)
        verified = verifier(
            manifest_path=manifest_path,
            workspace=workspace,
            config_path=config_path,
            expected_sha256=str(manifest_sha256).lower(),
        )
        manifest_bytes = verified.raw_bytes
        verified_manifest_sha256 = verified.sha256
    if seed_override is not None:
        config["seed"] = int(seed_override)
    if run_name_override is not None:
        config["run_name"] = run_name_override
    output_dir = (workspace / config["output_dir"] / config["run_name"]).resolve()
    if require_freeze_manifest:
        if output_dir.exists():
            raise FileExistsError(
                f"frozen external result directory already exists: {output_dir}"
            )
        output_dir.mkdir(parents=True)
        assert manifest_bytes is not None and verified_manifest_sha256 is not None
        (output_dir / "freeze_manifest.json").write_bytes(manifest_bytes)
        started = {
            "config_sha256": frozen_config_sha256,
            "manifest_sha256": verified_manifest_sha256,
            "schema": "nclab.frozen-run-start.v1",
            "started_unix_time_ns": time.time_ns(),
        }
        (output_dir / "run_started.json").write_text(
            json.dumps(started, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    set_seed(int(config["seed"]))
    device = choose_device(config.get("device", "cpu"))
    print(f"device={device}")
    bundle = build_data(config, workspace)
    expected_test_windows = configured_expected_windows
    if expected_test_windows is not None:
        if len(bundle.test) != expected_test_windows:
            raise RuntimeError(
                f"test window count {len(bundle.test)} != frozen expected count "
                f"{expected_test_windows}"
            )
    training = config["training"]
    train_loader, val_loader, test_loader = bundle.loaders(
        int(training["batch_size"]), int(training.get("num_workers", 0))
    )
    print(
        f"windows train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)} "
        f"channels={len(bundle.channels)} window={config['data']['window']}"
    )

    print("training fixed downstream brake-event probe")
    probe = train_probe(
        len(bundle.channels),
        train_loader,
        val_loader,
        int(training["probe_epochs"]),
        float(training["learning_rate"]),
        device,
    )
    raw_probe = probe_metrics(probe, test_loader, device)
    print(f"raw task probe: {raw_probe}")

    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), checkpoints / "task_probe.pt")

    results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for spec_index, spec in enumerate(config["models"]):
        name = spec["name"]
        spec_sha256 = _canonical_sha256(spec)
        print(f"training {name}")
        if name in {"lfzip", "lfzip_qel"} and bool(spec.get("train_per_epsilon", False)):
            for point_index, epsilon in enumerate(
                float(value) for value in spec["residual_epsilons"]
            ):
                point_spec = copy.deepcopy(spec)
                point_spec["training_epsilon"] = epsilon
                if name == "lfzip_qel":
                    point_spec["qel_epsilon"] = epsilon
                set_seed(int(config["seed"]))
                model = build_model(
                    point_spec,
                    len(bundle.channels),
                    int(config["data"]["window"]),
                    bundle.signal_groups,
                ).to(device)
                model, history, training_seconds = train_codec(
                    model, name, point_spec, train_loader, val_loader, training, device
                )
                label = f"epsilon={epsilon:g}"
                history_key = f"{name}/{label}"
                histories[history_key] = history
                checkpoint_suffix = str(epsilon).replace(".", "p")
                torch.save(model.state_dict(), checkpoints / f"{name}_epsilon_{checkpoint_suffix}.pt")
                point = {"label": label, "epsilon": epsilon}
                point_sha256 = _canonical_sha256(point)
                identity = {
                    "schema": "nclab.native-provenance.v1",
                    "frozen_config_sha256": frozen_config_sha256,
                    "freeze_manifest_sha256": verified_manifest_sha256,
                    "spec_index": spec_index,
                    "point_index": point_index,
                    "spec_sha256": spec_sha256,
                    "point_sha256": point_sha256,
                }
                result = evaluate_codec(
                    model,
                    point_spec,
                    point,
                    test_loader,
                    bundle,
                    probe,
                    raw_probe["f1"],
                    device,
                    training_seconds,
                    output_dir / "bitstreams" /
                    f"{spec_index:02d}_{point_index:02d}_{name}.scs",
                    provenance_identity=(identity if verified_manifest_sha256 else None),
                )
                result.update(identity)
                results.append(result)
                print(
                    f"  {name} {label}: CR={result['compression_ratio']:.2f} "
                    f"H={result['hard_symbol_entropy']:.3f} "
                    f"NRMSE={result['nrmse']:.4f} task_F1={result['task_f1']:.4f}"
                )
            continue
        set_seed(int(config["seed"]))
        model = build_model(
            spec,
            len(bundle.channels),
            int(config["data"]["window"]),
            bundle.signal_groups,
        ).to(device)
        model, history, training_seconds = train_codec(
            model, name, spec, train_loader, val_loader, training, device
        )
        history_key = (
            f"{name}:{spec.get('repair_optimizer', 'legacy')}:"
            f"cf{getattr(model, 'correction_format_ceiling')}"
            + (
                f":bf{getattr(model, 'base_frontier')}:"
                f"rm{getattr(model, 'cjrf_ratio_mask')}"
                if (
                    "base_frontier" in spec
                    or "relaxed_bound_ratio_ceiling" in spec
                )
                else ""
            )
            + (
                f":sf{getattr(model, 'surrogate_frontier')}:"
                f"sm{getattr(model, 'cfsr_ratio_mask')}"
                if (
                    "surrogate_frontier" in spec
                    or "cfsr_bound_ratio_ceiling" in spec
                )
                else ""
            )
            if name == "qpet_correlation"
            else name
        )
        histories[history_key] = history
        if parameter_count(model):
            torch.save(model.state_dict(), checkpoints / f"{name}.pt")
        for point_index, point in enumerate(operating_points(model, spec)):
            point_sha256 = _canonical_sha256(point)
            identity = {
                "schema": "nclab.native-provenance.v1",
                "frozen_config_sha256": frozen_config_sha256,
                "freeze_manifest_sha256": verified_manifest_sha256,
                "spec_index": spec_index,
                "point_index": point_index,
                "spec_sha256": spec_sha256,
                "point_sha256": point_sha256,
            }
            result = evaluate_codec(
                model,
                spec,
                point,
                test_loader,
                bundle,
                probe,
                raw_probe["f1"],
                device,
                training_seconds,
                output_dir / "bitstreams" /
                f"{spec_index:02d}_{point_index:02d}_{name}.scs",
                provenance_identity=(identity if verified_manifest_sha256 else None),
            )
            result.update(identity)
            results.append(result)
            print(
                f"  {name} {point['label']}: CR={result['compression_ratio']:.2f} "
                f"NRMSE={result['nrmse']:.4f} task_F1={result['task_f1']:.4f}"
            )
    save_outputs(output_dir, config, results, histories, raw_probe)
    print(f"results={output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the neural compression benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed")
    parser.add_argument("--run-name", default=None, help="Override the configured output directory name")
    parser.add_argument("--manifest", type=Path, help="Frozen external manifest to verify")
    parser.add_argument(
        "--manifest-sha256",
        help="Prerecorded SHA-256 of --manifest; required for a frozen external run",
    )
    args = parser.parse_args()
    run(
        args.config.resolve(),
        seed_override=args.seed,
        run_name_override=args.run_name,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
    )


if __name__ == "__main__":
    main()
