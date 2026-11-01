"""Fast prospective UCR classification benchmark with persisted-codec replay.

Stages must be run in order::

    python -m experiments.vldb.external_utility.ucr_classification download
    python -m experiments.vldb.external_utility.ucr_classification freeze
    python -m experiments.vldb.external_utility.ucr_classification evaluate

``freeze`` machine-reads and hashes the official TEST values and labels for
identity, but uses only dataset shape and train-only normalization.  It does
not compute a compressed-data or raw-test classification score.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterable
import urllib.request
import zipfile

import numpy as np
import torch
import zstandard

from neural_compression_lab.classical import _qpet_stream_metadata, _wsl_path
from neural_compression_lab.models import build_model
from neural_compression_lab.runner import (
    _canonical_sha256,
    _encode_native_collection_stream,
    _enforce_persisted_joint_constraints,
    _normalization_stream,
    _provenance_stream,
    _self_contained_native_stream,
    _standardized_joint_metrics,
)

from experiments.vldb.utility.artifacts import (
    make_baseline_scs2,
    parse_scs2,
    replay_official_baseline_payload,
    replay_qpet_scs2,
    run_official_baseline,
)

from .protocol import (
    BASELINE_POINT_ABS_TOLERANCE,
    BASELINES,
    EXPECTED_PANEL,
    PROTOCOL,
    PROTOCOL_SHA256,
    QPET_POINTS,
    QPET_ACCEPTANCE_ABS_TOLERANCE,
    QPET_SPEC,
    SELECTION_SEED,
    UCR_METADATA_URL,
    UCR_ROOT_URL,
    canonical_json_bytes,
    canonical_sha256,
    load_freeze,
    write_or_verify_freeze,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "tmp" / "external_utility" / "ucr"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "vldb"
    / "external_utility"
    / "results"
    / "ucr_quick_panel_v5"
)

_NUMERIC_METADATA = ("ID", "Train", "Test", "Class", "Length")
_ALLOWED_METADATA = ("ID", "Type", "Name", "Train", "Test", "Class", "Length")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    if path.read_bytes() != payload:
        raise RuntimeError(f"atomic persistence changed bytes: {path}")


def _download(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "nclab-ucr-benchmark/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")
    temporary.replace(path)


def _safe_extract_dataset(archive: Path, destination: Path, name: str) -> None:
    train_name = f"{name}_TRAIN.txt"
    test_name = f"{name}_TEST.txt"
    if (destination / train_name).is_file() and (destination / test_name).is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = {Path(item.filename).name: item for item in source.infolist()}
        for required in (train_name, test_name):
            item = members.get(required)
            if item is None or item.is_dir():
                raise RuntimeError(f"{archive.name} does not contain {required}")
            output = destination / required
            temporary = output.with_name(f".{output.name}.tmp")
            with source.open(item) as origin, temporary.open("wb") as target:
                shutil.copyfileobj(origin, target)
            temporary.replace(output)


def download(data_root: Path = DEFAULT_DATA) -> dict[str, Any]:
    metadata = data_root / "DataSummary.csv"
    _download(UCR_METADATA_URL, metadata)
    files: list[dict[str, Any]] = [
        {"path": metadata.name, "bytes": metadata.stat().st_size, "sha256": _sha256_file(metadata)}
    ]
    for name in EXPECTED_PANEL:
        archive = data_root / f"{name}.zip"
        _download(f"{UCR_ROOT_URL}/{name}.zip", archive)
        _safe_extract_dataset(archive, data_root / name, name)
        files.append(
            {"path": archive.name, "bytes": archive.stat().st_size, "sha256": _sha256_file(archive)}
        )
    return {"data_root": str(data_root), "files": files}


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            stripped = {str(key).strip(): str(value).strip() for key, value in raw.items()}
            row: dict[str, Any] = {key: stripped[key] for key in _ALLOWED_METADATA}
            for key in _NUMERIC_METADATA:
                if key == "Length" and str(row[key]).lower() == "vary":
                    row[key] = 0
                    continue
                numeric = float(row[key])
                if not numeric.is_integer():
                    raise ValueError(f"non-integral UCR metadata {key}: {row[key]}")
                row[key] = int(numeric)
            rows.append(row)
    if not rows:
        raise RuntimeError("empty UCR metadata")
    return rows


def select_panel(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for input_row in rows:
        row = {key: input_row[key] for key in _ALLOWED_METADATA}
        if not (
            int(row["ID"]) <= 85
            and int(row["Train"]) >= 20
            and 50 <= int(row["Test"]) <= 200
            and 64 <= int(row["Length"]) <= 512
            and int(row["Test"]) * int(row["Length"]) <= 100000
        ):
            continue
        row["selection_sha256"] = hashlib.sha256(
            f"{SELECTION_SEED}:{row['Type']}:{row['Name']}".encode("utf-8")
        ).hexdigest()
        eligible.append(row)
    selected: list[dict[str, Any]] = []
    for kind in sorted({str(row["Type"]) for row in eligible}):
        group = [row for row in eligible if row["Type"] == kind]
        selected.append(min(group, key=lambda item: str(item["selection_sha256"])))
    names = tuple(str(row["Name"]) for row in selected)
    expected_by_type = {
        "ECG": "ECG200",
        "Image": "ArrowHead",
        "Motion": "GunPoint",
        "Sensor": "Earthquakes",
        "Simulated": "ShapeletSim",
        "Spectro": "Meat",
    }
    if {str(row["Type"]): str(row["Name"]) for row in selected} != expected_by_type:
        raise RuntimeError(f"UCR metadata no longer selects the frozen panel: {names}")
    return selected


def _dataset_paths(data_root: Path, name: str) -> tuple[Path, Path, Path]:
    directory = data_root / name
    return (
        data_root / f"{name}.zip",
        directory / f"{name}_TRAIN.txt",
        directory / f"{name}_TEST.txt",
    )


def load_dataset(data_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, train_path, test_path = _dataset_paths(data_root, name)
    train = np.loadtxt(train_path, dtype=np.float32, ndmin=2)
    test = np.loadtxt(test_path, dtype=np.float32, ndmin=2)
    if train.shape[1] < 4 or train.shape[1] != test.shape[1]:
        raise RuntimeError(f"invalid fixed-length UCR dataset: {name}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError(f"non-finite UCR dataset is outside this protocol: {name}")
    return (
        np.ascontiguousarray(train[:, 1:], dtype=np.float32),
        np.ascontiguousarray(train[:, 0], dtype=np.float32),
        np.ascontiguousarray(test[:, 1:], dtype=np.float32),
        np.ascontiguousarray(test[:, 0], dtype=np.float32),
    )


def train_standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean64 = float(np.mean(train, dtype=np.float64))
    centered = train.astype(np.float64) - mean64
    std64 = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(std64) or std64 <= 1e-12:
        raise RuntimeError("UCR train partition has zero global variance")
    mean = np.asarray([mean64], dtype=np.float32)
    std = np.asarray([std64], dtype=np.float32)
    train_std = np.ascontiguousarray((train - mean[0]) / std[0], dtype=np.float32)
    test_std = np.ascontiguousarray((test - mean[0]) / std[0], dtype=np.float32)
    return train_std, test_std, mean, std


def _unit_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    centered = array - array.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 1e-15):
        raise RuntimeError("constant series has undefined Pearson shape")
    return centered / norms


def _pairwise_distances(query: np.ndarray, archive: np.ndarray) -> np.ndarray:
    # Unit vectors make this algebraically equivalent to Euclidean distance,
    # while avoiding a [query, archive, time] temporary.
    squared = np.maximum(0.0, 2.0 - 2.0 * (query @ archive.T))
    return np.sqrt(squared)


def classification_utility(
    train: np.ndarray,
    train_labels: np.ndarray,
    raw_test: np.ndarray,
    test_labels: np.ndarray,
    reconstructed_test: np.ndarray,
    contract_rho: float | None,
    k: int = 5,
) -> dict[str, float | int | None]:
    if raw_test.shape != reconstructed_test.shape or train.shape[1] != raw_test.shape[1]:
        raise ValueError("classification tensors have incompatible shapes")
    if train.shape[0] < k or raw_test.shape[0] < 1:
        raise ValueError("classification task has too few series")
    train_unit = _unit_rows(train)
    raw_unit = _unit_rows(raw_test)
    decoded_unit = _unit_rows(reconstructed_test)
    raw_distance = _pairwise_distances(raw_unit, train_unit)
    decoded_distance = _pairwise_distances(decoded_unit, train_unit)
    raw_order = np.argsort(raw_distance, axis=1, kind="stable")
    decoded_order = np.argsort(decoded_distance, axis=1, kind="stable")
    raw_nn = raw_order[:, 0]
    decoded_nn = decoded_order[:, 0]
    raw_prediction = train_labels[raw_nn]
    decoded_prediction = train_labels[decoded_nn]
    raw_correct = int(np.sum(raw_prediction == test_labels))
    decoded_correct = int(np.sum(decoded_prediction == test_labels))
    agreement = int(np.sum(raw_prediction == decoded_prediction))
    identity = int(np.sum(raw_nn == decoded_nn))
    topk_recall = np.mean(
        [
            len(set(truth[:k].tolist()) & set(found[:k].tolist())) / k
            for truth, found in zip(raw_order, decoded_order)
        ]
    )
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    ideal = float(np.sum(discounts))
    ndcg = np.mean(
        [
            sum(
                discounts[position]
                for position, item in enumerate(found[:k])
                if int(item) in set(truth[:k].tolist())
            )
            / ideal
            for truth, found in zip(raw_order, decoded_order)
        ]
    )
    actual_delta = np.linalg.norm(raw_unit - decoded_unit, axis=1)
    raw_gap = raw_distance[np.arange(raw_distance.shape[0]), raw_order[:, 1]] - raw_distance[
        np.arange(raw_distance.shape[0]), raw_order[:, 0]
    ]
    actual_nn_certificate = raw_gap > 2.0 * actual_delta + 1e-12
    label_margin = np.empty(raw_test.shape[0], dtype=np.float64)
    for index, winner_label in enumerate(raw_prediction):
        same = train_labels == winner_label
        other = ~same
        label_margin[index] = float(
            np.min(raw_distance[index, other]) - np.min(raw_distance[index, same])
        )
    actual_label_certificate = label_margin > 2.0 * actual_delta + 1e-12
    if contract_rho is None or contract_rho < -1.0 or contract_rho > 1.0:
        contract_delta = None
        contract_nn_coverage = None
        contract_label_coverage = None
    else:
        contract_delta = float(np.sqrt(2.0 * (1.0 - contract_rho)))
        if np.any(actual_delta > contract_delta + 1e-10):
            raise RuntimeError("persisted reconstruction exceeds its Pearson shape bound")
        contract_nn_certificate = raw_gap > 2.0 * contract_delta + 1e-12
        contract_label_certificate = label_margin > 2.0 * contract_delta + 1e-12
        contract_nn_coverage = float(np.mean(contract_nn_certificate))
        contract_label_coverage = float(np.mean(contract_label_certificate))
        if np.any(contract_nn_certificate & (raw_nn != decoded_nn)):
            raise RuntimeError("Pearson NN certificate implication failed")
        if np.any(contract_label_certificate & (raw_prediction != decoded_prediction)):
            raise RuntimeError("Pearson label certificate implication failed")
    if np.any(actual_nn_certificate & (raw_nn != decoded_nn)):
        raise RuntimeError("actual-shape NN certificate implication failed")
    if np.any(actual_label_certificate & (raw_prediction != decoded_prediction)):
        raise RuntimeError("actual-shape label certificate implication failed")
    count = int(raw_test.shape[0])
    return {
        "test_cases": count,
        "raw_correct": raw_correct,
        "reconstructed_correct": decoded_correct,
        "raw_accuracy": raw_correct / count,
        "reconstructed_accuracy": decoded_correct / count,
        "accuracy_delta": (decoded_correct - raw_correct) / count,
        "prediction_agreement": agreement / count,
        "nearest_neighbor_identity_retention": identity / count,
        "top5_set_recall": float(topk_recall),
        "ndcg_at_5": float(ndcg),
        "actual_shape_delta_max": float(np.max(actual_delta)),
        "actual_nn_certificate_coverage": float(np.mean(actual_nn_certificate)),
        "actual_label_certificate_coverage": float(np.mean(actual_label_certificate)),
        "contract_shape_delta_bound": contract_delta,
        "contract_nn_certificate_coverage": contract_nn_coverage,
        "contract_label_certificate_coverage": contract_label_coverage,
    }


def _implementation_manifest() -> dict[str, str]:
    paths = (
        "experiments/vldb/external_utility/protocol.py",
        "experiments/vldb/external_utility/ucr_classification.py",
        "experiments/vldb/utility/artifacts.py",
        "src/neural_compression_lab/classical.py",
        "src/neural_compression_lab/data.py",
        "src/neural_compression_lab/metrics.py",
        "src/neural_compression_lab/models.py",
        "src/neural_compression_lab/runner.py",
        "tools/qpet_correlation/qpet_correlation_batch.cpp",
        "tools/qpet_correlation/bin/qpet_correlation_batch",
        "tools/sperr/bin/sperr_batch",
        "tools/sz2016/bin/sz2016_batch",
        "tools/sz2016/sz2016.config",
    )
    return {path: _sha256_file(PROJECT_ROOT / path) for path in paths}


def _runtime_manifest() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "zstandard": zstandard.__version__,
    }


def _run_checked(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
        raise RuntimeError(f"runtime identity command failed: {detail}")
    return process.stdout.strip()


def _normalized_ldd_lines(ldd_output: str) -> list[str]:
    """Canonicalize ldd while retaining dependency identity, excluding ASLR."""
    normalized: list[str] = []
    for raw_line in ldd_output.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if "not found" in line:
            raise RuntimeError(f"unresolved WSL native dependency: {line}")
        line = re.sub(r"\s+\(0x[0-9A-Fa-f]+\)$", " (<load-address>)", line)
        normalized.append(line)
    return sorted(normalized)


def _wsl_native_runtime_manifest() -> dict[str, Any]:
    binaries = (
        "tools/qpet_correlation/bin/qpet_correlation_batch",
        "tools/sperr/bin/sperr_batch",
        "tools/sz2016/bin/sz2016_batch",
    )
    result: dict[str, Any] = {
        "uname": _run_checked(["wsl", "--exec", "uname", "-a"]),
        "ldd_version": _run_checked(["wsl", "--exec", "ldd", "--version"]).splitlines()[0],
        "binaries": {},
    }
    for relative in binaries:
        windows_path = PROJECT_ROOT / relative
        linux_binary = _wsl_path(windows_path)
        ldd_output = _run_checked(["wsl", "--exec", "ldd", linux_binary])
        candidates: list[str] = []
        for line in ldd_output.splitlines():
            match = re.search(r"=>\s+(/\S+)", line)
            if match is not None:
                candidates.append(match.group(1))
                continue
            token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
            if token.startswith("/"):
                candidates.append(token)
        resolved: list[str] = []
        for candidate in sorted(set(candidates)):
            value = _run_checked(
                ["wsl", "--exec", "readlink", "-f", "--", candidate]
            )
            if not value.startswith("/"):
                raise RuntimeError(f"invalid resolved WSL DSO path: {candidate}")
            resolved.append(value)
        hashes: dict[str, str] = {}
        if resolved:
            sha_output = _run_checked(
                ["wsl", "--exec", "sha256sum", "--", *sorted(set(resolved))]
            )
            for line in sha_output.splitlines():
                digest, path = line.split(maxsplit=1)
                hashes[path.strip()] = digest
        result["binaries"][relative] = {
            "wsl_path": linux_binary,
            "ldd_normalized_sha256": canonical_sha256(
                _normalized_ldd_lines(ldd_output)
            ),
            "dependencies": hashes,
        }
    return result


def _git_anchor(output: Path) -> dict[str, Any]:
    commit = _run_checked(["git", "rev-parse", "HEAD"])
    paths = (
        "experiments/vldb/external_utility/protocol.py",
        "experiments/vldb/external_utility/ucr_classification.py",
        str((output / "freeze.json").relative_to(PROJECT_ROOT)).replace("\\", "/"),
    )
    blobs: dict[str, str] = {}
    for relative in paths:
        process = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            timeout=120,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"prospective input is not anchored in git commit {commit}: {relative}"
            )
        working = (PROJECT_ROOT / relative).read_bytes()
        if process.stdout != working:
            raise RuntimeError(
                f"prospective input differs from git commit {commit}: {relative}"
            )
        blobs[relative] = _sha256_bytes(process.stdout)
    return {"commit": commit, "anchored_blobs_sha256": blobs}


def _dataset_manifest(data_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    name = str(row["Name"])
    archive, train_path, test_path = _dataset_paths(data_root, name)
    train, train_labels, test, test_labels = load_dataset(data_root, name)
    train_std, test_std, mean, std = train_standardize(train, test)
    expected = (int(row["Train"]), int(row["Test"]), int(row["Length"]))
    if (train.shape[0], test.shape[0], train.shape[1]) != expected or test.shape[1] != expected[2]:
        raise RuntimeError(f"UCR metadata/data shape mismatch for {name}")
    return {
        "name": name,
        "type": str(row["Type"]),
        "id": int(row["ID"]),
        "classes": int(row["Class"]),
        "train_cases": int(train.shape[0]),
        "test_cases": int(test.shape[0]),
        "length": int(test.shape[1]),
        "selection_sha256": str(row["selection_sha256"]),
        "archive_url": f"{UCR_ROOT_URL}/{name}.zip",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256_file(archive),
        "train_file_sha256": _sha256_file(train_path),
        "test_file_sha256": _sha256_file(test_path),
        "train_values_sha256": _tensor_sha256(train),
        "test_values_sha256": _tensor_sha256(test),
        "train_labels_sha256": _tensor_sha256(train_labels),
        "test_labels_sha256": _tensor_sha256(test_labels),
        "train_standardized_sha256": _tensor_sha256(train_std),
        "test_standardized_sha256": _tensor_sha256(test_std),
        "train_mean_float32_hex": mean.tobytes().hex(),
        "train_std_float32_hex": std.tobytes().hex(),
    }


def freeze(data_root: Path = DEFAULT_DATA, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    download(data_root)
    metadata_path = data_root / "DataSummary.csv"
    selected = select_panel(_metadata_rows(metadata_path))
    manifests = [_dataset_manifest(data_root, row) for row in selected]
    record = {
        "schema": "nclab.ucr-quick-panel-freeze.v5",
        "protocol_sha256": PROTOCOL_SHA256,
        "evidence_tier": PROTOCOL["evidence_tier"],
        "metadata_sha256": _sha256_file(metadata_path),
        "selection_allowed_columns": list(_ALLOWED_METADATA),
        "selected_datasets": manifests,
        "implementation_sha256": _implementation_manifest(),
        "runtime": _runtime_manifest(),
        "wsl_native_runtime": _wsl_native_runtime_manifest(),
        "forbidden_selection_inputs": [
            "UCR ED/DTW/default-rate columns",
            "raw TEST classification accuracy",
            "any reconstructed TEST tensor",
            "any compressed size",
            "any compressed-data task metric",
        ],
    }
    freeze_path = output / "freeze.json"
    digest = write_or_verify_freeze(freeze_path, record)
    return {
        "freeze_file": str(freeze_path.relative_to(PROJECT_ROOT)),
        "freeze_sha256": digest,
        "datasets": [item["name"] for item in manifests],
    }


def _verify_freeze(data_root: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_path = data_root / "DataSummary.csv"
    selected = select_panel(_metadata_rows(metadata_path))
    observed = [_dataset_manifest(data_root, row) for row in selected]
    expected = {
        "metadata_sha256": _sha256_file(metadata_path),
        "selection_allowed_columns": list(_ALLOWED_METADATA),
        "selected_datasets": observed,
        "implementation_sha256": _implementation_manifest(),
        "runtime": _runtime_manifest(),
        "wsl_native_runtime": _wsl_native_runtime_manifest(),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"frozen UCR input changed: {key}")
    return observed


def _bundle_like(mean: np.ndarray, std: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        channels=["value"],
        standardizer=SimpleNamespace(mean=mean, std=std),
    )


def _persist_qpet(
    payloads: list[bytes],
    bundle: SimpleNamespace,
    freeze_sha256: str,
    point: dict[str, Any],
    point_index: int,
    artifact: Path,
) -> dict[str, Any]:
    collection, _ = _encode_native_collection_stream(
        payloads, str(QPET_SPEC["collection_frontier"])
    )
    identity = {
        "schema": "nclab.native-provenance.v1",
        "frozen_config_sha256": PROTOCOL_SHA256,
        "freeze_manifest_sha256": freeze_sha256,
        "spec_index": 0,
        "point_index": int(point_index),
        "spec_sha256": _canonical_sha256(QPET_SPEC),
        "point_sha256": _canonical_sha256(point),
    }
    stream = _self_contained_native_stream(
        "qpet_correlation",
        _normalization_stream(bundle),
        _provenance_stream(identity),
        collection,
    )
    _atomic_write(artifact, stream)
    return identity


def _verify_normalization(parsed: Any, mean: np.ndarray, std: np.ndarray) -> None:
    if parsed.channels != ("value",):
        raise RuntimeError("persisted UCR channel schema changed")
    if not np.array_equal(parsed.mean.view(np.uint32), mean.view(np.uint32)):
        raise RuntimeError("persisted UCR mean changed")
    if not np.array_equal(parsed.std.view(np.uint32), std.view(np.uint32)):
        raise RuntimeError("persisted UCR std changed")


def _contiguous_row_blocks(source: np.ndarray, rows_per_payload: int) -> list[np.ndarray]:
    if source.ndim != 3 or source.shape[0] < 1:
        raise ValueError("QPET source must be a nonempty [batch,channel,time] tensor")
    if rows_per_payload < 1:
        raise ValueError("QPET rows-per-payload must be positive")
    blocks = [
        np.ascontiguousarray(source[start : start + rows_per_payload], dtype=np.float32)
        for start in range(0, source.shape[0], rows_per_payload)
    ]
    if sum(block.shape[0] for block in blocks) != source.shape[0]:
        raise RuntimeError("QPET block partition did not cover the source exactly")
    return blocks


def _qpet_payload_diagnostics(
    payloads: Iterable[bytes], rows_per_payload: int
) -> dict[str, Any]:
    items = list(payloads)
    if not items or rows_per_payload < 1:
        raise ValueError("QPET payload diagnostics require payloads and a positive block size")
    metadata = [_qpet_stream_metadata(payload) for payload in items]
    batch_rows = [int(item["batch"]) for item in metadata]
    if any(value != rows_per_payload for value in batch_rows[:-1]):
        raise RuntimeError("non-final QPET payload has a noncanonical batch size")
    if batch_rows[-1] < 1 or batch_rows[-1] > rows_per_payload:
        raise RuntimeError("final QPET payload has an invalid batch size")
    if any(int(item["channels"]) != 1 for item in metadata):
        raise RuntimeError("UCR QPET payload changed the single-channel schema")
    time_points = {int(item["time"]) for item in metadata}
    if len(time_points) != 1:
        raise RuntimeError("UCR QPET payloads disagree on series length")
    stream_versions = sorted({int(item["version"]) for item in metadata})
    payload_stream_versions = [int(item["version"]) for item in metadata]
    return {
        "payload_bytes": sum(len(payload) for payload in items),
        "native_base_payload_bytes": sum(
            int(item["base_payload_bytes"]) for item in metadata
        ),
        "native_correction_payload_bytes": sum(
            int(item["correction_payload_bytes"]) for item in metadata
        ),
        "native_stream_version": (
            stream_versions[0] if len(stream_versions) == 1 else None
        ),
        "native_stream_versions": stream_versions,
        "native_payload_stream_versions": payload_stream_versions,
        "native_payload_count": len(items),
        "native_rows_per_payload": rows_per_payload,
        "native_payload_batch_rows": batch_rows,
    }


def _compress_qpet_block(
    block: np.ndarray, *, point: dict[str, Any]
) -> tuple[np.ndarray, bytes, dict[str, Any], str]:
    model = build_model(QPET_SPEC, channels=1, window=block.shape[-1], groups={})
    immediate_tensor, payload = model.compress_reconstruct_point(
        torch.from_numpy(block), point
    )
    decoded = np.ascontiguousarray(immediate_tensor.cpu().numpy(), dtype=np.float32)
    if decoded.shape != block.shape or not payload:
        raise RuntimeError("QPET block encoder returned an invalid result")
    return (
        decoded,
        payload,
        dict(getattr(model, "last_metadata", {})),
        str(model.upstream_commit),
    )


def _evaluate_qpet(
    name: str,
    train_std: np.ndarray,
    train_labels: np.ndarray,
    test_std: np.ndarray,
    test_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    point: dict[str, Any],
    point_index: int,
    freeze_sha256: str,
    output: Path,
) -> dict[str, Any]:
    source = np.ascontiguousarray(test_std[:, None, :], dtype=np.float32)
    rows_per_payload = int(PROTOCOL["execution"]["qpet_rows_per_payload"])
    blocks = _contiguous_row_blocks(source, rows_per_payload)
    block_worker = partial(_compress_qpet_block, point=point)
    configured_workers = int(PROTOCOL["execution"]["qpet_block_workers"])
    if configured_workers < 1:
        raise RuntimeError("frozen QPET block-worker count must be positive")
    with ThreadPoolExecutor(
        max_workers=min(configured_workers, len(blocks)),
        thread_name_prefix="qpet-block",
    ) as executor:
        encoded_blocks = list(executor.map(block_worker, blocks))
    immediate_blocks = [item[0] for item in encoded_blocks]
    payloads = [item[1] for item in encoded_blocks]
    block_metadata = [item[2] for item in encoded_blocks]
    upstream_commits = {item[3] for item in encoded_blocks}
    if len(upstream_commits) != 1:
        raise RuntimeError("QPET block encoders disagree on upstream identity")
    block_rows = [int(block.shape[0]) for block in blocks]
    immediate = np.concatenate(immediate_blocks, axis=0)
    if immediate.shape != source.shape or sum(block_rows) != source.shape[0]:
        raise RuntimeError("QPET block partition did not cover the source exactly")
    variant = str(point["variant"])
    artifact = output / "artifacts" / name / f"qpet_{variant}.scs"
    expected_provenance = _persist_qpet(
        payloads,
        _bundle_like(mean, std),
        freeze_sha256,
        point,
        point_index,
        artifact,
    )
    persisted = artifact.read_bytes()
    replay, parsed = replay_qpet_scs2(
        artifact, QPET_SPEC, channels=1, window=source.shape[-1]
    )
    if parsed.payloads != tuple(payloads):
        raise RuntimeError("persisted QPET collection changed a block payload")
    if not np.array_equal(replay.view(np.uint32), immediate.view(np.uint32)):
        raise RuntimeError("persisted QPET replay differs from immediate decoder")
    _verify_normalization(parsed, mean, std)
    if parsed.provenance != expected_provenance:
        raise RuntimeError("persisted QPET PRV1 provenance changed")
    if parsed.byte_size != len(persisted) or parsed.sha256 != _sha256_bytes(persisted):
        raise RuntimeError("persisted QPET parser accounting changed")
    metrics = _standardized_joint_metrics(source, replay, int(QPET_SPEC["max_lag"]))
    _enforce_persisted_joint_constraints(metrics, point)
    task = classification_utility(
        train_std,
        train_labels,
        test_std,
        test_labels,
        replay[:, 0, :],
        (
            max(-1.0, float(point["min_correlation"]) - QPET_ACCEPTANCE_ABS_TOLERANCE)
            if float(point["min_correlation"]) >= 0
            else None
        ),
    )
    payload_diagnostics = _qpet_payload_diagnostics(payloads, rows_per_payload)
    if payload_diagnostics["native_payload_batch_rows"] != block_rows:
        raise RuntimeError("QPET payload headers disagree with the input partition")
    if [int(observed["version"]) for observed in block_metadata] != payload_diagnostics[
        "native_payload_stream_versions"
    ]:
        raise RuntimeError("QPET immediate metadata disagrees with payload versions")
    return {
        "dataset": name,
        "method": "qpet_correlation",
        "variant": variant,
        "operating_point": str(point["label"]),
        "epsilon": float(point["epsilon"]),
        "acf_epsilon": float(point["acf_epsilon"]),
        "min_correlation": float(point["min_correlation"]),
        "values": int(source.size),
        "raw_bytes": int(source.nbytes),
        "artifact_bytes": len(persisted),
        "compression_ratio": source.nbytes / len(persisted),
        "bits_per_value": 8.0 * len(persisted) / source.size,
        "artifact_file": str(artifact.relative_to(output)).replace("\\", "/"),
        "artifact_sha256": _sha256_bytes(persisted),
        "artifact_byte_roundtrip_verified": True,
        "artifact_numeric_roundtrip_verified": True,
        "collection_format": parsed.collection_format,
        "upstream_commit": next(iter(upstream_commits)),
        **metrics,
        **task,
        **payload_diagnostics,
    }


def _evaluate_baseline(
    codec_name: str,
    name: str,
    train_std: np.ndarray,
    train_labels: np.ndarray,
    test_std: np.ndarray,
    test_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float,
    spec_index: int,
    freeze_sha256: str,
    output: Path,
) -> dict[str, Any]:
    source = np.ascontiguousarray(test_std[:, None, :], dtype=np.float32)
    immediate, payload, upstream_commit, metadata = run_official_baseline(
        codec_name, source, epsilon
    )
    stream, collection = make_baseline_scs2(
        codec_name,
        payload,
        _bundle_like(mean, std),
        PROTOCOL_SHA256,
        freeze_sha256,
        spec_index,
        0,
        epsilon,
    )
    artifact = output / "artifacts" / name / f"{codec_name}.scs"
    _atomic_write(artifact, stream)
    persisted = artifact.read_bytes()
    parsed = parse_scs2(artifact)
    _verify_normalization(parsed, mean, std)
    expected_spec = {"name": codec_name, "collection_frontier": "stream_closure_v1"}
    expected_point = {"epsilon": float(epsilon)}
    expected_provenance = {
        "schema": "nclab.native-provenance.v1",
        "frozen_config_sha256": PROTOCOL_SHA256,
        "freeze_manifest_sha256": freeze_sha256,
        "spec_index": int(spec_index),
        "point_index": 0,
        "spec_sha256": _canonical_sha256(expected_spec),
        "point_sha256": _canonical_sha256(expected_point),
    }
    if parsed.provenance != expected_provenance:
        raise RuntimeError(f"persisted {codec_name} PRV1 provenance changed")
    if parsed.byte_size != len(persisted) or parsed.sha256 != _sha256_bytes(persisted):
        raise RuntimeError(f"persisted {codec_name} parser accounting changed")
    if len(parsed.payloads) != 1:
        raise RuntimeError("official baseline SCS2 must contain one payload")
    replay = replay_official_baseline_payload(codec_name, parsed.payloads[0], source.shape)
    immediate = np.ascontiguousarray(immediate, dtype=np.float32)
    replay = np.ascontiguousarray(replay, dtype=np.float32)
    if not np.array_equal(replay.view(np.uint32), immediate.view(np.uint32)):
        raise RuntimeError(f"persisted {codec_name} replay differs from immediate decoder")
    metrics = _standardized_joint_metrics(source, replay, int(QPET_SPEC["max_lag"]))
    if (
        metrics["persisted_standardized_max_abs_error"]
        > epsilon + BASELINE_POINT_ABS_TOLERANCE
    ):
        raise RuntimeError(f"{codec_name} violated the common point-wise bound")
    task = classification_utility(
        train_std,
        train_labels,
        test_std,
        test_labels,
        replay[:, 0, :],
        None,
    )
    return {
        "dataset": name,
        "method": codec_name,
        "variant": "point_error_baseline",
        "operating_point": f"epsilon={epsilon:g}",
        "epsilon": float(epsilon),
        "acf_epsilon": None,
        "min_correlation": None,
        "values": int(source.size),
        "raw_bytes": int(source.nbytes),
        "artifact_bytes": len(persisted),
        "compression_ratio": source.nbytes / len(persisted),
        "bits_per_value": 8.0 * len(persisted) / source.size,
        "artifact_file": str(artifact.relative_to(output)).replace("\\", "/"),
        "artifact_sha256": _sha256_bytes(persisted),
        "artifact_byte_roundtrip_verified": True,
        "artifact_numeric_roundtrip_verified": True,
        "collection_format": parsed.collection_format,
        "payload_bytes": len(payload),
        "upstream_commit": upstream_commit,
        **metrics,
        **task,
        "native_base_payload_bytes": metadata.get("base_payload_bytes"),
        "native_correction_payload_bytes": metadata.get("correction_payload_bytes"),
        "native_stream_version": metadata.get("stream_version"),
        "native_collection_format": collection.collection_format,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["method"]), str(row["variant"])) for row in rows})
    for method, variant in keys:
        group = [row for row in rows if row["method"] == method and row["variant"] == variant]
        cases = sum(int(row["test_cases"]) for row in group)
        raw_correct = sum(int(row["raw_correct"]) for row in group)
        decoded_correct = sum(int(row["reconstructed_correct"]) for row in group)
        raw_bytes = sum(int(row["raw_bytes"]) for row in group)
        artifact_bytes = sum(int(row["artifact_bytes"]) for row in group)
        output.append(
            {
                "method": method,
                "variant": variant,
                "datasets": len(group),
                "test_cases": cases,
                "raw_accuracy_micro": raw_correct / cases,
                "reconstructed_accuracy_micro": decoded_correct / cases,
                "accuracy_delta_micro": (decoded_correct - raw_correct) / cases,
                "accuracy_delta_macro": float(np.mean([row["accuracy_delta"] for row in group])),
                "prediction_agreement_macro": float(
                    np.mean([row["prediction_agreement"] for row in group])
                ),
                "nearest_neighbor_identity_retention_macro": float(
                    np.mean([row["nearest_neighbor_identity_retention"] for row in group])
                ),
                "top5_set_recall_macro": float(np.mean([row["top5_set_recall"] for row in group])),
                "ndcg_at_5_macro": float(np.mean([row["ndcg_at_5"] for row in group])),
                "complete_scs2_compression_ratio": raw_bytes / artifact_bytes,
                "complete_scs2_bits_per_value": 32.0 * artifact_bytes / raw_bytes,
                "worst_point_error": max(
                    float(row["persisted_standardized_max_abs_error"]) for row in group
                ),
                "worst_acf_error": max(
                    float(row["persisted_standardized_acf_max_abs"]) for row in group
                ),
                "worst_min_pearson": min(
                    float(row["persisted_standardized_correlation_series_min"])
                    for row in group
                ),
            }
        )
    return output


def _expected_provenance_for_row(
    row: dict[str, Any], freeze_sha256: str
) -> dict[str, Any]:
    if row["method"] == "qpet_correlation":
        matches = [
            (index, point)
            for index, point in enumerate(QPET_POINTS)
            if point["variant"] == row["variant"]
        ]
        if len(matches) != 1:
            raise RuntimeError("result row does not identify one QPET point")
        point_index, point = matches[0]
        spec = QPET_SPEC
        spec_index = 0
    else:
        matches = [
            (index, baseline)
            for index, baseline in enumerate(BASELINES, start=1)
            if baseline["name"] == row["method"]
        ]
        if len(matches) != 1:
            raise RuntimeError("result row does not identify one official baseline")
        spec_index, baseline = matches[0]
        spec = {
            "name": str(baseline["name"]),
            "collection_frontier": "stream_closure_v1",
        }
        point = {"epsilon": float(baseline["epsilon"])}
        point_index = 0
    return {
        "schema": "nclab.native-provenance.v1",
        "frozen_config_sha256": PROTOCOL_SHA256,
        "freeze_manifest_sha256": freeze_sha256,
        "spec_index": int(spec_index),
        "point_index": int(point_index),
        "spec_sha256": _canonical_sha256(spec),
        "point_sha256": _canonical_sha256(point),
    }


def _final_artifact_audit(
    rows: list[dict[str, Any]],
    data_root: Path,
    output: Path,
    freeze_sha256: str,
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    dataset_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in rows:
        name = str(row["dataset"])
        if name not in dataset_cache:
            train, _, test, _ = load_dataset(data_root, name)
            _, test_std, mean, std = train_standardize(train, test)
            dataset_cache[name] = (test_std, mean, std)
        test_std, mean, std = dataset_cache[name]
        source = np.ascontiguousarray(test_std[:, None, :], dtype=np.float32)
        artifact = output / str(row["artifact_file"])
        persisted = artifact.read_bytes()
        observed_sha = _sha256_bytes(persisted)
        if len(persisted) != int(row["artifact_bytes"]) or observed_sha != row["artifact_sha256"]:
            raise RuntimeError(f"final artifact identity changed: {artifact}")
        parsed = parse_scs2(artifact)
        _verify_normalization(parsed, mean, std)
        if parsed.byte_size != len(persisted) or parsed.sha256 != observed_sha:
            raise RuntimeError(f"final parser accounting changed: {artifact}")
        expected_provenance = _expected_provenance_for_row(row, freeze_sha256)
        if parsed.provenance != expected_provenance:
            raise RuntimeError(f"final PRV1 binding changed: {artifact}")
        qpet_diagnostics: dict[str, Any] | None = None
        if row["method"] == "qpet_correlation":
            if len(parsed.payloads) != int(row["native_payload_count"]):
                raise RuntimeError(f"final QPET payload count changed: {artifact}")
            qpet_diagnostics = _qpet_payload_diagnostics(
                parsed.payloads,
                int(PROTOCOL["execution"]["qpet_rows_per_payload"]),
            )
            for key, value in qpet_diagnostics.items():
                if row.get(key) != value:
                    raise RuntimeError(
                        f"final QPET payload diagnostic changed: {artifact}: {key}"
                    )
            replay, _ = replay_qpet_scs2(
                artifact, QPET_SPEC, channels=1, window=source.shape[-1]
            )
            point = next(
                point for point in QPET_POINTS if point["variant"] == row["variant"]
            )
            metrics = _standardized_joint_metrics(
                source, replay, int(QPET_SPEC["max_lag"])
            )
            _enforce_persisted_joint_constraints(metrics, point)
        else:
            if len(parsed.payloads) != 1:
                raise RuntimeError(f"final baseline payload count changed: {artifact}")
            replay = replay_official_baseline_payload(
                str(row["method"]), parsed.payloads[0], source.shape
            )
            metrics = _standardized_joint_metrics(
                source, replay, int(QPET_SPEC["max_lag"])
            )
            if (
                metrics["persisted_standardized_max_abs_error"]
                > float(row["epsilon"]) + BASELINE_POINT_ABS_TOLERANCE
            ):
                raise RuntimeError(f"final baseline point gate failed: {artifact}")
        for key, value in metrics.items():
            if float(row[key]) != float(value):
                raise RuntimeError(f"final reconstructed metric changed: {artifact}: {key}")
        audited.append(
            {
                "artifact_file": str(row["artifact_file"]),
                "artifact_bytes": len(persisted),
                "artifact_sha256": observed_sha,
                "codec_id": parsed.codec_id,
                "collection_format": parsed.collection_format,
                "provenance_sha256": canonical_sha256(parsed.provenance),
                "numeric_replay_verified": True,
                "qpet_payload_diagnostics": qpet_diagnostics,
            }
        )
    return audited


def _write_results(output: Path, result: dict[str, Any]) -> tuple[str, str]:
    json_path = output / "results.json"
    csv_path = output / "results.csv"
    if json_path.exists() or csv_path.exists():
        raise RuntimeError(
            "result files already exist; use a new output directory instead of overwriting evidence"
        )
    json_payload = canonical_json_bytes(result) + b"\n"
    _atomic_write(json_path, json_payload)
    rows = list(result["rows"])
    columns = sorted({key for row in rows for key in row})
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(csv_path, text.getvalue().encode("utf-8"))
    return _sha256_file(json_path), _sha256_file(csv_path)


def _preflight_empty_output(output: Path) -> None:
    forbidden = (output / "results.json", output / "results.csv", output / "artifacts")
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "evaluation output already exists; use a new evidence directory: "
            + ", ".join(existing)
        )


def _identity_guard(
    data_root: Path,
    output: Path,
    freeze_record: dict[str, Any],
    freeze_sha256: str,
    git_anchor: dict[str, Any],
) -> None:
    """Revalidate every frozen identity at a precisely defined run boundary."""
    observed_record, observed_sha256 = load_freeze(output / "freeze.json")
    if observed_sha256 != freeze_sha256 or observed_record != freeze_record:
        raise RuntimeError("freeze changed during UCR evaluation")
    _verify_freeze(data_root, freeze_record)
    if _git_anchor(output) != git_anchor:
        raise RuntimeError("Git evidence anchor changed during UCR evaluation")


def _ordered_parallel_dataset_rows(
    manifests: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], list[dict[str, Any]]],
    max_workers: int,
) -> list[dict[str, Any]]:
    """Run independent datasets concurrently while preserving frozen order."""
    if max_workers < 1:
        raise ValueError("dataset worker count must be positive")
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(manifests)),
        thread_name_prefix="ucr-dataset",
    ) as executor:
        groups = list(executor.map(worker, manifests))
    return [row for group in groups for row in group]


def _evaluate_dataset(
    manifest: dict[str, Any],
    *,
    data_root: Path,
    freeze_sha256: str,
    output: Path,
) -> list[dict[str, Any]]:
    name = str(manifest["name"])
    train, train_labels, test, test_labels = load_dataset(data_root, name)
    train_std, test_std, mean, std = train_standardize(train, test)
    rows: list[dict[str, Any]] = []
    for point_index, point in enumerate(QPET_POINTS):
        rows.append(
            _evaluate_qpet(
                name,
                train_std,
                train_labels,
                test_std,
                test_labels,
                mean,
                std,
                point,
                point_index,
                freeze_sha256,
                output,
            )
        )
    for baseline_index, baseline in enumerate(BASELINES, start=1):
        rows.append(
            _evaluate_baseline(
                str(baseline["name"]),
                name,
                train_std,
                train_labels,
                test_std,
                test_labels,
                mean,
                std,
                float(baseline["epsilon"]),
                baseline_index,
                freeze_sha256,
                output,
            )
        )
    return rows


def evaluate(data_root: Path = DEFAULT_DATA, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _preflight_empty_output(output)
    freeze_record, freeze_sha = load_freeze(output / "freeze.json")
    git_anchor = _git_anchor(output)
    manifests = _verify_freeze(data_root, freeze_record)
    worker = partial(
        _evaluate_dataset,
        data_root=data_root,
        freeze_sha256=freeze_sha,
        output=output,
    )
    rows = _ordered_parallel_dataset_rows(
        manifests,
        worker,
        int(PROTOCOL["execution"]["dataset_workers"]),
    )
    # Re-hash every input, implementation file, native DSO, and the Git anchor
    # after all codecs have finished.  This prevents a mixed run if anything
    # changes after the initial preflight.
    _identity_guard(data_root, output, freeze_record, freeze_sha, git_anchor)
    artifact_audit = _final_artifact_audit(rows, data_root, output, freeze_sha)
    result = {
        "schema": "nclab.ucr-quick-panel-results.v5",
        "evidence_tier": PROTOCOL["evidence_tier"],
        "protocol_sha256": PROTOCOL_SHA256,
        "freeze_sha256": freeze_sha,
        "git_anchor": git_anchor,
        "dataset_order": [item["name"] for item in manifests],
        "rows": rows,
        "aggregate": _aggregate(rows),
        "artifact_audit": artifact_audit,
        "claim_scope": PROTOCOL["claim_scope"],
    }
    result_sha, csv_sha = _write_results(output, result)
    if _final_artifact_audit(rows, data_root, output, freeze_sha) != artifact_audit:
        raise RuntimeError("artifact audit changed while result files were committed")
    # This is deliberately the final operation before returning evidence.  It
    # closes the interval occupied by the second decoder replay and result
    # persistence, so a late code, input, DSO, freeze, or Git change cannot be
    # reported under the run's original identity.
    _identity_guard(data_root, output, freeze_record, freeze_sha, git_anchor)
    return {
        "results_file": str((output / "results.json").relative_to(PROJECT_ROOT)),
        "results_sha256": result_sha,
        "csv_sha256": csv_sha,
        "aggregate": result["aggregate"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "freeze", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage == "download":
        value = download(args.data_root)
    elif args.stage == "freeze":
        value = freeze(args.data_root, args.output)
    else:
        value = evaluate(args.data_root, args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
