#!/usr/bin/env python3
"""Create an auditable, canonical experiment-freeze manifest.

The tool hashes inputs and implementation artifacts only.  It never imports
the experiment runner, loads a dataset, trains a model, or runs compression.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import yaml


SCHEMA = "nclab.experiment-freeze-manifest.v2"
HASH_ALGORITHM = "sha256"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
CANONICALIZATION = "UTF-8; JSON keys sorted; compact separators; one trailing LF"
DATA_ROLES: tuple[tuple[str, str], ...] = (
    ("train", "train_files"),
    ("validation", "val_files"),
    ("test", "test_files"),
)
TOP_LEVEL_FIELDS = {
    "artifacts",
    "canonicalization",
    "git",
    "hash_algorithm",
    "manifest_output",
    "runtime",
    "schema",
    "upstreams",
}


@dataclasses.dataclass(frozen=True)
class UpstreamPin:
    name: str
    checkout: str
    repository: str
    commit: str


@dataclasses.dataclass(frozen=True)
class VerifiedManifest:
    """One-read verification result for runner/audit hand-off."""

    document: Mapping[str, object]
    raw_bytes: bytes
    sha256: str
    role_paths: tuple[tuple[str, tuple[Path, ...]], ...]


OFFICIAL_PINS: tuple[UpstreamPin, ...] = (
    UpstreamPin(
        "FPcompress",
        "tmp/upstreams/FPcompress",
        "https://github.com/burtscher/FPcompress.git",
        "97f037249bd28682bfd83462ea1129e14df36d81",
    ),
    UpstreamPin(
        "LFZip",
        "tmp/upstreams/LFZip-official",
        "https://github.com/shubhamchandak94/LFZip.git",
        "14b2e9f79b1d9bc3f60d5cb69e20eb1529c709b4",
    ),
    UpstreamPin(
        "QPET",
        "tmp/upstreams/QPET-szfamily-revision",
        "https://github.com/JLiu-1/QPET-Artifact.git",
        "5d17cb1647caf001bd7355c617856c72c4b6475e",
    ),
    UpstreamPin(
        "SPERR",
        "tmp/upstreams/SPERR",
        "https://github.com/NCAR/SPERR.git",
        "9f930a3970f86c8e390e135c97162b79a40d165b",
    ),
    UpstreamPin(
        "Serf",
        "tmp/upstreams/Serf",
        "https://github.com/Spatio-Temporal-Lab/Serf.git",
        "b38450b56825eabc96be8e25d6880127dc688c95",
    ),
    UpstreamPin(
        "SZ2-historical",
        "tools/sz2016/build/sz-1.1",
        "https://github.com/szcompressor/sz2.git",
        "2123f4899b76e6dfe28861da0a70d63a6cd43f83",
    ),
    UpstreamPin(
        "SymEngine",
        "tmp/upstreams/symengine-v0.14.0",
        "https://github.com/symengine/symengine.git",
        "fac9314c78f2809570494017efc6603befeb4eda",
    ),
    UpstreamPin(
        "fpzip",
        "tmp/upstreams/fpzip",
        "https://github.com/LLNL/fpzip.git",
        "4a539c06d98b1c029b08324a086d4b75689a2b72",
    ),
)


IMPLEMENTATION_FILES: Mapping[str, str] = {
    "audit_results": "tools/qpet_correlation/audit_results.py",
    "audit_results_test": "tools/qpet_correlation/test_audit_results.py",
    "bundle_frozen_run": "tools/qpet_correlation/bundle_frozen_run.py",
    "bundle_frozen_run_test": (
        "tools/qpet_correlation/test_bundle_frozen_run.py"
    ),
    "classical_adapters": "src/neural_compression_lab/classical.py",
    "cfsr_development_protocol": "docs/CFSR_DEVELOPMENT_PROTOCOL.md",
    "data_pipeline": "src/neural_compression_lab/data.py",
    "fvro_gate_evaluator": (
        "tools/qpet_correlation/evaluate_fvro_gates.py"
    ),
    "fvro_gate_evaluator_test": (
        "tools/qpet_correlation/test_evaluate_fvro_gates.py"
    ),
    "fpcompress_binary": "tools/fpcompress/bin/fpcompress_batch",
    "fpcompress_build": "tools/fpcompress/build_wsl.sh",
    "fpcompress_cmake": "tools/fpcompress/CMakeLists.txt",
    "fpcompress_decode_bridge": "tools/fpcompress/fpcompress_decode_bridge.cpp",
    "fpcompress_encode_bridge": "tools/fpcompress/fpcompress_encode_bridge.cpp",
    "fpcompress_wrapper": "tools/fpcompress/fpcompress_batch.cpp",
    "fpzip_binary": "tools/fpzip/bin/fpzip_batch",
    "fpzip_build": "tools/fpzip/build_wsl.sh",
    "fpzip_cmake": "tools/fpzip/CMakeLists.txt",
    "fpzip_wrapper": "tools/fpzip/fpzip_batch.cpp",
    "fpzip_adapter": "tools/fpzip/fpzip_adapter.cpp",
    "fpzip_adapter_header": "tools/fpzip/fpzip_adapter.hpp",
    "freeze_tool": "tools/qpet_correlation/freeze_manifest.py",
    "freeze_tool_test": "tools/qpet_correlation/test_freeze_manifest.py",
    "fvro_development_protocol": "docs/FVRO_DEVELOPMENT_PROTOCOL.md",
    "fvro_novelty_audit": "docs/FVRO_NOVELTY_AUDIT.md",
    "fixed_serializer_ablation": (
        "tools/qpet_correlation/fixed_serializer_ablation.py"
    ),
    "fixed_serializer_ablation_test": (
        "tools/qpet_correlation/test_fixed_serializer_ablation.py"
    ),
    "generic_validation_smoke": "tools/qpet_correlation/generic_validation_smoke.py",
    "metrics": "src/neural_compression_lab/metrics.py",
    "model_registry": "src/neural_compression_lab/models.py",
    "project_metadata": "pyproject.toml",
    "qpet_binary": "tools/qpet_correlation/bin/qpet_correlation_batch",
    "qpet_build": "tools/qpet_correlation/build_wsl.sh",
    "qpet_cmake": "tools/qpet_correlation/CMakeLists.txt",
    "qpet_cpp": "tools/qpet_correlation/qpet_correlation_batch.cpp",
    "qpet_feasible_value_header": (
        "tools/qpet_correlation/FeasibleValueTargets.hpp"
    ),
    "qpet_feasible_value_test": (
        "tools/qpet_correlation/test_feasible_value_targets.cpp"
    ),
    "qpet_header": "tools/qpet_correlation/CorrelationQoI.hpp",
    "qpet_readme": "tools/qpet_correlation/README.md",
    "qpet_sequence_beam_header": (
        "tools/qpet_correlation/SequenceBeamSearch.hpp"
    ),
    "qpet_sequence_beam_test": (
        "tools/qpet_correlation/test_sequence_beam_search.cpp"
    ),
    "qpet_subset_rank_header": (
        "tools/qpet_correlation/SubsetRankCodec.hpp"
    ),
    "qpet_subset_rank_test": (
        "tools/qpet_correlation/test_subset_rank_codec.cpp"
    ),
    "qpet_upstream_buffer_patch": (
        "tools/qpet_correlation/patches/0001-fix-tuning-huffman-buffer.patch"
    ),
    "qpet_upstream_ub_patch": (
        "tools/qpet_correlation/patches/"
        "0002-eliminate-sanitizer-undefined-behavior.patch"
    ),
    "qpet_upstream_fail_closed_patch": (
        "tools/qpet_correlation/patches/"
        "0003-fail-closed-huffman-and-symbol-count.patch"
    ),
    "qpet_upstream_safety_test": (
        "tools/qpet_correlation/test_qpet_upstream_safety.cpp"
    ),
    "qpet_qoi_test": "tools/qpet_correlation/test_correlation_qoi.cpp",
    "qpet_serf_runtime": "tmp/builds/Serf-b38450b/libserf.so",
    "qpet_serf_test": "tools/qpet_correlation/test_serf_official.cpp",
    "runner": "src/neural_compression_lab/runner.py",
    "runner_integration_test": "tests/test_lab.py",
    "sperr_binary": "tools/sperr/bin/sperr_batch",
    "sperr_build": "tools/sperr/build_wsl.sh",
    "sperr_wrapper": "tools/sperr/sperr_batch.cpp",
    "sz2016_binary": "tools/sz2016/bin/sz2016_batch",
    "sz2016_build": "tools/sz2016/build_wsl.sh",
    "sz2016_config": "tools/sz2016/sz2016.config",
    "sz2016_wrapper": "tools/sz2016/sz2016_batch.c",
}

RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "scipy",
    "torch",
    "zstandard",
)


class ManifestError(RuntimeError):
    """A fail-fast manifest validation error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _run_git(repository: Path, *arguments: str, allow_failure: bool = False) -> str:
    command = [
        "git",
        "-c",
        "core.quotepath=false",
        "-C",
        str(repository),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise ManifestError(f"cannot execute git: {error}") from error
    if result.returncode != 0:
        if allow_failure:
            return ""
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise ManifestError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return result.stdout.strip()


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"{label} does not exist or cannot be resolved: {path}") from error
    if not resolved.is_file():
        raise ManifestError(f"{label} is not a regular file: {resolved}")
    return resolved


def _display_path(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_file(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ManifestError(f"cannot hash {path}: {error}") from error
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size != after.st_size
    ):
        raise ManifestError(f"file changed while it was being hashed: {path}")
    return digest.hexdigest(), size


def _file_record(path: Path, workspace: Path) -> dict[str, object]:
    digest, size = _sha256_file(path)
    return {
        "path": _display_path(path, workspace),
        "sha256": digest,
        "size_bytes": size,
    }


def _git_record(repository: Path, workspace: Path) -> dict[str, object]:
    if not repository.is_dir():
        raise ManifestError(f"git repository directory is missing: {repository}")
    commit = _run_git(repository, "rev-parse", "--verify", "HEAD")
    toplevel = Path(_run_git(repository, "rev-parse", "--show-toplevel")).resolve()
    status_lines = sorted(
        line
        for line in _run_git(
            repository, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    status_text = "\n".join(status_lines).encode("utf-8")
    tracked_diff = _run_git(repository, "diff", "--binary", "HEAD").encode("utf-8")
    branch = _run_git(repository, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    return {
        "branch": branch or None,
        "commit": commit,
        "dirty": bool(status_lines),
        "path": _display_path(toplevel, workspace),
        "status": status_lines,
        "status_sha256": hashlib.sha256(status_text).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _upstream_record(pin: UpstreamPin, workspace: Path) -> dict[str, object]:
    checkout = (workspace / pin.checkout).resolve()
    git = _git_record(checkout, workspace)
    actual_commit = str(git["commit"])
    if actual_commit != pin.commit:
        raise ManifestError(
            f"{pin.name} checkout is at {actual_commit}; expected pinned commit {pin.commit}"
        )
    origin = _run_git(checkout, "remote", "get-url", "origin")
    if origin.rstrip("/") != pin.repository.rstrip("/"):
        raise ManifestError(
            f"{pin.name} origin is {origin}; expected official repository {pin.repository}"
        )
    return {
        "actual_commit": actual_commit,
        "checkout": _display_path(checkout, workspace),
        "dirty": git["dirty"],
        "expected_commit": pin.commit,
        "matches_pin": True,
        "origin": origin,
        "repository": pin.repository,
        "status": git["status"],
        "status_sha256": git["status_sha256"],
        "tracked_diff_bytes": git["tracked_diff_bytes"],
        "tracked_diff_sha256": git["tracked_diff_sha256"],
    }


def _normalize_timestamp(timestamp: str | None) -> str:
    if timestamp is None:
        value = dt.datetime.now(dt.timezone.utc)
    else:
        candidate = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        try:
            value = dt.datetime.fromisoformat(candidate)
        except ValueError as error:
            raise ManifestError(f"invalid ISO-8601 timestamp: {timestamp}") from error
        if value.tzinfo is None:
            raise ManifestError("--timestamp must include a UTC offset or trailing Z")
        value = value.astimezone(dt.timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _runtime_record(captured_at_utc: str) -> dict[str, object]:
    captured = dt.datetime.fromisoformat(captured_at_utc.replace("Z", "+00:00"))
    local_at_capture = captured.astimezone()
    offset = local_at_capture.utcoffset()
    package_versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise ManifestError(f"required runtime package is not installed: {package}") from error
    return {
        "captured_at_utc": captured_at_utc,
        "local_timezone": str(local_at_capture.tzinfo),
        "local_utc_offset_seconds": int(offset.total_seconds()) if offset is not None else None,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:5]),
        },
        "python_packages": package_versions,
    }


def _unique_sorted_files(paths: Iterable[Path], label: str) -> list[Path]:
    resolved: dict[str, Path] = {}
    for path in paths:
        item = _resolve_file(path, label)
        key = os.path.normcase(str(item))
        if key in resolved:
            raise ManifestError(f"duplicate {label}: {item}")
        resolved[key] = item
    return sorted(resolved.values(), key=lambda item: item.as_posix())


def _load_experiment_config(config_file: Path) -> dict[str, object]:
    try:
        with config_file.open("r", encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot parse experiment config {config_file}: {error}") from error
    if not isinstance(config, dict):
        raise ManifestError("experiment config must be a YAML mapping")
    _validate_correction_format_ceilings(config)
    _validate_collection_frontiers(config)
    _validate_base_frontiers(config)
    _validate_surrogate_frontiers(config)
    return config


def _validate_correction_format_ceilings(
    config: Mapping[str, object],
) -> None:
    """Validate the nested SSBB correction portfolio without reading data.

    ``correction_format_ceiling`` post-dates the original frozen manifests, so
    an omitted ``models`` section or an omitted field remains valid.  Once the
    field is present it is an exact protocol value: bools and stringified
    integers are deliberately rejected, and formats 7/8/9/10 are SSBB-only.
    """

    models = config.get("models")
    if models is None:
        return
    if not isinstance(models, list):
        raise ManifestError("experiment config models must be a list")
    for spec_index, spec in enumerate(models):
        if not isinstance(spec, dict):
            raise ManifestError(f"experiment config model {spec_index} must be a mapping")
        if spec.get("name") != "qpet_correlation":
            continue
        if "correction_format_ceiling" not in spec:
            continue
        ceiling = spec["correction_format_ceiling"]
        if type(ceiling) is not int or ceiling not in {6, 7, 8, 9, 10}:
            raise ManifestError(
                f"experiment config model {spec_index} "
                "correction_format_ceiling must be the integer 6, 7, 8, 9, or 10"
            )
        optimizer = spec.get("repair_optimizer", "legacy")
        if optimizer not in {"ssbb_value", "ssbb"} and ceiling != 6:
            raise ManifestError(
                f"experiment config model {spec_index} non-SSBB repair "
                "optimizers require correction_format_ceiling=6"
            )


def _validate_collection_frontiers(config: Mapping[str, object]) -> None:
    """Validate the optional versioned collection-level stream frontier.

    The field applies to every model family because collection closure wraps a
    complete persisted native stream rather than a QPET-only record.  Omission
    preserves legacy configs.  Once present, the value is a strict protocol
    string so YAML booleans, containers, nulls, and unknown future spellings
    cannot silently select an adapter default.
    """

    models = config.get("models")
    if models is None:
        return
    if not isinstance(models, list):
        raise ManifestError("experiment config models must be a list")
    for spec_index, spec in enumerate(models):
        if not isinstance(spec, dict):
            raise ManifestError(f"experiment config model {spec_index} must be a mapping")
        if "collection_frontier" not in spec:
            continue
        frontier = spec["collection_frontier"]
        if type(frontier) is not str or frontier not in {
            "none",
            "stream_closure_v1",
        }:
            raise ManifestError(
                f"experiment config model {spec_index} collection_frontier "
                "must be the string none or stream_closure_v1"
            )


def _validate_base_frontiers(config: Mapping[str, object]) -> None:
    """Validate the versioned encoder-only CJRF configuration contract.

    Both fields post-date the v2 manifest schema. Complete omission therefore
    retains the byte-compatible incumbent-only policy. Once either field is
    declared, both become mandatory so a frozen config cannot silently rely on
    an adapter or native-CLI default.
    """

    models = config.get("models")
    if models is None:
        return
    if not isinstance(models, list):
        raise ManifestError("experiment config models must be a list")
    for spec_index, spec in enumerate(models):
        if not isinstance(spec, dict) or spec.get("name") != "qpet_correlation":
            continue
        has_frontier = "base_frontier" in spec
        has_ratio = "relaxed_bound_ratio_ceiling" in spec
        if not has_frontier and not has_ratio:
            continue
        if has_frontier != has_ratio:
            raise ManifestError(
                f"experiment config model {spec_index} base_frontier and "
                "relaxed_bound_ratio_ceiling must be configured together"
            )
        frontier = spec["base_frontier"]
        ratio = spec["relaxed_bound_ratio_ceiling"]
        if frontier not in {"incumbent_only", "cjrf_v1"}:
            raise ManifestError(
                f"experiment config model {spec_index} base_frontier must be "
                "incumbent_only or cjrf_v1"
            )
        if frontier == "incumbent_only":
            if ratio != "none":
                raise ManifestError(
                    f"experiment config model {spec_index} incumbent_only "
                    "requires relaxed_bound_ratio_ceiling=none"
                )
            continue
        if (
            type(ratio) not in {int, float}
            or not math.isfinite(float(ratio))
            or float(ratio) not in {1.25, 1.5, 2.0}
        ):
            raise ManifestError(
                f"experiment config model {spec_index} CJRF relaxed ratio "
                "ceiling must be exactly 1.25, 1.5, or 2.0"
            )
        if (
            spec.get("repair_optimizer") != "ssbb"
            or spec.get("correction_format_ceiling") != 9
            or spec.get("correction_mode") != "posthoc_greedy"
            or spec.get("candidate_profile") != "generic"
            or spec.get("group_size") != 1
        ):
            raise ManifestError(
                f"experiment config model {spec_index} CJRF v1 requires "
                "generic full SSBB, posthoc_greedy, group_size=1, and "
                "correction_format_ceiling=9"
            )


def _validate_surrogate_frontiers(config: Mapping[str, object]) -> None:
    """Validate the versioned CFSR feasible-surrogate frontier contract.

    Complete omission preserves the pre-CFSR native invocation.  An explicit
    pair freezes either a zero-work baseline or one of the three fixed nested
    CFSR prefixes; active CFSR v1 is deliberately not composable with CJRF.
    """

    models = config.get("models")
    if models is None:
        return
    if not isinstance(models, list):
        raise ManifestError("experiment config models must be a list")
    for spec_index, spec in enumerate(models):
        if not isinstance(spec, dict) or spec.get("name") != "qpet_correlation":
            continue
        has_frontier = "surrogate_frontier" in spec
        has_ratio = "cfsr_bound_ratio_ceiling" in spec
        if not has_frontier and not has_ratio:
            continue
        if has_frontier != has_ratio:
            raise ManifestError(
                f"experiment config model {spec_index} surrogate_frontier and "
                "cfsr_bound_ratio_ceiling must be configured together"
            )
        frontier = spec["surrogate_frontier"]
        ratio = spec["cfsr_bound_ratio_ceiling"]
        if frontier not in {"none", "cfsr_v1"}:
            raise ManifestError(
                f"experiment config model {spec_index} surrogate_frontier must "
                "be none or cfsr_v1"
            )
        if frontier == "none":
            if ratio != "none":
                raise ManifestError(
                    f"experiment config model {spec_index} surrogate frontier "
                    "none requires cfsr_bound_ratio_ceiling=none"
                )
            continue
        if (
            type(ratio) not in {int, float}
            or not math.isfinite(float(ratio))
            or float(ratio) not in {0.25, 0.5, 1.0}
        ):
            raise ManifestError(
                f"experiment config model {spec_index} CFSR bound ratio "
                "ceiling must be exactly 0.25, 0.5, or 1.0"
            )
        if (
            spec.get("repair_optimizer") != "ssbb"
            or spec.get("correction_format_ceiling") != 9
            or spec.get("correction_mode") != "posthoc_greedy"
            or spec.get("candidate_profile") != "generic"
            or spec.get("group_size") != 1
            or spec.get("base_frontier", "incumbent_only")
            != "incumbent_only"
            or spec.get("relaxed_bound_ratio_ceiling", "none") != "none"
        ):
            raise ManifestError(
                f"experiment config model {spec_index} CFSR v1 requires "
                "generic full SSBB, posthoc_greedy, group_size=1, "
                "correction_format_ceiling=9, and the "
                "incumbent_only/none base frontier"
            )


def _validate_external_contract(
    config: Mapping[str, object],
    data: Mapping[str, object],
) -> None:
    external = config.get("require_freeze_manifest", False)
    if type(external) is not bool:
        raise ManifestError("require_freeze_manifest must be a boolean")
    if not external:
        return
    expected_windows = data.get("expected_test_windows")
    if type(expected_windows) is not int or expected_windows <= 0:
        raise ManifestError(
            "external config data.expected_test_windows must be a positive integer"
        )
    if data.get("max_test_windows") is not None:
        raise ManifestError(
            "external config data.max_test_windows must be absent or null"
        )


def _configured_data_files(
    config_file: Path,
    workspace: Path,
) -> dict[str, list[Path]]:
    """Resolve the runner's train/validation/test files without loading them."""
    config = _load_experiment_config(config_file)
    data = config.get("data")
    if not isinstance(data, dict):
        raise ManifestError("experiment config must contain a data mapping")
    _validate_external_contract(config, data)
    root_value = data.get("root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ManifestError("experiment config data.root must be a non-empty path")
    data_root = Path(root_value).expanduser()
    if not data_root.is_absolute():
        data_root = workspace / data_root

    roles: dict[str, list[Path]] = {}
    assigned: dict[str, list[str]] = {}
    resolved_by_key: dict[str, Path] = {}
    for role, config_key in DATA_ROLES:
        names = data.get(config_key)
        if not isinstance(names, list) or not names:
            raise ManifestError(
                f"experiment config data.{config_key} must be a non-empty list"
            )
        role_paths: list[Path] = []
        role_keys: set[str] = set()
        for index, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                raise ManifestError(
                    f"experiment config data.{config_key}[{index}] must be a non-empty path"
                )
            candidate = Path(name).expanduser()
            if not candidate.is_absolute():
                candidate = data_root / candidate
            resolved = _resolve_file(candidate, f"configured {role} data file")
            normalized = os.path.normcase(str(resolved))
            if normalized in role_keys:
                raise ManifestError(
                    f"configured data file is assigned more than once "
                    f"within {role}: {resolved}"
                )
            role_keys.add(normalized)
            assigned.setdefault(normalized, []).append(role)
            resolved_by_key[normalized] = resolved
            role_paths.append(resolved)
        roles[role] = role_paths

    for normalized, participating_roles in assigned.items():
        if len(participating_roles) == 1:
            continue
        resolved = resolved_by_key[normalized]
        data_format = data.get("format", "csv")
        if not isinstance(data_format, str) or data_format.lower() not in {"csv", "npy"}:
            raise ManifestError(
                "shared data files require data.format to be 'csv' or 'npy': "
                f"{resolved}"
            )
        format_label = data_format.upper()

        configured_slices = data.get("role_slices")
        if not isinstance(configured_slices, dict):
            raise ManifestError(
                "configured data file is assigned more than once; "
                f"shared {format_label} input requires explicit data.role_slices for "
                f"all participating roles ({', '.join(participating_roles)}): {resolved}"
            )
        intervals: dict[str, tuple[int, int]] = {}
        for role in participating_roles:
            requested = configured_slices.get(role)
            if (
                not isinstance(requested, list)
                or len(requested) != 2
                or type(requested[0]) is not int
                or type(requested[1]) is not int
                or requested[0] < 0
                or requested[1] <= requested[0]
            ):
                raise ManifestError(
                    f"shared {format_label} data file requires an explicit valid [start, end] "
                    f"data.role_slices.{role}: {resolved}"
                )
            intervals[role] = (requested[0], requested[1])

        for index, left_role in enumerate(participating_roles):
            left_start, left_end = intervals[left_role]
            for right_role in participating_roles[index + 1 :]:
                right_start, right_end = intervals[right_role]
                if left_end > right_start and right_end > left_start:
                    raise ManifestError(
                        f"shared {format_label} data file has overlapping role slices "
                        f"({left_role}=[{left_start}, {left_end}], "
                        f"{right_role}=[{right_start}, {right_end}]): {resolved}"
                    )
    return roles


def _validate_requested_data_files(
    data_files: Sequence[Path],
    configured: Mapping[str, Sequence[Path]],
    workspace: Path,
) -> list[Path]:
    if not data_files:
        raise ManifestError("at least one --data file is required")
    requested = _unique_sorted_files(data_files, "data file")
    expected = [path for role, _ in DATA_ROLES for path in configured[role]]
    expected_by_key = {os.path.normcase(str(path)): path for path in expected}
    requested_by_key = {os.path.normcase(str(path)): path for path in requested}
    missing_keys = expected_by_key.keys() - requested_by_key.keys()
    extra_keys = requested_by_key.keys() - expected_by_key.keys()
    missing = sorted(
        _display_path(expected_by_key[key], workspace) for key in missing_keys
    )
    extra = sorted(
        _display_path(requested_by_key[key], workspace) for key in extra_keys
    )
    if missing or extra:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"extra={extra}")
        raise ManifestError(
            "--data files must exactly match config train/validation/test files; "
            + "; ".join(detail)
        )
    return requested


def _data_records(
    configured: Mapping[str, Sequence[Path]],
    workspace: Path,
) -> dict[str, list[dict[str, object]]]:
    return {
        role: [_file_record(path, workspace) for path in configured[role]]
        for role, _ in DATA_ROLES
    }


def _implementation_records(
    workspace: Path,
) -> tuple[dict[str, dict[str, object]], list[Path]]:
    records: dict[str, dict[str, object]] = {}
    paths: list[Path] = []
    for role, relative_path in sorted(IMPLEMENTATION_FILES.items()):
        path = _resolve_file(workspace / relative_path, role)
        paths.append(path)
        records[role] = _file_record(path, workspace)
    return records, paths


def _upstream_records(
    official_pins: Sequence[UpstreamPin],
    workspace: Path,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for pin in sorted(official_pins, key=lambda item: item.name):
        if pin.name in records:
            raise ManifestError(f"duplicate upstream pin name: {pin.name}")
        records[pin.name] = _upstream_record(pin, workspace)
    return records


def build_manifest(
    *,
    workspace: Path,
    config: Path,
    output: Path,
    data_files: Sequence[Path],
    timestamp: str | None = None,
    official_pins: Sequence[UpstreamPin] = OFFICIAL_PINS,
) -> dict[str, object]:
    """Validate all freeze inputs and return the complete manifest object."""
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ManifestError(f"workspace is not a directory: {workspace}")
    config_file = _resolve_file(config, "config")
    configured_data = _configured_data_files(config_file, workspace)
    requested_data = _validate_requested_data_files(
        data_files, configured_data, workspace
    )
    output_file = output.expanduser().resolve(strict=False)

    implementation, implementation_paths = _implementation_records(workspace)

    protected = {os.path.normcase(str(config_file))}
    protected.update(os.path.normcase(str(path)) for path in requested_data)
    protected.update(os.path.normcase(str(path)) for path in implementation_paths)
    if os.path.normcase(str(output_file)) in protected:
        raise ManifestError(
            "manifest output must not overwrite the config, data, or implementation artifacts"
        )

    pins = _upstream_records(official_pins, workspace)

    captured_at_utc = _normalize_timestamp(timestamp)
    config_record = _file_record(config_file, workspace)
    workspace_git = _git_record(workspace, workspace)
    runtime = _runtime_record(captured_at_utc)
    # Data content is intentionally the last input class hashed.  Invalid
    # config/code/pin/runtime state therefore fails before touching a holdout.
    data_records = _data_records(configured_data, workspace)
    return {
        "artifacts": {
            "config": config_record,
            "data": data_records,
            "implementation": implementation,
        },
        "canonicalization": CANONICALIZATION,
        "git": workspace_git,
        "hash_algorithm": HASH_ALGORITHM,
        "manifest_output": _display_path(output_file, workspace),
        "runtime": runtime,
        "schema": SCHEMA,
        "upstreams": pins,
    }


def canonical_json(manifest: Mapping[str, object]) -> bytes:
    """Serialize without host-dependent whitespace or dictionary ordering."""
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _parse_manifest_bytes(payload: bytes, manifest_path: Path) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
        document = json.loads(
            text,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ManifestError(f"cannot parse manifest {manifest_path}: {error}") from error
    if not isinstance(document, dict):
        raise ManifestError("manifest root must be a JSON object")
    try:
        canonical = canonical_json(document)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"manifest is not canonicalizable: {error}") from error
    if payload != canonical:
        raise ManifestError("manifest bytes are not canonical JSON")
    return document


def _validate_expected_digest(payload: bytes, expected_sha256: str | None) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is None:
        return digest
    normalized = expected_sha256.strip().lower()
    try:
        valid_hex = len(normalized) == 64 and int(normalized, 16) >= 0
    except ValueError:
        valid_hex = False
    if not valid_hex:
        raise ManifestError("expected manifest SHA-256 must be 64 hexadecimal characters")
    if digest != normalized:
        raise ManifestError(
            f"manifest SHA-256 mismatch: observed {digest}, expected {normalized}"
        )
    return digest


def _require_record(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ManifestError(f"manifest {label} does not match the current frozen state")


def _validate_runtime_shape(runtime: object) -> None:
    if not isinstance(runtime, dict):
        raise ManifestError("manifest runtime must be an object")
    expected_fields = {
        "captured_at_utc",
        "local_timezone",
        "local_utc_offset_seconds",
        "platform",
        "python",
        "python_packages",
    }
    if set(runtime) != expected_fields:
        raise ManifestError("manifest runtime fields are invalid")
    captured = runtime.get("captured_at_utc")
    if not isinstance(captured, str) or _normalize_timestamp(captured) != captured:
        raise ManifestError("manifest runtime timestamp is not canonical UTC")
    if not isinstance(runtime.get("platform"), dict):
        raise ManifestError("manifest runtime platform must be an object")
    if not isinstance(runtime.get("python"), dict):
        raise ManifestError("manifest runtime python must be an object")
    packages = runtime.get("python_packages")
    if not isinstance(packages, dict) or set(packages) != set(RUNTIME_PACKAGES):
        raise ManifestError("manifest runtime package set is invalid")


def _validate_manifest_output(value: object, workspace: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError("manifest_output must be a non-empty path")
    declared = Path(value).expanduser()
    if not declared.is_absolute():
        declared = workspace / declared
    declared = declared.resolve(strict=False)
    if _display_path(declared, workspace) != value:
        raise ManifestError("manifest_output is not a canonical path declaration")
    return declared


def verify_manifest(
    manifest_path: Path,
    workspace: Path,
    config_path: Path,
    *,
    official_pins: Sequence[UpstreamPin] = OFFICIAL_PINS,
    expected_sha256: str | None = None,
) -> VerifiedManifest:
    """Verify a frozen manifest in one read and return its bytes and digest.

    When supplied, ``expected_sha256`` must be the externally committed digest.
    Data bytes are deliberately hashed only after the manifest, config,
    implementation, Git state, and upstream records have passed validation.
    """
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ManifestError(f"workspace is not a directory: {workspace}")
    manifest_file = _resolve_file(manifest_path, "manifest")
    try:
        payload = manifest_file.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read manifest {manifest_file}: {error}") from error
    digest = _validate_expected_digest(payload, expected_sha256)
    manifest = _parse_manifest_bytes(payload, manifest_file)

    if set(manifest) != TOP_LEVEL_FIELDS:
        raise ManifestError("manifest top-level fields are invalid")
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(f"manifest schema must be {SCHEMA}")
    if manifest.get("hash_algorithm") != HASH_ALGORITHM:
        raise ManifestError(f"manifest hash_algorithm must be {HASH_ALGORITHM}")
    if manifest.get("canonicalization") != CANONICALIZATION:
        raise ManifestError("manifest canonicalization declaration is invalid")
    declared_output = _validate_manifest_output(
        manifest.get("manifest_output"), workspace
    )
    _validate_runtime_shape(manifest.get("runtime"))

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "config", "data", "implementation"
    }:
        raise ManifestError("manifest artifacts fields are invalid")
    config_file = _resolve_file(config_path, "config")
    _require_record(
        artifacts.get("config"),
        _file_record(config_file, workspace),
        "config artifact",
    )

    implementation, implementation_paths = _implementation_records(workspace)
    _require_record(
        artifacts.get("implementation"),
        implementation,
        "implementation artifacts",
    )
    _require_record(
        manifest.get("git"),
        _git_record(workspace, workspace),
        "workspace Git record",
    )
    _require_record(
        manifest.get("upstreams"),
        _upstream_records(official_pins, workspace),
        "upstream records",
    )

    configured_data = _configured_data_files(config_file, workspace)
    protected = {os.path.normcase(str(config_file))}
    protected.update(
        os.path.normcase(str(path))
        for role, _ in DATA_ROLES
        for path in configured_data[role]
    )
    protected.update(os.path.normcase(str(path)) for path in implementation_paths)
    if os.path.normcase(str(declared_output)) in protected:
        raise ManifestError(
            "manifest_output declares a config, data, or implementation artifact"
        )
    _require_record(
        artifacts.get("data"),
        _data_records(configured_data, workspace),
        "role-bound data artifacts",
    )
    role_paths = tuple(
        (role, tuple(configured_data[role]))
        for role, _ in DATA_ROLES
    )
    return VerifiedManifest(
        document=manifest,
        raw_bytes=payload,
        sha256=digest,
        role_paths=role_paths,
    )


def write_manifest(output: Path, payload: bytes) -> None:
    output = output.expanduser().resolve(strict=False)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ManifestError(f"cannot create manifest directory {output.parent}: {error}") from error
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except OSError as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise ManifestError(f"cannot atomically write manifest {output}: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash and freeze a QPET-correlation experiment without running it."
    )
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML file")
    parser.add_argument("--output", required=True, type=Path, help="manifest JSON destination")
    parser.add_argument(
        "--data",
        required=True,
        action="append",
        type=Path,
        metavar="FILE",
        help="data file to hash; repeat once per file",
    )
    parser.add_argument(
        "--timestamp",
        help="optional ISO-8601 time override for a byte-reproducible manifest test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace = Path(__file__).resolve().parents[2]
    try:
        manifest = build_manifest(
            workspace=workspace,
            config=arguments.config,
            output=arguments.output,
            data_files=arguments.data,
            timestamp=arguments.timestamp,
        )
        payload = canonical_json(manifest)
        write_manifest(arguments.output, payload)
    except ManifestError as error:
        print(f"freeze_manifest: {error}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(payload).hexdigest()
    print(f"wrote {Path(arguments.output).resolve()} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
