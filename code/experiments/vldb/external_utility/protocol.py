"""Frozen protocol for the fast prospective UCR classification panel.

The panel is selected without using any classification-error column.  It is a
small, explicitly labelled quick panel intended to produce external evidence
quickly; it is not a substitute for the eventual all-archive evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SELECTION_SEED = 20260902
QPET_ACCEPTANCE_ABS_TOLERANCE = 1e-7
BASELINE_POINT_ABS_TOLERANCE = 1e-5
UCR_ROOT_URL = "https://timeseriesclassification.com/aeon-toolkit"
UCR_METADATA_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
)

EXPECTED_PANEL = (
    "ECG200",
    "ArrowHead",
    "GunPoint",
    "Earthquakes",
    "ShapeletSim",
    "Meat",
)

QPET_SPEC: dict[str, Any] = {
    "name": "qpet_correlation",
    "collection_frontier": "stream_closure_v1",
    "candidate_profile": "generic",
    "repair_optimizer": "ssbb",
    "correction_format_ceiling": 10,
    "base_frontier": "incumbent_only",
    "relaxed_bound_ratio_ceiling": "none",
    "surrogate_frontier": "none",
    "cfsr_bound_ratio_ceiling": "none",
    "max_lag": 16,
    "qoz_level": 1,
    "correction_mode": "posthoc_greedy",
    "estimator_scales": [1.75, 2.0, 2.25, 2.5],
    "qpet_confidence": 0.999,
    "qpet_k": 2.0,
    "group_size": 1,
    "repair_denominators": [7, 15, 31, 63, 127],
}

# The stress point is deliberately fixed before any external compressed-data
# task score is computed.  Error is in train-standardized units.  The
# point-only ablation keeps the identical QPET stack and makes the ACF bound
# mathematically vacuous because two correlations in [-1, 1] differ by at most
# two; Pearson is disabled.
QPET_POINTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "joint",
        "label": "joint-e1-acf0.05-r0.99",
        "epsilon": 1.0,
        "acf_epsilon": 0.05,
        "min_correlation": 0.99,
    },
    {
        "variant": "point_only",
        "label": "point-only-e1",
        "epsilon": 1.0,
        "acf_epsilon": 2.0,
        "min_correlation": -1.0,
    },
)

BASELINES: tuple[dict[str, Any], ...] = (
    {"name": "sperr", "epsilon": 1.0},
    {"name": "sz2016", "epsilon": 1.0},
)

PROTOCOL: dict[str, Any] = {
    "schema": "nclab.ucr-quick-panel-protocol.v5",
    "evidence_tier": "prospective-external-quick-panel",
    "selection": {
        "seed": SELECTION_SEED,
        "metadata_url": UCR_METADATA_URL,
        "allowed_columns": ["ID", "Type", "Name", "Train", "Test", "Class", "Length"],
        "eligible": {
            "legacy_id_max": 85,
            "train_min": 20,
            "test_min": 50,
            "test_max": 200,
            "length_min": 64,
            "length_max": 512,
            "test_values_max": 100000,
        },
        "rule": (
            "group eligible rows by Type; select the lexicographically smallest "
            "SHA256('20260902:<Type>:<Name>') in each group"
        ),
        "expected_panel": list(EXPECTED_PANEL),
        "explicitly_forbidden_columns": [
            "ED (w=0)",
            "DTW (learned_w)",
            "DTW (w=100)",
            "Default rate",
        ],
    },
    "data": {
        "archive_root_url": UCR_ROOT_URL,
        "split": "official TRAIN/TEST",
        "encoding": "float32",
        "normalization": "single train-only scalar mean/std per dataset; stored in NOR2",
        "encoded_partition": "TEST only",
    },
    "task": {
        "kind": "one-nearest-neighbor classification",
        "train_archive": "raw official TRAIN",
        "queries": "raw or reconstructed official TEST",
        "shape_normalization": "per-series centered unit-L2 normalization",
        "distance": "Euclidean",
        "tie_break": "first official TRAIN row",
        "secondary": [
            "prediction agreement",
            "nearest-neighbor identity retention",
            "top-5 set recall",
            "NDCG@5",
            "Pearson margin-certificate coverage",
        ],
    },
    "distortion": {
        "space": "train-standardized float32",
        "max_lag": 16,
        "acf_definition": "Pearson(x[0:T-lag],x[lag:T])",
        "qpet_acceptance_absolute_tolerance": QPET_ACCEPTANCE_ABS_TOLERANCE,
        "official_baseline_point_absolute_tolerance": BASELINE_POINT_ABS_TOLERANCE,
    },
    "codecs": {
        "qpet_spec": QPET_SPEC,
        "qpet_points": list(QPET_POINTS),
        "official_baselines": list(BASELINES),
        "accounting": "sum of complete reread SCS2 bytes; raw TEST float32 bytes",
    },
    "runtime_identity": {
        "schema": "wsl-native-runtime.v2",
        "ldd_normalization": "strip ASLR load addresses; sort normalized dependency lines",
        "native_identity": "hash executable bytes plus every resolved dynamic-library byte stream",
    },
    "execution": {
        "dataset_workers": 1,
        "qpet_block_workers": 6,
        "parallelism_unit": "QPET payload blocks within one frozen-order dataset",
        "qpet_rows_per_payload": 32,
        "qpet_partition": (
            "contiguous official TEST row order within each dataset; "
            "never cross a dataset; final block may contain fewer than 32 rows"
        ),
        "within_dataset_order": [
            "qpet_joint",
            "qpet_point_only",
            "sperr",
            "sz2016",
        ],
        "result_order": "frozen dataset order, then declared within-dataset order",
    },
    "claim_scope": (
        "same-nominal-L-infinity stress comparison on a fast prospective mechanism "
        "panel; no same-rate claim, all-UCR generalization, or statistical significance claim"
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(PROTOCOL)


def write_or_verify_freeze(path: Path, record: dict[str, Any]) -> str:
    payload = canonical_json_bytes(record) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"freeze already exists with different bytes: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def load_freeze(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid freeze record: {path}") from error
    if not isinstance(record, dict) or canonical_json_bytes(record) + b"\n" != payload:
        raise ValueError(f"freeze is not canonical JSON: {path}")
    if record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("freeze belongs to a different UCR protocol")
    return record, hashlib.sha256(payload).hexdigest()
