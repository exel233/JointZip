"""Frozen, test-independent protocol constants.

The benchmark is deliberately two-stage.  ``freeze`` may inspect only the raw
training and validation windows and records the selected forecasting
hyperparameters.  ``evaluate`` requires that record and may then decode the
test archives.  This prevents test-utility results from changing the task
definition or model selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL: dict[str, Any] = {
    "schema": "nclab.vldb-utility-protocol.v1",
    "seed": 42,
    "dataset": {
        "config": "configs/development_stream_closure_tesla_qpet_matched32.yaml",
        "qpet_result_dir": (
            "experiments/results/development_stream_closure_tesla_qpet_matched32"
        ),
        "qpet_spec_index": 1,
        "qpet_artifacts": [
            "bitstreams/01_00_qpet_correlation.scs",
            "bitstreams/01_01_qpet_correlation.scs",
            "bitstreams/01_02_qpet_correlation.scs",
        ],
        "expected_shape": [32, 16, 128],
    },
    "distortion": {
        "space": "training-standardized-float32",
        "max_lag": 16,
        "acf_definition": "Pearson(x[0:T-lag],x[lag:T])",
    },
    "forecast": {
        "kind": "channel-wise-direct-multistep-ridge",
        "horizon": 16,
        "orders": [4, 8, 16, 24],
        "ridge_alphas": [1e-6, 1e-4, 1e-2, 1.0],
        "selection": "minimum raw-validation NRMSE; ties by order then alpha",
        "test_target": "raw [window_end:window_end+horizon], outside encoded window",
    },
    "retrieval": {
        "query": "raw-test-window",
        "archive": "reconstructed-test-window",
        "normalization": "per-window-per-channel z-normalization",
        "distance": "squared Euclidean",
        "k": 5,
        "exclude_self": True,
    },
    "dynamics": {
        "acf_lags": 16,
        "ar_fit": "channel-wise archive ridge, no cross-window transitions",
        "ar_order_and_alpha": "reuse frozen forecast selection",
        "claim_scope": "lag-Pearson ACF signature and AR coefficients; not PSD",
    },
    "codecs": {
        "qpet": "persisted matched32 SCS2/NCS/QPC outer-decoder replay",
        "baselines": [
            {"name": "sperr", "epsilons": [0.5, 0.25, 0.1]},
            {"name": "sz2016", "epsilons": [0.5, 0.25, 0.1]},
        ],
        "baseline_collection_frontier": "stream_closure_v1",
    },
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
    """Create an immutable canonical freeze record or verify an existing one."""
    payload = canonical_json_bytes(record) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = path.read_bytes()
        if observed != payload:
            raise RuntimeError(
                f"freeze record already exists with different content: {path}"
            )
        return digest
    path.write_bytes(payload)
    return digest


def load_freeze(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid freeze record: {path}") from exc
    if not isinstance(record, dict) or canonical_json_bytes(record) + b"\n" != payload:
        raise ValueError(f"freeze record is not canonical JSON: {path}")
    if record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("freeze record belongs to a different utility protocol")
    return record, hashlib.sha256(payload).hexdigest()
