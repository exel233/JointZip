"""EDBT-source data preparation and frozen, codec-independent utility tasks.

Inputs are train-standardized float32. Within either evaluation split, selected
input supports cannot overlap any selected future. Preceding split history may
provide context at a split boundary, as in the official forecasting loaders.
Every Solar variate is a separate bundle/artifact with its own billed NOR2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "tmp/upstreams/EvalImpLSTS-audit/data/raw"
UPSTREAM_COMMIT = "61392420bb39f42eaa3a53f308e1e454ae8e39d6"
WINDOW = 128
CONTEXT = 96
HORIZON = 24
DATASETS = ("ETTm1", "ETTm2", "Solar", "Weather", "ElecDem", "Wind")
SPECS = {
    "ETTm1": ("ETTm1/ETTm1.parquet", "datetime", "OT", 69680, 900, 96),
    "ETTm2": ("ETTm2/ETTm2.parquet", "datetime", "OT", 69680, 900, 96),
    "Solar": ("Solar/solar.parquet", "Date_Time", None, 52560, 600, 144),
    "Weather": ("Weather/weather.parquet", "datetime", "OT", 52704, 600, 144),
    "ElecDem": ("AUSElecDem/aus_electrical_demand.parquet", "datetime", "y", 230736, 1800, 48),
    "Wind": ("Wind/wind.parquet", "datetime", "active power", 432000, 2, 43200),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Canonical little-endian float32 data identity; shape is stored separately."""
    return hashlib.sha256(np.asarray(value, dtype="<f4", order="C").tobytes()).hexdigest()


def split_boundaries(name: str, length: int) -> dict[str, list[int]]:
    if name not in SPECS or length < 1:
        raise ValueError("unknown dataset or invalid length")
    if name in {"ETTm1", "ETTm2"}:
        train_end, val_end, test_end = 34560, 46080, 57600
        if length < test_end:
            raise ValueError("ETT artifact is shorter than its official fixed split")
    else:
        train_end = int(length * 0.7)
        test_length = int(length * 0.2)
        val_end, test_end = length - test_length, length
    return {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, test_end]}


def select_starts(
    target_begin: int,
    target_end: int,
    count: int,
    *,
    window: int = WINDOW,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Select length-only, ordered origins including the first/last usable one.

    Targets lie wholly in [target_begin, target_end). Context may use preceding
    history, as in the official loaders. A stride of at least window+horizon
    ensures no chosen label occurs in any chosen encoder input in this split.
    """
    if count < 1 or window < 1 or horizon < 1 or target_end < target_begin:
        raise ValueError("invalid window selection arguments")
    first_origin = max(target_begin, window)
    last_origin = target_end - horizon
    if last_origin < first_origin:
        return np.empty(0, dtype=np.int64)
    available = (last_origin - first_origin) // (window + horizon) + 1
    chosen = min(int(count), available)
    if chosen == 1:
        origins = np.asarray([first_origin], dtype=np.int64)
    else:
        offsets = np.arange(chosen, dtype=np.int64) * (last_origin - first_origin) // (chosen - 1)
        origins = first_origin + offsets
    return origins - window


def _windows(series: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if starts.size == 0:
        raise ValueError("split has no eligible context/future pair")
    source = np.stack([series[s:s + WINDOW] for s in starts])[:, None, :]
    future = np.stack([series[s + WINDOW:s + WINDOW + HORIZON] for s in starts])
    return np.ascontiguousarray(source, dtype=np.float32), np.ascontiguousarray(future, dtype=np.float32)


def prepare_dataset(
    name: str,
    *,
    data_root: str | Path | None = None,
    validation_windows: int = 8,
    test_windows: int = 32,
) -> list[dict[str, Any]]:
    """Prepare one bundle per declared variate; no value-dependent selection.

    ``train_1d`` is standardized; ``train_raw_1d`` is physical float32.
    ``starts`` are absolute raw-file row positions, not split-relative indices.
    Mean/std are float32-rounded TRAIN-only statistics computed in float64.
    The caller must persist each bundle's scalar mean/std in its artifact NOR2.
    """
    import pandas as pd  # parquet engine is only required for actual preparation

    aliases = {entry.lower(): entry for entry in DATASETS}
    aliases.update({"auselecdem": "ElecDem", "aus": "ElecDem"})
    if name.lower() not in aliases:
        raise ValueError(f"unknown dataset: {name}")
    name = aliases[name.lower()]
    relative, time_column, target, expected_length, frequency, seasonality = SPECS[name]
    path = (Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT) / relative
    digest = file_sha256(path)
    frame = pd.read_parquet(path)
    if len(frame) != expected_length or time_column not in frame:
        raise ValueError(f"unexpected official artifact structure: {name}, shape={frame.shape}")
    if frame[time_column].isna().any():
        raise ValueError(f"missing timestamps: {name}")
    channels = [column for column in frame if column != time_column] if target is None else [target]
    if name == "Solar" and len(channels) != 137:
        raise ValueError("Solar must retain all 137 declared variates")
    if any(column not in frame for column in channels):
        raise ValueError(f"missing declared target in {name}")
    boundaries = split_boundaries(name, len(frame))
    train_end = boundaries["train"][1]
    val_starts = select_starts(*boundaries["validation"], validation_windows)
    starts = select_starts(*boundaries["test"], test_windows)
    bundles: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        physical = frame[channel].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(physical).all():
            raise ValueError(f"nonfinite values in {name}/{channel}; no implicit imputation")
        train_raw = physical[:train_end].copy()
        mean = np.float32(np.mean(train_raw, dtype=np.float64))
        observed_std = np.float32(np.std(train_raw, dtype=np.float64))
        std = np.float32(1.0) if observed_std < 1e-6 else observed_std
        standardized = ((physical - mean) / std).astype(np.float32)
        if not np.isfinite(standardized).all():
            raise ValueError(f"nonfinite normalized input in {name}/{channel}")
        source, future = _windows(standardized, starts)
        val_source, val_future = _windows(standardized, val_starts)
        series_id = f"{name}__{index:03d}"
        bundles.append({
            "dataset": name,
            "series_id": series_id,
            "series_index": index,
            "channel_name": str(channel),
            "source": source,
            "future": future,
            "val_source": val_source,
            "val_future": val_future,
            "starts": starts.copy(),
            "val_starts": val_starts.copy(),
            "train_1d": standardized[:train_end].copy(),
            "train_raw_1d": train_raw,
            "mean": float(mean),
            "std": float(std),
            "observed_train_std": float(observed_std),
            "source_file": str(path.resolve()),
            "source_file_sha256": digest,
            "physical_source_sha256": array_sha256(physical),
            "train_standardized_sha256": array_sha256(standardized[:train_end]),
            "source_standardized_sha256": array_sha256(source),
            "future_standardized_sha256": array_sha256(future),
            "val_source_standardized_sha256": array_sha256(val_source),
            "val_future_standardized_sha256": array_sha256(val_future),
            "split_boundaries": boundaries,
            "raw_length": len(physical),
            "sampling_interval_seconds": frequency,
            "seasonal_period": seasonality,
            "window": WINDOW,
            "forecast_context": CONTEXT,
            "horizon": HORIZON,
            "upstream_commit": UPSTREAM_COMMIT,
            "selection": "length-only equispaced origins; disjoint 128+24 supports within split",
            "normalization": "float64 TRAIN population mean/std rounded to float32; float32 transform; std<1e-6 replaced by 1",
        })
    if digest != file_sha256(path):
        raise RuntimeError(f"source changed while preparing: {path}")
    return bundles


def _one_dimensional(value: np.ndarray, label: str) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    if data.ndim != 1 or data.size < 1 or not np.isfinite(data).all():
        raise ValueError(f"{label} must be nonempty finite 1D")
    return data


def _batch(value: np.ndarray, label: str) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    if data.ndim == 3 and data.shape[1] == 1:
        data = data[:, 0, :]
    if data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError(f"{label} must be finite nonempty [B,T] or [B,1,T]")
    return data


def _ridge_statistics(train: np.ndarray, order: int, horizon: int) -> tuple[np.ndarray, np.ndarray, int]:
    data = _one_dimensional(train, "train")
    count = len(data) - order - horizon + 1
    if order < 1 or order > CONTEXT or horizon < 1 or count < 1:
        raise ValueError("invalid ridge order/horizon or insufficient training data")
    windows = np.lib.stride_tricks.sliding_window_view(data, order + horizon)
    gram = np.zeros((order + 1, order + 1), dtype=np.float64)
    cross = np.zeros((order + 1, horizon), dtype=np.float64)
    for begin in range(0, count, 8192):
        chunk = windows[begin:begin + 8192]
        design = np.empty((len(chunk), order + 1), dtype=np.float64)
        design[:, :order] = chunk[:, :order]
        design[:, order] = 1.0
        targets = np.ascontiguousarray(chunk[:, order:])
        gram += design.T @ design
        cross += design.T @ targets
    return gram, cross, count


def _solve_ridge(gram: np.ndarray, cross: np.ndarray, order: int, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    regularized = gram.copy()
    regularized[np.arange(order), np.arange(order)] += alpha
    return np.linalg.solve(regularized, cross)


def fit_ridge(train_1d: np.ndarray, *, order: int = 16, alpha: float = 1.0, horizon: int = HORIZON) -> dict[str, Any]:
    gram, cross, count = _ridge_statistics(train_1d, order, horizon)
    weights = _solve_ridge(gram, cross, order, alpha)
    return {"model": "direct-ridge", "order": order, "p": order, "alpha": float(alpha), "horizon": horizon,
            "coefficients": weights[:-1].tolist(), "weights": weights[:-1].tolist(),
            "intercept": weights[-1].tolist(), "training_pairs": count,
            "fit_policy": "all contiguous TRAIN pairs, sum squared loss, intercept unpenalized"}


def predict_ridge(model: dict[str, Any], source: np.ndarray) -> np.ndarray:
    batch = _batch(source, "forecast source")
    order = int(model["order"])
    if batch.shape[1] < order or order > CONTEXT:
        raise ValueError("insufficient allowed forecast context")
    prediction = batch[:, -order:] @ np.asarray(model["coefficients"]) + np.asarray(model["intercept"])
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite ridge prediction")
    return prediction


def predict_persistence(source: np.ndarray, *, horizon: int = HORIZON) -> np.ndarray:
    """Repeat the final observed context value; no fitting or future access."""
    batch = _batch(source, "persistence source")
    if horizon < 1:
        raise ValueError("persistence horizon must be positive")
    return np.repeat(batch[:, -1:], horizon, axis=1)


def select_ridge(
    train_1d: np.ndarray,
    val_source: np.ndarray,
    val_future: np.ndarray,
    *,
    orders: Iterable[int] = (4, 16, 32),
    alphas: Iterable[float] = (0.01, 1.0, 100.0),
) -> dict[str, Any]:
    """Select by raw-validation MSE only; never accepts test data.

    Ties use the declared order grid then alpha grid, without tolerance-based
    winner substitution. Returns the model and all validation candidates.
    """
    target = _batch(val_future, "validation future")
    validation = _batch(val_source, "validation source")
    if validation.shape[0] != target.shape[0]:
        raise ValueError("validation batch sizes differ")
    order_grid, alpha_grid = tuple(orders), tuple(alphas)
    if not order_grid or not alpha_grid:
        raise ValueError("empty validation grid")
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for order in order_grid:
        gram, cross, count = _ridge_statistics(train_1d, int(order), target.shape[1])
        for alpha in alpha_grid:
            weights = _solve_ridge(gram, cross, int(order), float(alpha))
            model = {"model": "direct-ridge", "order": int(order), "p": int(order), "alpha": float(alpha),
                     "horizon": target.shape[1], "coefficients": weights[:-1].tolist(),
                     "weights": weights[:-1].tolist(), "intercept": weights[-1].tolist(),
                     "training_pairs": count,
                     "fit_policy": "all contiguous TRAIN pairs, sum squared loss, intercept unpenalized"}
            mse = float(np.mean((predict_ridge(model, validation) - target) ** 2))
            candidates.append({"order": int(order), "alpha": float(alpha), "validation_mse": mse})
            if best is None or mse < best["validation_mse"]:
                best = {**model, "validation_mse": mse, "validation_loss": mse}
    assert best is not None
    best["validation_candidates"] = candidates
    best["selection_rule"] = "minimum raw-validation MSE; grid order breaks exact ties"
    return best


def forecast_metrics(
    future: np.ndarray,
    raw_prediction: np.ndarray,
    reconstructed_prediction: np.ndarray,
    train_1d: np.ndarray,
    *,
    seasonal_period: int = 1,
) -> dict[str, Any]:
    labels = _batch(future, "future")
    raw = _batch(raw_prediction, "raw prediction")
    decoded = _batch(reconstructed_prediction, "reconstructed prediction")
    if labels.shape != raw.shape or labels.shape != decoded.shape:
        raise ValueError("future and predictions must have identical shape")
    training = _one_dimensional(train_1d, "train")
    if seasonal_period < 1 or seasonal_period >= training.size:
        raise ValueError("invalid seasonal scale period")
    train_range = float(np.ptp(training))
    seasonal_diff = training[seasonal_period:] - training[:-seasonal_period]
    mase_scale = float(np.mean(np.abs(seasonal_diff)))
    rmsse_scale = float(np.mean(seasonal_diff ** 2))
    raw_errors, reconstructed_errors = raw - labels, decoded - labels
    raw_rmse = float(np.sqrt(np.mean(raw_errors ** 2)))
    reconstructed_rmse = float(np.sqrt(np.mean(reconstructed_errors ** 2)))
    raw_mae = float(np.mean(np.abs(raw_errors)))
    reconstructed_mae = float(np.mean(np.abs(reconstructed_errors)))
    raw_centered = raw - raw.mean(axis=1, keepdims=True)
    decoded_centered = decoded - decoded.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(raw_centered, axis=1) * np.linalg.norm(decoded_centered, axis=1)
    same_prediction = np.all(raw == decoded, axis=1)
    correlations = np.divide(np.sum(raw_centered*decoded_centered, axis=1), denominator,
                             out=same_prediction.astype(np.float64), where=denominator > 1e-15)
    safe_ratio = lambda value, denominator: float(value / denominator) if denominator > 1e-15 else None
    return {
        "forecast_raw_mse": float(np.mean(raw_errors ** 2)),
        "forecast_reconstructed_mse": float(np.mean(reconstructed_errors ** 2)),
        "forecast_excess_mse": float(np.mean(reconstructed_errors ** 2) - np.mean(raw_errors ** 2)),
        "forecast_raw_rmse": raw_rmse,
        "forecast_reconstructed_rmse": reconstructed_rmse,
        "forecast_excess_rmse": reconstructed_rmse - raw_rmse,
        "forecast_relative_excess_rmse": safe_ratio(reconstructed_rmse - raw_rmse, raw_rmse),
        "forecast_raw_mae": raw_mae,
        "forecast_reconstructed_mae": reconstructed_mae,
        "forecast_excess_mae": reconstructed_mae - raw_mae,
        "forecast_prediction_correlation_mean": float(np.mean(correlations)),
        "forecast_prediction_correlation_min": float(np.min(correlations)),
        "forecast_prediction_correlation_policy": "per-origin horizon Pearson; degenerate pair=1 only for identical predictions, else 0",
        "forecast_raw_nrmse": safe_ratio(raw_rmse, train_range),
        "forecast_reconstructed_nrmse": safe_ratio(reconstructed_rmse, train_range),
        "forecast_excess_nrmse": safe_ratio(reconstructed_rmse - raw_rmse, train_range),
        "forecast_raw_mase": safe_ratio(raw_mae, mase_scale),
        "forecast_reconstructed_mase": safe_ratio(reconstructed_mae, mase_scale),
        "forecast_raw_rmsse": safe_ratio(raw_rmse, np.sqrt(rmsse_scale)),
        "forecast_reconstructed_rmsse": safe_ratio(reconstructed_rmse, np.sqrt(rmsse_scale)),
        "forecast_prediction_rmse": float(np.sqrt(np.mean((decoded - raw) ** 2))),
        "forecast_per_origin_raw_rmse": np.sqrt(np.mean(raw_errors ** 2, axis=1)).tolist(),
        "forecast_per_origin_reconstructed_rmse": np.sqrt(np.mean(reconstructed_errors ** 2, axis=1)).tolist(),
        "forecast_scale_policy": "NRMSE: raw TRAIN range; MASE/RMSSE: raw TRAIN seasonal differences; zero denominator -> null",
        "forecast_train_range": train_range,
        "forecast_mase_scale": mase_scale,
        "forecast_rmsse_squared_scale": rmsse_scale,
        "forecast_seasonal_period": seasonal_period,
        "forecast_origins": labels.shape[0],
        "forecast_horizon": labels.shape[1],
    }


def train_prototypes(train_1d: np.ndarray, *, count: int = 256, window: int = WINDOW) -> np.ndarray:
    data = _one_dimensional(train_1d, "train prototypes")
    if count < 1 or window < 1 or data.size < window:
        raise ValueError("insufficient TRAIN or invalid prototype parameters")
    number = min(count, data.size // window)
    starts = np.asarray([0], dtype=np.int64) if number == 1 else (
        np.arange(number, dtype=np.int64) * (data.size - window) // (number - 1)
    )
    return np.asarray([data[start:start + window] for start in starts], dtype=np.float32)


def _shape_normalize(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = batch - batch.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    constants = norms[:, 0] <= 1e-15
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-15), constants


def shape_retrieval_metrics(raw: np.ndarray, reconstructed: np.ndarray, prototypes: np.ndarray, *, k: int = 5) -> dict[str, Any]:
    """Compressed queries against frozen raw TRAIN whole-window prototypes."""
    source, decoded, archive = _batch(raw, "raw shape"), _batch(reconstructed, "decoded shape"), _batch(prototypes, "prototypes")
    if source.shape != decoded.shape or archive.shape[1] != source.shape[1] or k < 1:
        raise ValueError("incompatible shape retrieval support or k")
    x, raw_constant = _shape_normalize(source)
    y, decoded_constant = _shape_normalize(decoded)
    a, prototype_constant = _shape_normalize(archive)
    # Explicit distances include the zero-vector convention for constant series.
    dx = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(a*a, axis=1)[None, :] - 2*x @ a.T, 0)
    dy = np.maximum(np.sum(y*y, axis=1)[:, None] + np.sum(a*a, axis=1)[None, :] - 2*y @ a.T, 0)
    effective_k = min(k, archive.shape[0])
    raw_order_all = np.argsort(dx, axis=1, kind="stable")
    rx = raw_order_all[:, :effective_k]
    ry = np.argsort(dy, axis=1, kind="stable")[:, :effective_k]
    discounts = 1.0 / np.log2(np.arange(effective_k) + 2)
    recalls, ndcgs = [], []
    for raw_order, decoded_order in zip(rx, ry):
        relevant = set(raw_order.tolist())
        hits = np.asarray([int(item in relevant) for item in decoded_order])
        recalls.append(float(hits.mean()))
        ndcgs.append(float(np.dot(hits, discounts) / discounts.sum()))
    delta = np.linalg.norm(x-y, axis=1)
    if archive.shape[0] >= 2:
        distances = np.sqrt(np.take_along_axis(dx, raw_order_all[:, :2], axis=1))
        margins = distances[:, 1] - distances[:, 0]
        certified = margins > 2*delta + 1e-12
        violations = certified & (rx[:, 0] != ry[:, 0])
        if np.any(violations):
            raise RuntimeError("actual-displacement nearest-neighbor certificate violated")
        coverage: float | None = float(np.mean(certified))
        minimum_margin: float | None = float(np.min(margins))
    else:
        coverage, minimum_margin = None, None
    return {
        "retrieval_nn_identity_retention": float(np.mean(rx[:, 0] == ry[:, 0])),
        "retrieval_recall_at_k": float(np.mean(recalls)),
        "retrieval_ndcg_at_k": float(np.mean(ndcgs)),
        "retrieval_effective_k": effective_k,
        "retrieval_prototypes": archive.shape[0],
        "retrieval_shape_delta_max": float(np.max(delta)),
        "retrieval_actual_nn_certificate_coverage": coverage,
        "retrieval_actual_nn_certificate_violations": 0,
        "retrieval_raw_nearest_margin_min": minimum_margin,
        "retrieval_certificate_policy": "a posteriori actual shape displacement: d2-d1>2*delta+1e-12; not a contract-rho coverage claim",
        "retrieval_constant_raw_queries": int(raw_constant.sum()),
        "retrieval_constant_reconstructed_queries": int(decoded_constant.sum()),
        "retrieval_constant_prototypes": int(prototype_constant.sum()),
        "retrieval_constant_policy": "centered norm<=1e-15 -> zero vector; stable prototype row order breaks ties; no Pearson certificate for constants",
        "retrieval_task": "one-sided compressed query, fixed raw TRAIN prototypes, whole 128-point normalized Euclidean shape",
    }


def lag_acf(value: np.ndarray, *, max_lag: int = 16) -> np.ndarray:
    """Project contract ACF: Pearson between separately centered lag slices."""
    batch = _batch(value, "ACF input")
    if max_lag < 1 or max_lag > batch.shape[1] - 2:
        raise ValueError("invalid finite-lag ACF support")
    result = np.empty((batch.shape[0], max_lag), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        left, right = batch[:, :-lag], batch[:, lag:]
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        numerator = np.sum(left*right, axis=1)
        denominator = np.sqrt(np.sum(left*left, axis=1) * np.sum(right*right, axis=1))
        result[:, lag-1] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)
    return result


def lag_metrics(raw: np.ndarray, reconstructed: np.ndarray, *, max_lag: int = 16) -> dict[str, float]:
    source, decoded = _batch(raw, "raw ACF"), _batch(reconstructed, "decoded ACF")
    if source.shape != decoded.shape:
        raise ValueError("ACF task inputs must have identical support")
    difference = lag_acf(source, max_lag=max_lag) - lag_acf(decoded, max_lag=max_lag)
    return {"lag_signature_rmse": float(np.sqrt(np.mean(difference**2))),
            "lag_signature_max_abs": float(np.max(np.abs(difference))),
            "lag_signature_mae": float(np.mean(np.abs(difference)))}


def evaluate_tasks(bundle: dict[str, Any], reconstructed: np.ndarray, model: dict[str, Any], *, prototypes: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate already-decoded reconstruction; this module never calls a codec."""
    source = bundle["source"]
    if np.asarray(source).shape != np.asarray(reconstructed).shape:
        raise ValueError("decoded tensor does not match frozen encoder input")
    archive = train_prototypes(bundle["train_1d"]) if prototypes is None else prototypes
    metrics = forecast_metrics(bundle["future"], predict_ridge(model, source), predict_ridge(model, reconstructed),
                               bundle["train_1d"], seasonal_period=int(bundle.get("seasonal_period", 1)))
    persistence = forecast_metrics(
        bundle["future"], predict_persistence(source, horizon=bundle["future"].shape[1]),
        predict_persistence(reconstructed, horizon=bundle["future"].shape[1]),
        bundle["train_1d"], seasonal_period=int(bundle.get("seasonal_period", 1)),
    )
    metrics.update({key.replace("forecast_", "persistence_", 1): value for key, value in persistence.items()})
    metrics.update(shape_retrieval_metrics(source, reconstructed, archive))
    metrics.update(lag_metrics(source, reconstructed))
    return metrics


def task_metrics(bundle: dict[str, Any], reconstruction: np.ndarray, fitted: dict[str, Any], *, prototypes: np.ndarray | None = None) -> dict[str, Any]:
    """Public orchestration alias; inputs must be final persisted reconstructions."""
    return evaluate_tasks(bundle, reconstruction, fitted, prototypes=prototypes)
