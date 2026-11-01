from __future__ import annotations

import numpy as np


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    error = reconstructed - original
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    per_channel_std = np.std(original, axis=(0, 2))
    per_channel_rmse = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    nrmse = float(np.mean(per_channel_rmse / np.maximum(per_channel_std, 1e-8)))
    correlations = []
    for channel in range(original.shape[1]):
        x = original[:, channel].ravel()
        y = reconstructed[:, channel].ravel()
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    series_correlations = []
    for original_window, reconstructed_window in zip(original, reconstructed):
        for x, y in zip(original_window, reconstructed_window):
            x_centered = x.astype(np.float64) - float(np.mean(x))
            y_centered = y.astype(np.float64) - float(np.mean(y))
            x_energy = float(np.dot(x_centered, x_centered))
            y_energy = float(np.dot(y_centered, y_centered))
            if x_energy <= 1e-15:
                series_correlations.append(1.0 if np.array_equal(x, y) else 0.0)
                continue
            denominator = float(np.sqrt(x_energy * y_energy))
            series_correlations.append(
                float(np.dot(x_centered, y_centered) / denominator)
                if denominator > 1e-15 else 0.0
            )
    return {
        "mse": mse,
        "mae": mae,
        "nrmse": nrmse,
        "correlation": float(np.mean(correlations)) if correlations else 0.0,
        "correlation_series_min": (
            float(np.min(series_correlations)) if series_correlations else 0.0
        ),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def acf_preservation_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    max_lag: int,
) -> dict[str, float]:
    """Compare per-window, per-channel Pearson ACFs at lags 1..max_lag."""
    max_lag = min(int(max_lag), original.shape[-1] - 2)
    if max_lag < 1:
        return {"acf_mae": 0.0, "acf_series_max_mae": 0.0, "acf_max_abs": 0.0}
    series_errors = []
    all_errors = []
    for original_window, reconstructed_window in zip(original, reconstructed):
        for original_series, reconstructed_series in zip(original_window, reconstructed_window):
            lag_errors = []
            for lag in range(1, max_lag + 1):
                left = original_series[:-lag] - original_series[:-lag].mean()
                right = original_series[lag:] - original_series[lag:].mean()
                reconstructed_left = reconstructed_series[:-lag] - reconstructed_series[:-lag].mean()
                reconstructed_right = reconstructed_series[lag:] - reconstructed_series[lag:].mean()
                denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
                reconstructed_denominator = float(np.sqrt(
                    np.dot(reconstructed_left, reconstructed_left) *
                    np.dot(reconstructed_right, reconstructed_right)
                ))
                original_acf = float(np.dot(left, right) / denominator) if denominator > 1e-15 else 0.0
                reconstructed_acf = (
                    float(np.dot(reconstructed_left, reconstructed_right) / reconstructed_denominator)
                    if reconstructed_denominator > 1e-15 else 0.0
                )
                lag_errors.append(abs(original_acf - reconstructed_acf))
            series_errors.append(float(np.mean(lag_errors)))
            all_errors.extend(lag_errors)
    return {
        "acf_mae": float(np.mean(all_errors)) if all_errors else 0.0,
        "acf_series_max_mae": float(np.max(series_errors)) if series_errors else 0.0,
        "acf_max_abs": float(np.max(all_errors)) if all_errors else 0.0,
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    labels_bool = labels.astype(bool)
    tp = int(np.logical_and(predictions, labels_bool).sum())
    fp = int(np.logical_and(predictions, ~labels_bool).sum())
    fn = int(np.logical_and(~predictions, labels_bool).sum())
    tn = int(np.logical_and(~predictions, ~labels_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
