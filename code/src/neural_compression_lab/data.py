from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def _numeric_or_code(series: pd.Series) -> pd.Series:
    """Convert numeric columns and strings such as ``ACTIVE|3`` to floats."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return numeric
    extracted = series.astype(str).str.extract(r"(?:\||^)(-?\d+(?:\.\d+)?)$")[0]
    coded = pd.to_numeric(extracted, errors="coerce")
    return numeric.fillna(coded)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> "Standardizer":
        joined = np.concatenate(list(arrays), axis=0).astype(np.float64)
        mean = np.nanmean(joined, axis=0)
        std = np.nanstd(joined, axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse(self, array: np.ndarray) -> np.ndarray:
        return array * self.std + self.mean


def load_trip(
    path: Path,
    channels: list[str],
    label_column: str | None,
    time_slice: Sequence[int] | None = None,
    *,
    role: str = "dataset",
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected CSV columns as ``(time, channel)`` with optional labels.

    CSV role slices use the same half-open ``[start, end)`` convention as NPY
    input.  ``label_column=None`` is the explicit unlabeled-data contract and
    yields an all-zero label vector.  ``skipinitialspace`` normalizes official
    CSV headers such as ``, RESP`` to the configured column name ``RESP``.
    """
    if not channels:
        raise ValueError("data.channels must contain at least one channel")
    usecols = list(channels)
    if label_column is not None and label_column not in usecols:
        usecols.append(label_column)
    frame = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
        skipinitialspace=True,
    )
    values = pd.DataFrame({c: _numeric_or_code(frame[c]) for c in channels})
    values = values.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        numeric_values = values.to_numpy(dtype=np.float32)
    if not bool(np.isfinite(numeric_values).all()):
        raise ValueError(f"CSV values are not finite after float32 conversion: {path}")
    if label_column is None:
        labels = np.zeros(len(frame), dtype=np.float32)
    else:
        labels = (
            _numeric_or_code(frame[label_column])
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        labels = (labels > 0).astype(np.float32)
    start, end = _validated_time_slice(time_slice, len(frame), role=role)
    return numeric_values[start:end].copy(), labels[start:end].copy()


def _validated_time_slice(
    requested: Sequence[int] | None,
    time_steps: int,
    *,
    role: str,
) -> tuple[int, int]:
    """Return a validated half-open ``[start, end)`` interval."""
    if requested is None:
        return 0, time_steps
    if isinstance(requested, (str, bytes)) or len(requested) != 2:
        raise ValueError(f"data.role_slices.{role} must be a [start, end] pair")
    start, end = requested
    if (
        isinstance(start, (bool, np.bool_))
        or isinstance(end, (bool, np.bool_))
        or not isinstance(start, (int, np.integer))
        or not isinstance(end, (int, np.integer))
    ):
        raise ValueError(f"data.role_slices.{role} bounds must be integers")
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > time_steps:
        raise ValueError(
            f"data.role_slices.{role}=[{start}, {end}] is outside "
            f"the valid half-open range [0, {time_steps}]"
        )
    return start, end


def load_npy_trip(
    path: Path,
    channels: list[str],
    array_layout: str,
    time_slice: Sequence[int] | None = None,
    *,
    role: str = "dataset",
) -> tuple[np.ndarray, np.ndarray]:
    """Load an unlabeled 1-D/2-D NPY array as ``(time, channel)``.

    ``array_layout`` is deliberately mandatory at the configuration layer.  It
    prevents a shape-compatible array from being silently interpreted with the
    wrong time axis.  Role slices use half-open ``[start, end)`` indices on that
    time axis.  NPY evaluation arrays have no task labels, so the returned label
    vector is identically zero.
    """
    if array_layout not in {"channel_time", "time_channel"}:
        raise ValueError(
            "data.array_layout must be 'channel_time' or 'time_channel' for NPY input"
        )
    if not channels:
        raise ValueError("data.channels must contain at least one channel")

    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise ValueError(f"NPY input must contain one ndarray: {path}")
    if loaded.ndim == 1:
        time_major = loaded.reshape(-1, 1)
    elif loaded.ndim == 2:
        time_major = loaded.T if array_layout == "channel_time" else loaded
    else:
        raise ValueError(
            f"NPY input must be 1-D or 2-D, got shape {loaded.shape} in {path}"
        )

    if time_major.shape[1] != len(channels):
        raise ValueError(
            f"NPY channel count mismatch in {path}: layout {array_layout!r} gives "
            f"{time_major.shape[1]} channels, but data.channels declares {len(channels)}"
        )
    start, end = _validated_time_slice(
        time_slice,
        int(time_major.shape[0]),
        role=role,
    )
    selected = np.asarray(time_major[start:end])
    try:
        finite = np.isfinite(selected)
    except TypeError as exc:
        raise ValueError(f"NPY input must have a numeric dtype: {path}") from exc
    if not bool(finite.all()):
        raise ValueError(f"NPY input contains non-finite values in the {role} slice: {path}")
    with np.errstate(over="ignore", invalid="ignore"):
        values = selected.astype(np.float32, copy=True)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"NPY values are not finite after float32 conversion: {path}")
    labels = np.zeros(values.shape[0], dtype=np.float32)
    return values, labels


class WindowDataset(Dataset):
    def __init__(
        self,
        trips: list[np.ndarray],
        labels: list[np.ndarray],
        standardizer: Standardizer,
        window: int,
        stride: int,
        max_windows: int | None = None,
        seed: int = 0,
    ) -> None:
        self.trips = [standardizer.transform(t) for t in trips]
        self.labels = labels
        self.window = window
        self.index: list[tuple[int, int]] = []
        for trip_id, trip in enumerate(trips):
            self.index.extend((trip_id, start) for start in range(0, len(trip) - window + 1, stride))
        if max_windows is not None and len(self.index) > max_windows:
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(len(self.index), size=max_windows, replace=False))
            self.index = [self.index[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        trip_id, start = self.index[item]
        end = start + self.window
        x = torch.from_numpy(self.trips[trip_id][start:end].T.copy())
        # A window is positive when it contains any brake application.
        y = torch.tensor(float(self.labels[trip_id][start:end].max()), dtype=torch.float32)
        return x, y


@dataclass
class DataBundle:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    standardizer: Standardizer
    channels: list[str]
    signal_groups: dict[str, list[int]]

    def loaders(self, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
        kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        train = DataLoader(self.train, shuffle=True, **kwargs)
        val = DataLoader(self.val, shuffle=False, **kwargs)
        test = DataLoader(self.test, shuffle=False, **kwargs)
        return train, val, test


def build_data(config: dict, workspace: Path) -> DataBundle:
    data_cfg = config["data"]
    root = (workspace / data_cfg["root"]).resolve()
    channels = list(data_cfg["channels"])
    data_format = str(data_cfg.get("format", "csv")).lower()
    if data_format not in {"csv", "npy"}:
        raise ValueError("data.format must be 'csv' or 'npy'")

    configured_slices = data_cfg.get("role_slices", {})
    if not isinstance(configured_slices, dict):
        raise ValueError("data.role_slices must be a mapping")
    allowed_roles = {"train", "validation", "test"}
    unknown_roles = set(configured_slices) - allowed_roles
    if unknown_roles:
        raise ValueError(
            "data.role_slices has unknown roles: " + ", ".join(sorted(unknown_roles))
        )
    role_slices: dict[str, Sequence[int]] = configured_slices

    if data_format == "csv":
        label_column = data_cfg.get("label_column")
        if label_column is not None and (
            not isinstance(label_column, str) or not label_column.strip()
        ):
            raise ValueError("data.label_column must be a non-empty string or null")
        array_layout = None
    else:
        label_column = None
        if "array_layout" not in data_cfg:
            raise ValueError("data.array_layout is required when data.format is 'npy'")
        array_layout = str(data_cfg["array_layout"])

    def read_many(
        names: list[str],
        role: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        trips, labels = [], []
        for name in names:
            path = root / name
            if not path.exists():
                raise FileNotFoundError(path)
            if data_format == "csv":
                trip, label = load_trip(
                    path,
                    channels,
                    label_column,
                    role_slices.get(role),
                    role=role,
                )
            else:
                assert array_layout is not None
                trip, label = load_npy_trip(
                    path,
                    channels,
                    array_layout,
                    role_slices.get(role),
                    role=role,
                )
            trips.append(trip)
            labels.append(label)
        return trips, labels

    train_trips, train_labels = read_many(data_cfg["train_files"], "train")
    val_trips, val_labels = read_many(data_cfg["val_files"], "validation")
    test_trips, test_labels = read_many(data_cfg["test_files"], "test")
    standardizer = Standardizer.fit(train_trips)

    common = dict(
        standardizer=standardizer,
        window=int(data_cfg["window"]),
        stride=int(data_cfg["stride"]),
    )
    train = WindowDataset(
        train_trips,
        train_labels,
        max_windows=data_cfg.get("max_train_windows"),
        seed=int(config["seed"]),
        **common,
    )
    val = WindowDataset(
        val_trips,
        val_labels,
        max_windows=data_cfg.get("max_val_windows"),
        seed=int(config["seed"]) + 1,
        **common,
    )
    test = WindowDataset(
        test_trips,
        test_labels,
        max_windows=data_cfg.get("max_test_windows"),
        seed=int(config["seed"]) + 2,
        **common,
    )
    return DataBundle(
        train=train,
        val=val,
        test=test,
        standardizer=standardizer,
        channels=channels,
        signal_groups={k: list(v) for k, v in data_cfg.get("signal_groups", {}).items()},
    )
