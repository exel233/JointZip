#!/usr/bin/env python3
"""Thin batch/bitstream adapter around the official CAMEO Cython implementation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys

import numpy as np


OFFICIAL_COMMIT = "97d146b4d06662107fe6ac5aaf7b24f55a45d862"
MAGIC = b"CAM1"
HEADER = struct.Struct("<4sIIIIdd")


def _write_uvarint(output: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("CAMEO index deltas must be non-negative")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def _read_uvarint(payload: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if position >= len(payload) or shift > 63:
            raise ValueError("invalid CAMEO index varint")
        byte = payload[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7


def _acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    values = []
    for lag in range(1, max_lag + 1):
        left, right = series[:-lag], series[lag:]
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = math.sqrt(float(np.dot(left_centered, left_centered)) *
                                float(np.dot(right_centered, right_centered)))
        if denominator <= 1e-15:
            values.append(0.0)
        else:
            values.append(float(np.dot(left_centered, right_centered) / denominator))
    return np.asarray(values, dtype=np.float64)


def _encode_series(indices: np.ndarray, values: np.ndarray) -> bytes:
    if indices.size > 0xFFFF:
        raise ValueError("CAMEO adapter supports at most 65535 retained points per series")
    output = bytearray(struct.pack("<H", indices.size))
    previous = 0
    for index in indices:
        current = int(index)
        _write_uvarint(output, current - previous)
        previous = current
    output.extend(np.asarray(values, dtype="<f4").tobytes())
    return bytes(output)


def _decode_series(payload: bytes, position: int, length: int) -> tuple[np.ndarray, int]:
    if position + 2 > len(payload):
        raise ValueError("truncated CAMEO retained-point count")
    count = struct.unpack_from("<H", payload, position)[0]
    position += 2
    if count < 2:
        raise ValueError("CAMEO stream must retain both endpoints")
    indices = np.empty(count, dtype=np.int64)
    current = 0
    for item in range(count):
        delta, position = _read_uvarint(payload, position)
        current += delta
        indices[item] = current
    value_bytes = count * 4
    if position + value_bytes > len(payload):
        raise ValueError("truncated CAMEO retained values")
    values = np.frombuffer(payload, dtype="<f4", count=count, offset=position).astype(np.float64)
    position += value_bytes
    if indices[0] != 0 or indices[-1] != length - 1 or np.any(np.diff(indices) <= 0):
        raise ValueError("invalid CAMEO retained indices")
    reconstruction = np.interp(np.arange(length), indices, values).astype(np.float32)
    return reconstruction, position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("stream", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("batch", type=int)
    parser.add_argument("channels", type=int)
    parser.add_argument("time", type=int)
    parser.add_argument("acf_epsilon", type=float)
    parser.add_argument("max_lag", type=int)
    parser.add_argument("blocking_factor", type=float)
    parser.add_argument("official_source", type=Path)
    args = parser.parse_args()

    if args.batch < 1 or args.channels < 1 or args.time < 4:
        raise ValueError("invalid CAMEO tensor shape")
    if not 0 < args.acf_epsilon:
        raise ValueError("CAMEO ACF epsilon must be positive")
    max_lag = min(args.max_lag, args.time - 2)
    if max_lag < 1:
        raise ValueError("CAMEO max_lag must be positive and shorter than the series")
    sys.path.insert(0, str(args.official_source))
    from compression.lpc.cameo import simplify_by_sip  # type: ignore

    source = np.fromfile(args.input, dtype="<f4")
    expected = args.batch * args.channels * args.time
    if source.size != expected:
        raise ValueError(f"input contains {source.size} values, expected {expected}")
    source = source.reshape(args.batch, args.channels, args.time)
    blocking = max(1, int(args.blocking_factor * math.log(args.time)))
    stream = bytearray(HEADER.pack(
        MAGIC, args.batch, args.channels, args.time, max_lag,
        args.acf_epsilon, args.blocking_factor,
    ))

    for window in source:
        for series32 in window:
            series = series32.astype(np.float64)
            if float(np.ptp(series)) <= 1e-12:
                indices = np.asarray([0, args.time - 1], dtype=np.int64)
            else:
                # The official routine mutates its input while evaluating removals.
                # Keep the source values immutable and use only its returned mask.
                mask = np.asarray(
                    simplify_by_sip(series.copy(), blocking, max_lag, args.acf_epsilon),
                    dtype=bool,
                )
                mask[0] = True
                mask[-1] = True
                indices = np.flatnonzero(mask)
            values = series32[indices]
            reconstruction = np.interp(np.arange(args.time), indices, values).astype(np.float64)
            deviation = float(np.mean(np.abs(_acf(series, max_lag) - _acf(reconstruction, max_lag))))
            if not np.isfinite(deviation) or deviation >= args.acf_epsilon + 1e-10:
                # Degenerate correlations and numerical edge cases should never be
                # reported as satisfying a statistical error guarantee.
                indices = np.arange(args.time, dtype=np.int64)
                values = series32
            stream.extend(_encode_series(indices, values))

    args.stream.write_bytes(stream)
    decoded = np.empty_like(source)
    payload = bytes(stream)
    position = HEADER.size
    for batch_index in range(args.batch):
        for channel_index in range(args.channels):
            decoded[batch_index, channel_index], position = _decode_series(
                payload, position, args.time
            )
    if position != len(payload):
        raise ValueError("CAMEO decoder did not consume the complete stream")
    decoded.tofile(args.reconstruction)


if __name__ == "__main__":
    main()
