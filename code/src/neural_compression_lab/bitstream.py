from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile
import zlib

import numpy as np


def packed_unsigned(values: np.ndarray, alphabet_size: int) -> bytes:
    """Pack fixed-width unsigned integers into an actual byte stream."""
    bits = max(1, math.ceil(math.log2(alphabet_size)))
    flat = np.asarray(values, dtype=np.uint64).ravel()
    out = bytearray()
    accumulator = 0
    available = 0
    for value in flat:
        accumulator |= int(value) << available
        available += bits
        while available >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        out.append(accumulator & 0xFF)
    return bytes(out)


def unpack_unsigned(payload: bytes, count: int, alphabet_size: int) -> np.ndarray:
    """Inverse of :func:`packed_unsigned`; trailing byte padding is ignored."""
    bits = max(1, math.ceil(math.log2(alphabet_size)))
    mask = (1 << bits) - 1
    values = np.empty(count, dtype=np.uint64)
    accumulator = 0
    available = 0
    offset = 0
    for index in range(count):
        while available < bits:
            if offset >= len(payload):
                raise ValueError("Truncated fixed-width payload")
            accumulator |= payload[offset] << available
            available += 8
            offset += 1
        values[index] = accumulator & mask
        accumulator >>= bits
        available -= bits
    if np.any(values >= alphabet_size):
        raise ValueError("Decoded symbol is outside the configured alphabet")
    return values


def packed_binary(values: np.ndarray) -> bytes:
    return np.packbits(np.asarray(values, dtype=np.uint8).ravel(), bitorder="little").tobytes()


def unpack_binary(payload: bytes, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:count]


def signed_varints(values: np.ndarray) -> bytes:
    """Zig-zag encode signed integers and emit unsigned LEB128 bytes."""
    out = bytearray()
    for raw in np.asarray(values, dtype=np.int64).ravel():
        value = (int(raw) << 1) ^ (int(raw) >> 63)
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


def decode_signed_varints(payload: bytes, count: int) -> np.ndarray:
    values = np.empty(count, dtype=np.int64)
    offset = 0
    for index in range(count):
        encoded = 0
        shift = 0
        while True:
            if offset >= len(payload):
                raise ValueError("Truncated varint payload")
            byte = payload[offset]
            offset += 1
            encoded |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
            if shift >= 70:
                raise ValueError("Invalid varint payload")
        values[index] = (encoded >> 1) ^ -(encoded & 1)
    if offset != len(payload):
        raise ValueError("Unexpected trailing varint bytes")
    return values


def compressed_size(payloads: list[bytes]) -> int:
    """Return bytes after a single deflate stream plus a tiny container header."""
    joined = b"".join(payloads)
    return len(zlib.compress(joined, level=9)) + 16


def bsc_compress(
    payloads: list[bytes],
    executable: Path | None = None,
    verify: bool = True,
) -> bytes:
    """Compress one logical LFZip stream with the paper's libbsc settings.

    The upstream LFZip implementation calls ``bsc e ... -b64p -e2``.  We use
    the official Windows executable with the same options and optionally
    verify a byte-exact decode before accepting the measured stream.
    """
    if executable is None:
        executable = Path(__file__).resolve().parents[2] / "tools" / "libbsc" / "bsc.exe"
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"libbsc executable not found at {executable}. "
            "Install the official bsc 3.3.12 binary documented in UPSTREAMS.md."
        )

    raw = b"".join(payloads)
    with tempfile.TemporaryDirectory(prefix="neural_compression_bsc_") as temp_dir:
        directory = Path(temp_dir)
        source = directory / "symbols.bin"
        encoded = directory / "symbols.bsc"
        decoded = directory / "symbols.decoded.bin"
        source.write_bytes(raw)
        completed = subprocess.run(
            [str(executable), "e", str(source), str(encoded), "-b64p", "-e2"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not encoded.is_file():
            raise RuntimeError(f"libbsc compression failed: {completed.stderr or completed.stdout}")
        stream = encoded.read_bytes()
        if verify:
            completed = subprocess.run(
                [str(executable), "d", str(encoded), str(decoded)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not decoded.is_file():
                raise RuntimeError(f"libbsc decompression failed: {completed.stderr or completed.stdout}")
            if decoded.read_bytes() != raw:
                raise RuntimeError("libbsc round-trip verification failed")
        return stream


def bsc_compressed_size(payloads: list[bytes], executable: Path | None = None) -> int:
    return len(bsc_compress(payloads, executable=executable, verify=True))
