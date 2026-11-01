from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn
import zstandard as zstd


def _frame_streams(streams: list[bytes]) -> bytes:
    """Prefix each independently decodable window stream with its byte length."""
    return b"".join(struct.pack("<I", len(stream)) + stream for stream in streams)


def _checked_float32_shape(
    batch: int, channels: int, time_points: int, label: str
) -> tuple[int, int, int, int]:
    """Validate a persisted [B,C,T] shape without fixed-width overflow."""
    if min(batch, channels, time_points) < 1:
        raise ValueError(f"invalid {label} tensor shape")
    values = batch * channels * time_points
    if values > np.iinfo(np.intp).max // np.dtype("<f4").itemsize:
        raise ValueError(f"{label} tensor shape exceeds addressable memory")
    return batch, channels, time_points, values


def _parse_framed_float32_shape(
    payload: bytes, magic: bytes, label: str, length_bytes: int = 4
) -> tuple[int, int, int, int]:
    """Parse the common magic/[B,C,T]/per-window-length framing prefix."""
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} payload must be bytes")
    header = struct.Struct("<4sIII")
    if len(payload) < header.size:
        raise ValueError(f"truncated {label} header")
    observed_magic, batch, channels, time_points = header.unpack_from(payload)
    if observed_magic != magic:
        raise ValueError(f"invalid {label} magic")
    shape = _checked_float32_shape(batch, channels, time_points, label)
    if batch > (len(payload) - header.size) // length_bytes:
        raise ValueError(f"truncated {label} window framing")
    return shape


def _decode_zstd_lossless_payload(payload: bytes) -> np.ndarray:
    batch, channels, time_points, _ = _parse_framed_float32_shape(
        payload, b"ZSL1", "Zstandard lossless"
    )
    header = struct.Struct("<4sIII")
    position = header.size
    raw_bytes = channels * time_points * np.dtype("<f4").itemsize
    windows: list[np.ndarray] = []
    for _ in range(batch):
        if position + 4 > len(payload):
            raise ValueError("truncated Zstandard frame length")
        frame_size = struct.unpack_from("<I", payload, position)[0]
        position += 4
        if frame_size < 1 or position + frame_size > len(payload):
            raise ValueError("truncated Zstandard frame")
        frame = payload[position:position + frame_size]
        try:
            if zstd.frame_content_size(frame) != raw_bytes:
                raise ValueError("Zstandard frame declares an invalid byte count")
            decoder = zstd.ZstdDecompressor().decompressobj()
            raw = decoder.decompress(frame)
        except zstd.ZstdError as error:
            raise ValueError("invalid Zstandard lossless frame") from error
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("Zstandard frame was not consumed exactly")
        if len(raw) != raw_bytes:
            raise ValueError("Zstandard frame decoded an invalid byte count")
        windows.append(
            np.frombuffer(raw, dtype="<f4").copy().reshape(channels, time_points)
        )
        position += frame_size
    if position != len(payload):
        raise ValueError("Zstandard lossless stream has trailing bytes")
    return np.stack(windows)


class NativeClassicalCodec(nn.Module):
    """Base class for classical CPU compressors that already emit compressed bytes."""

    symbol_kind = "native"
    requires_epsilon = True
    quality_key = "epsilon"
    quality_spec_key = "residual_epsilons"
    entropy_coder = "native"

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, quality: float | None) -> tuple[torch.Tensor, bytes]:
        raise NotImplementedError

    def operating_points(self, spec: dict) -> list[dict[str, object]]:
        if not self.requires_epsilon:
            return [{"label": "lossless"}]
        return [
            {"label": f"{self.quality_key}={value:g}", self.quality_key: float(value)}
            for value in spec[self.quality_spec_key]
        ]

    @torch.no_grad()
    def compress_reconstruct_point(
        self, x: torch.Tensor, point: dict[str, object]
    ) -> tuple[torch.Tensor, bytes]:
        quality = point.get(self.quality_key)
        return self.compress_reconstruct(x, None if quality is None else float(quality))


class ZstdLosslessCodec(NativeClassicalCodec):
    """Independent-window float32 baseline using the official Zstandard library."""

    method_name = "zstd_lossless"
    requires_epsilon = False

    def __init__(self, level: int = 19) -> None:
        super().__init__()
        if level < 1 or level > 22:
            raise ValueError("Zstandard level must be in [1,22]")
        self.level = int(level)
        self.entropy_coder = f"zstd-{'.'.join(map(str, zstd.ZSTD_VERSION))}-level{level}"

    @torch.no_grad()
    def compress_reconstruct(
        self, x: torch.Tensor, quality: float | None = None
    ) -> tuple[torch.Tensor, bytes]:
        if x.dtype != torch.float32:
            raise ValueError("Zstandard lossless requires a float32 tensor")
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4")
        if (
            source.ndim != 3
            or min(source.shape) < 1
            or not np.isfinite(source).all()
        ):
            raise ValueError("Zstandard lossless expects a finite [batch,channel,time] tensor")
        compressor = zstd.ZstdCompressor(
            level=self.level, write_content_size=True, write_checksum=False
        )
        streams: list[bytes] = []
        for window in source:
            raw = np.ascontiguousarray(window, dtype="<f4").tobytes()
            encoded = compressor.compress(raw)
            streams.append(encoded)
        header = struct.pack(
            "<4sIII", b"ZSL1", source.shape[0], source.shape[1], source.shape[2]
        )
        payload = header + _frame_streams(streams)
        reconstructed = _decode_zstd_lossless_payload(payload)
        if not np.array_equal(source.view(np.uint32), reconstructed.view(np.uint32)):
            raise RuntimeError("official Zstandard lossless stream changed float32 bytes")
        self.last_metadata = {"batch": int(source.shape[0]), "level": self.level}
        return torch.from_numpy(reconstructed).to(x.device, x.dtype), payload

    def decompress_payload(self, payload: bytes) -> np.ndarray:
        """Decode only the complete counted ZSL1 artifact."""
        return _decode_zstd_lossless_payload(payload)


class _OfficialWSLLosslessCodec(NativeClassicalCodec):
    """Shared adapter for official lossless batch CLIs with persisted streams."""

    requires_epsilon = False
    binary_relative_path: tuple[str, ...]
    build_hint: str
    container_magic: bytes
    container_length_bytes: int

    def __init__(self) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        self.binary = root.joinpath(*self.binary_relative_path)
        self.last_metadata: dict[str, object] = {}

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError(f"{self.method_name} requires WSL")
        if not self.binary.is_file():
            raise RuntimeError(
                f"{self.method_name} backend is missing; run `{self.build_hint}`"
            )
        return wsl

    @torch.no_grad()
    def compress_reconstruct(
        self, x: torch.Tensor, quality: float | None = None
    ) -> tuple[torch.Tensor, bytes]:
        if x.dtype != torch.float32:
            raise ValueError(f"{self.method_name} requires a float32 tensor")
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4")
        if source.ndim != 3 or min(source.shape) < 1:
            raise ValueError(f"{self.method_name} expects [batch,channel,time]")
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix=f"nclab_{self.method_name}_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "stream.bin"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            process = _run_native(
                [
                    wsl,
                    "--exec",
                    _wsl_path(self.binary),
                    _wsl_path(input_path),
                    _wsl_path(stream_path),
                    _wsl_path(reconstruction_path),
                    str(source.shape[0]),
                    str(source.shape[1]),
                    str(source.shape[2]),
                ],
                timeout=300,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown native error"
                raise RuntimeError(f"{self.method_name} backend failed: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype="<f4")
            payload = stream_path.read_bytes()
        if reconstructed.size != source.size:
            raise RuntimeError(f"{self.method_name} decoded an invalid value count")
        reconstructed = reconstructed.reshape(source.shape)
        if not np.array_equal(source.view(np.uint32), reconstructed.view(np.uint32)):
            raise RuntimeError(f"{self.method_name} failed its bit-exact round trip")
        if not payload:
            raise RuntimeError(f"{self.method_name} returned an empty stream")
        self.last_metadata = {"backend_stdout": process.stdout.strip()}
        return torch.from_numpy(reconstructed).to(x.device, x.dtype), payload

    def _payload_shape(self, payload: bytes) -> tuple[int, int, int, int]:
        return _parse_framed_float32_shape(
            payload,
            self.container_magic,
            f"{self.method_name} container",
            self.container_length_bytes,
        )

    def decompress_payload(self, payload: bytes) -> np.ndarray:
        """Replay a persisted official container through its native decoder."""
        batch, channels, time_points, values = self._payload_shape(payload)
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(
            prefix=f"nclab_{self.method_name}_decode_"
        ) as directory:
            temp = Path(directory)
            stream_path = temp / "stream.bin"
            reconstruction_path = temp / "reconstruction.f32"
            stream_path.write_bytes(payload)
            process = _run_native(
                [
                    wsl,
                    "--exec",
                    _wsl_path(self.binary),
                    "--decode",
                    _wsl_path(stream_path),
                    _wsl_path(reconstruction_path),
                ],
                timeout=300,
            )
            if process.returncode != 0:
                detail = (
                    process.stderr.strip()
                    or process.stdout.strip()
                    or "unknown native error"
                )
                raise ValueError(
                    f"persisted {self.method_name} decode failed: {detail}"
                )
            expected_bytes = values * np.dtype("<f4").itemsize
            if (
                not reconstruction_path.is_file()
                or reconstruction_path.stat().st_size != expected_bytes
            ):
                raise ValueError(
                    f"persisted {self.method_name} decoded an invalid byte count"
                )
            reconstructed = np.fromfile(reconstruction_path, dtype="<f4").reshape(
                batch, channels, time_points
            )
        self.last_metadata = {
            **self.last_metadata,
            "decode_backend_stdout": process.stdout.strip(),
            "persisted_decode_verified": True,
        }
        return reconstructed


class FPZipLosslessCodec(_OfficialWSLLosslessCodec):
    """LLNL fpzip 1.3.0 complete float32 stream in reversible mode."""

    method_name = "fpzip_lossless"
    upstream_commit = "4a539c06d98b1c029b08324a086d4b75689a2b72"
    entropy_coder = "official-fpzip-1.3.0-FPZIP_FP_FAST-prec0"
    binary_relative_path = ("tools", "fpzip", "bin", "fpzip_batch")
    build_hint = "wsl bash ./tools/fpzip/build_wsl.sh"
    container_magic = b"FPB1"
    container_length_bytes = 8

    def _payload_shape(self, payload: bytes) -> tuple[int, int, int, int]:
        if not isinstance(payload, bytes):
            raise TypeError("fpzip_lossless container payload must be bytes")
        header = struct.Struct("<4sIIIIIIQ")
        if len(payload) < header.size:
            raise ValueError("truncated fpzip_lossless container header")
        (
            magic, version, batch, channels, time_points, _, _,
            official_stream_bytes,
        ) = header.unpack_from(payload)
        if magic != self.container_magic or version != 1:
            raise ValueError("unsupported fpzip_lossless container version")
        shape = _checked_float32_shape(
            batch, channels, time_points, "fpzip_lossless container"
        )
        if batch > (len(payload) - header.size) // self.container_length_bytes:
            raise ValueError("truncated fpzip_lossless container window framing")
        framing_bytes = header.size + batch * self.container_length_bytes
        if official_stream_bytes != len(payload) - framing_bytes:
            raise ValueError("fpzip_lossless container byte accounting mismatch")
        return shape


class FPcompressLosslessCodec(_OfficialWSLLosslessCodec):
    """Official FPcompress v1.0.3 SPratio float32 stream."""

    method_name = "fpcompress_lossless"
    upstream_commit = "97f037249bd28682bfd83462ea1129e14df36d81"
    entropy_coder = "official-FPcompress-1.0.3-SPratio"
    binary_relative_path = ("tools", "fpcompress", "bin", "fpcompress_batch")
    build_hint = "wsl bash ./tools/fpcompress/build_wsl.sh"
    container_magic = b"FPR1"
    container_length_bytes = 4


def _wsl_path(path: Path) -> str:
    """Translate a local Windows path to the corresponding WSL mount path."""
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) < 3 or posix[1:3] != ":/":
        raise RuntimeError(f"SZ 2016 requires a drive-letter Windows path, got {resolved}")
    return f"/mnt/{posix[0].lower()}/{posix[3:]}"


def _run_native(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a native adapter with one bounded retry for transient WSL startup errors.

    WSL 2 can occasionally return ``CreateVm/E_ABORT`` while its VM is stopped.
    A second immediate launch starts the same immutable command; a persistent codec
    or input error is still returned to the caller unchanged.
    """
    completed: subprocess.CompletedProcess[str] | None = None
    for _ in range(2):
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode == 0:
            return completed
    assert completed is not None
    return completed


class SZ2016Codec(NativeClassicalCodec):
    """Original SZ 1.1 best-fit curve compressor from the IPDPS 2016 paper."""

    method_name = "sz2016"
    upstream_commit = "2123f4899b76e6dfe28861da0a70d63a6cd43f83"

    def __init__(self) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        self.binary = root / "tools" / "sz2016" / "bin" / "sz2016_batch"
        self.config = root / "tools" / "sz2016" / "sz2016.config"

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError("SZ 2016 uses its original Linux C code and requires WSL on Windows")
        if not self.binary.is_file():
            raise RuntimeError(
                "SZ 2016 native backend is missing; run `wsl bash ./tools/sz2016/build_wsl.sh`"
            )
        if not self.config.is_file():
            raise RuntimeError(f"SZ 2016 configuration is missing: {self.config}")
        return wsl

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, epsilon: float | None) -> tuple[torch.Tensor, bytes]:
        if epsilon is None:
            raise ValueError("SZ 2016 requires an absolute error bound")
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if source.ndim != 3:
            raise ValueError(f"SZ 2016 expects [batch, channel, time], got {source.shape}")
        if not np.isfinite(source).all():
            raise ValueError("SZ 2016 input contains NaN or infinity")
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix="neural_compression_sz2016_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "streams.sz"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            command = [
                wsl,
                "--exec",
                _wsl_path(self.binary),
                _wsl_path(self.config),
                _wsl_path(input_path),
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
                str(source.shape[0]),
                str(source.shape[1]),
                str(source.shape[2]),
                f"{float(epsilon):.17g}",
            ]
            process = _run_native(command, timeout=120)
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown native error"
                raise RuntimeError(f"SZ 2016 native backend failed: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype=np.float32)
            if reconstructed.size != source.size:
                raise RuntimeError(
                    f"SZ 2016 decoded {reconstructed.size} values, expected {source.size}"
                )
            reconstructed = reconstructed.reshape(source.shape)
            payload = stream_path.read_bytes()
        observed = float(np.max(np.abs(source - reconstructed)))
        if observed > float(epsilon) + 1e-5:
            raise RuntimeError(f"SZ 2016 violated the requested error bound: {observed} > {epsilon}")
        result = torch.from_numpy(reconstructed).to(device=x.device, dtype=x.dtype)
        return result, payload


class SPERRCodec(NativeClassicalCodec):
    """Official NCAR/SPERR 1D/2D point-wise-error mode over time-series windows."""

    method_name = "sperr"
    upstream_commit = "9f930a3970f86c8e390e135c97162b79a40d165b"
    entropy_coder = "official-SPERR-0.8.5"

    def __init__(self) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        self.binary = root / "tools" / "sperr" / "bin" / "sperr_batch"

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError("SPERR uses its official Linux C++ code and requires WSL on Windows")
        if not self.binary.is_file():
            raise RuntimeError(
                "SPERR native backend is missing; run `wsl bash ./tools/sperr/build_wsl.sh`"
            )
        return wsl

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, quality: float | None) -> tuple[torch.Tensor, bytes]:
        if quality is None:
            raise ValueError("SPERR requires an absolute point-wise error bound")
        epsilon = float(quality)
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if source.ndim != 3:
            raise ValueError(f"SPERR expects [batch, channel, time], got {source.shape}")
        if source.shape[1] < 1 or source.shape[2] < 2:
            raise ValueError("SPERR requires at least one channel and two time points")
        if not np.isfinite(source).all():
            raise ValueError("SPERR input contains NaN or infinity")
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix="neural_compression_sperr_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "streams.sperr"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            command = [
                wsl,
                "--exec",
                _wsl_path(self.binary),
                _wsl_path(input_path),
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
                str(source.shape[0]),
                str(source.shape[1]),
                str(source.shape[2]),
                f"{epsilon:.17g}",
            ]
            process = _run_native(command, timeout=180)
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown native error"
                raise RuntimeError(f"SPERR native backend failed: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype=np.float32)
            if reconstructed.size != source.size:
                raise RuntimeError(f"SPERR decoded {reconstructed.size} values, expected {source.size}")
            reconstructed = reconstructed.reshape(source.shape)
            payload = stream_path.read_bytes()
        observed = float(np.max(np.abs(source - reconstructed)))
        if observed > epsilon + max(1e-6, epsilon * 1e-5):
            raise RuntimeError(f"SPERR violated the requested error bound: {observed} > {epsilon}")
        self.last_metadata = {
            "spatial_mode": "official-SPECK1D_FLT" if source.shape[1] == 1 else "official-SPERR-2D-C-API"
        }
        result = torch.from_numpy(reconstructed).to(device=x.device, dtype=x.dtype)
        return result, payload


def _qpet_stream_metadata(payload: bytes) -> dict[str, object]:
    header = struct.Struct("<4sIIIIIIIIIddddd")
    legacy_window_header = struct.Struct("<IIddIIQQ")
    compact_window_header = struct.Struct("<HBBdIII")
    if len(payload) < header.size:
        raise ValueError("truncated QPET correlation stream header")
    (
        magic, version, batch, channels, time_points, max_lag, qoz_level, flags,
        correction_mode, group_size, epsilon, acf_epsilon, minimum_correlation,
        qpet_confidence, qpet_k,
    ) = header.unpack_from(payload)
    if magic != b"QPC1" or version not in {8, 9, 10, 11, 12, 13, 14, 15, 16}:
        raise ValueError("invalid QPET correlation stream header")
    position = header.size
    corrections = 0
    online_corrections = 0
    candidates = 0
    base_bytes = 0
    correction_bytes = 0
    selected_bounds = []
    selected_scales = []
    variable_eb_groups = 0
    global_eb_groups = 0
    double_correction_groups = 0
    float_exact_correction_groups = 0
    sparse_exact_correction_groups = 0
    sparse_xor_correction_groups = 0
    quantized_correction_groups = 0
    packed_quantized_correction_groups = 0
    adaptive_packed_correction_groups = 0
    factorized_packed_correction_groups = 0
    subset_rank_packed_correction_groups = 0
    predictive_packed_correction_groups = 0
    constant_channel_correction_groups = 0
    gradient_balanced_groups = 0
    sperr_groups = 0
    sperr_selected_bound_min = float("inf")
    sperr_selected_bound_max = 0.0
    lossless_zstd_groups = 0
    serf_xor_groups = 0
    serf_qt_groups = 0
    serf_smooth_groups = 0
    semantic_independent_groups = 0
    semantic_shared_groups = 0
    semantic_strict_constant_groups = 0
    decoded_windows = 0
    while decoded_windows < batch:
        window_header = compact_window_header if version >= 10 else legacy_window_header
        if position + window_header.size > len(payload):
            raise ValueError("truncated QPET correlation window header")
        if version >= 10:
            (
                logical_windows, base_mode, compact_flags, selected_bound,
                window_corrections, size, correction_size,
            ) = window_header.unpack_from(payload, position)
            if compact_flags != 0:
                raise ValueError("invalid QPET compact window flags")
            selected_scale = None
            window_candidates = 0
        else:
            (
                logical_windows, base_mode, selected_bound, selected_scale,
                window_corrections, window_candidates,
                size, correction_size,
            ) = window_header.unpack_from(payload, position)
        position += window_header.size
        if (
            logical_windows < 1
            or logical_windows > group_size
            or decoded_windows + logical_windows > batch
        ):
            raise ValueError("invalid QPET logical window group")
        valid_base_modes = {0, 1, 2, 3, 4, 5, 6, 7}
        if version >= 9:
            valid_base_modes.update({8, 9})
        if base_mode not in valid_base_modes:
            raise ValueError("invalid QPET base compressor mode")
        if size == 0 or position + size + correction_size > len(payload):
            raise ValueError("truncated official QPET window payload")
        if base_mode in {8, 9}:
            if size < 5 or payload[position:position + 4] != b"SBK1":
                raise ValueError("invalid semantic-block payload header")
            semantic_schema = payload[position + 4]
            if semantic_schema not in {1, 2} or (
                semantic_schema == 2 and version < 11
            ):
                raise ValueError("semantic-block schema is incompatible with stream version")
            semantic_strict_constant_groups += semantic_schema == 2
        correction_position = position + size
        if base_mode == 8:
            if correction_size != 0:
                raise ValueError("semantic independent-block stream has outer corrections")
        elif correction_mode == 0 and base_mode == 0:
            if correction_size != 0 or (version >= 10 and window_corrections != 0):
                raise ValueError("online QPET stream has an invalid outer correction field")
            if version < 10:
                online_corrections += window_corrections
        elif window_corrections == 0:
            if correction_size != 0:
                raise ValueError("unexpected QPET correction payload")
        else:
            if correction_size < 2:
                raise ValueError("truncated QPET correction payload")
            correction_format = payload[correction_position]
            if correction_format == 0:
                double_correction_groups += 1
            elif correction_format == 1:
                float_exact_correction_groups += 1
            elif correction_format == 2:
                sparse_exact_correction_groups += 1
            elif correction_format == 3:
                sparse_xor_correction_groups += 1
            elif correction_format == 4:
                quantized_correction_groups += 1
            elif correction_format == 5:
                if version < 10:
                    raise ValueError(
                        "constant-channel correction requires QPC stream version 10"
                    )
                constant_channel_correction_groups += 1
            elif correction_format == 6:
                if version < 12:
                    raise ValueError(
                        "packed quantized correction requires QPC stream version 12"
                    )
                packed_quantized_correction_groups += 1
            elif correction_format == 7:
                if version < 13:
                    raise ValueError(
                        "adaptive packed correction requires QPC stream version 13"
                    )
                adaptive_packed_correction_groups += 1
            elif correction_format == 8:
                if version < 14:
                    raise ValueError(
                        "factorized packed correction requires QPC stream version 14"
                    )
                factorized_packed_correction_groups += 1
            elif correction_format == 9:
                if version < 15:
                    raise ValueError(
                        "subset-rank packed correction requires QPC stream version 15"
                    )
                subset_rank_packed_correction_groups += 1
            elif correction_format == 10:
                if version < 16:
                    raise ValueError(
                        "predictive packed correction requires QPC stream version 16"
                    )
                predictive_packed_correction_groups += 1
            else:
                raise ValueError("invalid QPET correction payload format")
        position += size + correction_size
        if not (correction_mode == 0 and base_mode == 0):
            corrections += window_corrections
        candidates += window_candidates
        base_bytes += size
        correction_bytes += correction_size
        selected_bounds.append(selected_bound)
        if selected_scale is not None:
            selected_scales.append(selected_scale)
        variable_eb_groups += base_mode in {0, 2}
        global_eb_groups += base_mode == 1
        gradient_balanced_groups += base_mode == 2
        sperr_groups += base_mode == 3
        lossless_zstd_groups += base_mode == 4
        serf_xor_groups += base_mode == 5
        serf_qt_groups += base_mode == 6
        serf_smooth_groups += base_mode == 7
        semantic_independent_groups += base_mode == 8
        semantic_shared_groups += base_mode == 9
        if base_mode == 3:
            sperr_selected_bound_min = min(sperr_selected_bound_min, selected_bound)
            sperr_selected_bound_max = max(sperr_selected_bound_max, selected_bound)
        decoded_windows += logical_windows
    if decoded_windows != batch:
        raise ValueError("QPET logical window groups do not match batch size")
    if position != len(payload):
        raise ValueError("QPET correlation stream has trailing bytes")
    return {
        "version": version,
        "batch": batch,
        "channels": channels,
        "time": time_points,
        "max_lag": max_lag,
        "qoz_level": qoz_level,
        "flags": flags,
        "correction_mode": correction_mode,
        "group_size": group_size,
        "epsilon": epsilon,
        "acf_epsilon": acf_epsilon,
        "minimum_correlation": minimum_correlation,
        "qpet_confidence": qpet_confidence,
        "qpet_k": qpet_k,
        "corrections": corrections,
        "online_corrections": online_corrections,
        "candidates_tested": candidates,
        "base_payload_bytes": base_bytes,
        "correction_payload_bytes": correction_bytes,
        "variable_eb_groups": variable_eb_groups,
        "global_eb_groups": global_eb_groups,
        "double_correction_groups": double_correction_groups,
        "float_exact_correction_groups": float_exact_correction_groups,
        "sparse_exact_correction_groups": sparse_exact_correction_groups,
        "sparse_xor_correction_groups": sparse_xor_correction_groups,
        "quantized_correction_groups": quantized_correction_groups,
        "packed_quantized_correction_groups": packed_quantized_correction_groups,
        "adaptive_packed_correction_groups": adaptive_packed_correction_groups,
        "factorized_packed_correction_groups": factorized_packed_correction_groups,
        "subset_rank_packed_correction_groups": subset_rank_packed_correction_groups,
        "predictive_packed_correction_groups": predictive_packed_correction_groups,
        "constant_channel_correction_groups": constant_channel_correction_groups,
        "gradient_balanced_groups": gradient_balanced_groups,
        "sperr_groups": sperr_groups,
        "lossless_zstd_groups": lossless_zstd_groups,
        "serf_xor_groups": serf_xor_groups,
        "serf_qt_groups": serf_qt_groups,
        "serf_smooth_groups": serf_smooth_groups,
        "semantic_independent_groups": semantic_independent_groups,
        "semantic_shared_groups": semantic_shared_groups,
        "semantic_strict_constant_groups": semantic_strict_constant_groups,
        "sperr_selected_bound_min": (
            sperr_selected_bound_min if sperr_groups else None
        ),
        "sperr_selected_bound_max": (
            sperr_selected_bound_max if sperr_groups else None
        ),
        "selected_global_bound_min": min(selected_bounds),
        "selected_global_bound_max": max(selected_bounds),
        "selected_estimator_scale_min": min(selected_scales) if selected_scales else None,
        "selected_estimator_scale_max": max(selected_scales) if selected_scales else None,
    }


def _series_correlation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_energy = float(np.dot(x_centered, x_centered))
    y_energy = float(np.dot(y_centered, y_centered))
    if x_energy <= 1e-15:
        return 1.0 if np.array_equal(original, reconstructed) else 0.0
    denominator = float(np.sqrt(x_energy * y_energy))
    return float(np.dot(x_centered, y_centered) / denominator) if denominator > 1e-15 else 0.0


class QPETCorrelationCodec(NativeClassicalCodec):
    """Official QPET/HPEZ variable-EB pipeline with joint hard correlation constraints."""

    method_name = "qpet_correlation"
    upstream_commit = "5d17cb1647caf001bd7355c617856c72c4b6475e"
    entropy_coder = "official-QPET/HPEZ-or-SPERR-or-Serf+Zstd"

    def __init__(
        self,
        max_lag: int = 16,
        qoz_level: int = 3,
        correction_mode: str = "posthoc_greedy",
        estimator_scales: tuple[float, ...] | list[float] = (1.0, 2.0, 4.0, 8.0),
        qpet_confidence: float = 0.999,
        qpet_k: float = 2.0,
        group_size: int = 0,
        repair_denominators: tuple[int, ...] | list[int] = (7, 15, 31, 63, 127),
        candidate_profile: str = "generic",
        repair_optimizer: str = "legacy",
        correction_format_ceiling: int | None = None,
        base_frontier: str | None = None,
        relaxed_bound_ratio_ceiling: float | str | None = None,
        surrogate_frontier: str | None = None,
        cfsr_bound_ratio_ceiling: float | str | None = None,
    ) -> None:
        super().__init__()
        if max_lag < 1 or qoz_level not in range(5):
            raise ValueError("QPET max_lag must be positive and qoz_level must be in [0,4]")
        if correction_mode not in {"online", "posthoc_sequential", "posthoc_greedy"}:
            raise ValueError("invalid QPET correction mode")
        scales = tuple(float(value) for value in estimator_scales)
        if not scales or len(scales) > 16 or not all(np.isfinite(value) and value > 0 for value in scales):
            raise ValueError("QPET estimator_scales must contain 1-16 finite positive values")
        if not (0 < qpet_confidence < 1) or not np.isfinite(qpet_confidence):
            raise ValueError("QPET confidence must be finite and in (0,1)")
        if qpet_k <= 0 or not np.isfinite(qpet_k):
            raise ValueError("QPET k must be finite and positive")
        if group_size < 0:
            raise ValueError("QPET group_size must be nonnegative (zero means the full batch)")
        denominators = tuple(int(value) for value in repair_denominators)
        if len(denominators) > 16 or len(set(denominators)) != len(denominators):
            raise ValueError("QPET repair_denominators must contain at most 16 unique values")
        if any(value < 1 or value > 127 for value in denominators):
            raise ValueError("QPET repair_denominators must be in [1,127]")
        if candidate_profile not in {"generic", "vehicle_semantic"}:
            raise ValueError(
                "QPET candidate_profile must be generic or vehicle_semantic"
            )
        repair_optimizers = {
            "legacy",
            "zrcar",
            "fvro_value",
            "fvro",
            "ssbb_value",
            "ssbb",
        }
        if repair_optimizer not in repair_optimizers:
            raise ValueError(
                "QPET repair_optimizer must be legacy, zrcar, fvro_value, fvro, "
                "ssbb_value, or ssbb"
            )
        if correction_format_ceiling is None:
            effective_format_ceiling = (
                8 if repair_optimizer in {"ssbb_value", "ssbb"} else 6
            )
        else:
            if type(correction_format_ceiling) is not int:
                raise ValueError(
                    "QPET correction_format_ceiling must be an integer 6, 7, 8, 9, or 10"
                )
            effective_format_ceiling = correction_format_ceiling
        if effective_format_ceiling not in {6, 7, 8, 9, 10}:
            raise ValueError(
                "QPET correction_format_ceiling must be one of 6, 7, 8, 9, or 10"
            )
        if (
            repair_optimizer not in {"ssbb_value", "ssbb"}
            and effective_format_ceiling != 6
        ):
            raise ValueError(
                "QPET correction formats 7, 8, 9, and 10 require an SSBB optimizer"
            )
        frontier_is_explicit = (
            base_frontier is not None or relaxed_bound_ratio_ceiling is not None
        )
        if (base_frontier is None) != (relaxed_bound_ratio_ceiling is None):
            raise ValueError(
                "QPET base_frontier and relaxed_bound_ratio_ceiling must be "
                "configured together"
            )
        effective_base_frontier = (
            "incumbent_only" if base_frontier is None else base_frontier
        )
        if effective_base_frontier not in {"incumbent_only", "cjrf_v1"}:
            raise ValueError(
                "QPET base_frontier must be incumbent_only or cjrf_v1"
            )
        if effective_base_frontier == "incumbent_only":
            if frontier_is_explicit and relaxed_bound_ratio_ceiling != "none":
                raise ValueError(
                    "QPET incumbent_only requires "
                    "relaxed_bound_ratio_ceiling=none"
                )
            effective_relaxed_ceiling: float | str = "none"
            cjrf_ratio_mask = 0
        else:
            if (
                type(relaxed_bound_ratio_ceiling) not in {int, float}
                or not np.isfinite(float(relaxed_bound_ratio_ceiling))
                or float(relaxed_bound_ratio_ceiling) not in {1.25, 1.5, 2.0}
            ):
                raise ValueError(
                    "QPET CJRF relaxed_bound_ratio_ceiling must be exactly "
                    "1.25, 1.5, or 2.0"
                )
            effective_relaxed_ceiling = float(relaxed_bound_ratio_ceiling)
            cjrf_ratio_mask = {1.25: 1, 1.5: 3, 2.0: 7}[
                effective_relaxed_ceiling
            ]
            if (
                repair_optimizer != "ssbb"
                or effective_format_ceiling != 9
                or correction_mode != "posthoc_greedy"
                or candidate_profile != "generic"
                or group_size != 1
            ):
                raise ValueError(
                    "QPET CJRF v1 requires generic full SSBB, "
                    "posthoc_greedy, group_size=1, and "
                    "correction_format_ceiling=9"
                )
        surrogate_frontier_is_explicit = (
            surrogate_frontier is not None
            or cfsr_bound_ratio_ceiling is not None
        )
        if (surrogate_frontier is None) != (
            cfsr_bound_ratio_ceiling is None
        ):
            raise ValueError(
                "QPET surrogate_frontier and cfsr_bound_ratio_ceiling must "
                "be configured together"
            )
        effective_surrogate_frontier = (
            "none" if surrogate_frontier is None else surrogate_frontier
        )
        if effective_surrogate_frontier not in {"none", "cfsr_v1"}:
            raise ValueError(
                "QPET surrogate_frontier must be none or cfsr_v1"
            )
        if effective_surrogate_frontier == "none":
            if (
                surrogate_frontier_is_explicit
                and cfsr_bound_ratio_ceiling != "none"
            ):
                raise ValueError(
                    "QPET surrogate frontier none requires "
                    "cfsr_bound_ratio_ceiling=none"
                )
            effective_cfsr_ceiling: float | str = "none"
            cfsr_ratio_mask = 0
        else:
            if (
                type(cfsr_bound_ratio_ceiling) not in {int, float}
                or not np.isfinite(float(cfsr_bound_ratio_ceiling))
                or float(cfsr_bound_ratio_ceiling) not in {0.25, 0.5, 1.0}
            ):
                raise ValueError(
                    "QPET CFSR bound ratio ceiling must be exactly "
                    "0.25, 0.5, or 1.0"
                )
            effective_cfsr_ceiling = float(cfsr_bound_ratio_ceiling)
            cfsr_ratio_mask = {0.25: 1, 0.5: 3, 1.0: 7}[
                effective_cfsr_ceiling
            ]
            if (
                repair_optimizer != "ssbb"
                or effective_format_ceiling != 9
                or correction_mode != "posthoc_greedy"
                or candidate_profile != "generic"
                or group_size != 1
                or effective_base_frontier != "incumbent_only"
                or cjrf_ratio_mask != 0
            ):
                raise ValueError(
                    "QPET CFSR v1 requires generic full SSBB, "
                    "posthoc_greedy, group_size=1, "
                    "correction_format_ceiling=9, and the "
                    "incumbent_only/none base frontier"
                )
        uses_second_stage = repair_optimizer != "legacy"
        if uses_second_stage and correction_mode == "online":
            raise ValueError("QPET ZRCAR/FVRO/SSBB requires a post-hoc correction mode")
        if uses_second_stage and candidate_profile != "generic":
            raise ValueError(
                "QPET ZRCAR/FVRO/SSBB requires the generic candidate profile"
            )
        if uses_second_stage and group_size != 1:
            raise ValueError("QPET ZRCAR/FVRO/SSBB requires group_size=1")
        if repair_optimizer in {
            "fvro_value",
            "fvro",
            "ssbb_value",
            "ssbb",
        } and denominators != (
            7,
            15,
            31,
            63,
            127,
        ):
            raise ValueError(
                "QPET FVRO v1 and SSBB v2 require repair_denominators exactly "
                "(7,15,31,63,127) in that order"
            )
        root = Path(__file__).resolve().parents[2]
        self.binary = root / "tools" / "qpet_correlation" / "bin" / "qpet_correlation_batch"
        self.max_lag = int(max_lag)
        self.qoz_level = int(qoz_level)
        self.correction_mode = correction_mode
        self.estimator_scales = scales
        self.qpet_confidence = float(qpet_confidence)
        self.qpet_k = float(qpet_k)
        self.group_size = int(group_size)
        self.repair_denominators = denominators
        self.candidate_profile = candidate_profile
        self.repair_optimizer = repair_optimizer
        self.correction_format_ceiling = effective_format_ceiling
        self.base_frontier = effective_base_frontier
        self.relaxed_bound_ratio_ceiling = effective_relaxed_ceiling
        self.cjrf_ratio_mask = cjrf_ratio_mask
        self.base_frontier_is_explicit = frontier_is_explicit
        self.surrogate_frontier = effective_surrogate_frontier
        self.cfsr_bound_ratio_ceiling = effective_cfsr_ceiling
        self.cfsr_ratio_mask = cfsr_ratio_mask
        self.surrogate_frontier_is_explicit = surrogate_frontier_is_explicit
        self.last_metadata: dict[str, object] = {}

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError("QPET correlation uses official Linux C++ code and requires WSL")
        if not self.binary.is_file():
            raise RuntimeError(
                "QPET correlation backend is missing; run "
                "`wsl bash ./tools/qpet_correlation/build_wsl.sh`"
            )
        return wsl

    def operating_points(self, spec: dict) -> list[dict[str, object]]:
        points = []
        for item in spec["operating_points"]:
            point = {
                "epsilon": float(item["epsilon"]),
                "acf_epsilon": float(item.get("acf_epsilon", -1.0)),
                "min_correlation": float(item.get("min_correlation", -1.0)),
            }
            if point["acf_epsilon"] < 0 and point["min_correlation"] < 0:
                raise ValueError("each QPET point must enable ACF or Pearson correlation")
            point["label"] = item.get(
                "label",
                f"epsilon={point['epsilon']:g},acf={point['acf_epsilon']:g},"
                f"corr={point['min_correlation']:g}",
            )
            points.append(point)
        return points

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, quality: float | None) -> tuple[torch.Tensor, bytes]:
        raise TypeError("QPET correlation requires a full operating point, not one quality scalar")

    @torch.no_grad()
    def compress_reconstruct_point(
        self, x: torch.Tensor, point: dict[str, object]
    ) -> tuple[torch.Tensor, bytes]:
        epsilon = float(point["epsilon"])
        acf_epsilon = float(point.get("acf_epsilon", -1.0))
        minimum_correlation = float(point.get("min_correlation", -1.0))
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if source.ndim != 3:
            raise ValueError(f"QPET correlation expects [batch,channel,time], got {source.shape}")
        if source.shape[-1] < 3 or self.max_lag > source.shape[-1] - 2:
            raise ValueError("QPET max_lag must not exceed time-2")
        if not np.isfinite(source).all():
            raise ValueError("QPET correlation input contains NaN or infinity")
        if epsilon <= 0 or (acf_epsilon < 0 and minimum_correlation < 0):
            raise ValueError("invalid QPET joint-constraint operating point")
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix="neural_compression_qpet_correlation_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "stream.qpc"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            command = [
                wsl,
                "--exec",
                _wsl_path(self.binary),
                "compress",
                _wsl_path(input_path),
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
                str(source.shape[0]),
                str(source.shape[1]),
                str(source.shape[2]),
                f"{epsilon:.17g}",
                f"{acf_epsilon:.17g}",
                f"{minimum_correlation:.17g}",
                str(self.max_lag),
                str(self.qoz_level),
                self.correction_mode,
                ",".join(f"{value:.17g}" for value in self.estimator_scales),
                f"{self.qpet_confidence:.17g}",
                f"{self.qpet_k:.17g}",
                str(self.group_size),
                (
                    ",".join(str(value) for value in self.repair_denominators)
                    if self.repair_denominators
                    else "none"
                ),
                self.candidate_profile,
                self.repair_optimizer,
                str(self.correction_format_ceiling),
            ]
            if self.base_frontier_is_explicit:
                command.extend(
                    (
                        self.base_frontier,
                        (
                            "none"
                            if self.relaxed_bound_ratio_ceiling == "none"
                            else f"{float(self.relaxed_bound_ratio_ceiling):g}"
                        ),
                    )
                )
            if self.surrogate_frontier_is_explicit:
                if not self.base_frontier_is_explicit:
                    command.extend(("incumbent_only", "none"))
                command.extend(
                    (
                        self.surrogate_frontier,
                        (
                            "none"
                            if self.cfsr_bound_ratio_ceiling == "none"
                            else f"{float(self.cfsr_bound_ratio_ceiling):g}"
                        ),
                    )
                )
            process = _run_native(command, timeout=1800)
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown native error"
                raise RuntimeError(f"QPET correlation backend failed: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype=np.float32)
            if reconstructed.size != source.size:
                raise RuntimeError(
                    f"QPET correlation decoded {reconstructed.size} values, expected {source.size}"
                )
            reconstructed = reconstructed.reshape(source.shape)
            payload = stream_path.read_bytes()

        metadata = _qpet_stream_metadata(payload)
        allowed_versions = (
            {12, 13, 14, 15, 16}
            if self.repair_optimizer in {"ssbb_value", "ssbb"}
            else {12}
        )
        if metadata["version"] not in allowed_versions:
            raise RuntimeError(
                "QPET correlation backend emitted an incompatible stream version"
            )
        required_stream_version = (
            16
            if metadata["predictive_packed_correction_groups"] > 0
            else 15
            if metadata["subset_rank_packed_correction_groups"] > 0
            else 14
            if metadata["factorized_packed_correction_groups"] > 0
            else 13
            if metadata["adaptive_packed_correction_groups"] > 0
            else 12
        )
        if metadata["version"] != required_stream_version:
            raise RuntimeError(
                "QPET stream version and correction-format usage disagree"
            )
        native_fields = {
            key: value
            for token in process.stdout.split()
            if "=" in token
            for key, value in [token.split("=", 1)]
        }
        ssbb_native_fields = (
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
        cjrf_native_fields = (
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
        cfsr_native_fields = (
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
        required_native_fields = {
            "corrections", "online_corrections", "candidates", "base_bytes",
            "correction_bytes", "strict_constant_groups", "max_abs", "max_acf",
            "min_corr", "packed_quantized_groups", "adaptive_packed_groups",
            "factorized_packed_groups",
            "subset_rank_packed_groups",
            "predictive_packed_groups",
            "shared_base_search_groups",
            "candidate_failures", "candidate_profile",
            "repair_optimizer", "correction_format_ceiling",
            "zrcar_base_records", "zrcar_states_generated",
            "zrcar_legacy_seeds", "zrcar_fvro_seeds",
            "zrcar_infeasible_states", "zrcar_target_states_pruned",
            "zrcar_infeasible_frontier_expansions",
            "zrcar_infeasible_frontier_recoveries", "zrcar_candidates",
            "zrcar_prefix_candidates", "zrcar_local_candidates",
            "zrcar_selected_groups", "zrcar_saved_bytes",
            "zrcar_from_fvro_seed_groups",
            "fvro_base_records", "fvro_global_generated",
            "fvro_global_infeasible", "fvro_global_serialized",
            "fvro_channel_generated", "fvro_channel_infeasible",
            "fvro_channel_serialized", "fvro_local_generated",
            "fvro_local_infeasible", "fvro_selected_channels",
            "fvro_strict_channel_steps", "fvro_seeds_retained",
            "fvro_reference_quantized_groups", "fvro_final_quantized_groups",
            "fvro_reference_format4_groups", "fvro_reference_format6_groups",
            "fvro_final_format4_groups", "fvro_final_format6_groups",
            "fvro_final_format7_groups",
            "fvro_final_format8_groups",
            "fvro_final_format9_groups",
            "fvro_final_format10_groups",
            "fvro_reference_exact_escapes", "fvro_final_exact_escapes",
            "fvro_reference_format4_exact_escapes",
            "fvro_reference_format6_exact_escapes",
            "fvro_final_format4_exact_escapes",
            "fvro_final_format6_exact_escapes",
            "fvro_final_format7_exact_escapes",
            "fvro_final_format8_exact_escapes",
            "fvro_final_format9_exact_escapes",
            "fvro_final_format10_exact_escapes",
            "fvro_ref6_to_final6_groups", "fvro_ref6_to_non6_groups",
            "fvro_ref6_to_final6_reference_exact_escapes",
            "fvro_ref6_to_final6_final_exact_escapes",
            "fvro_ref6_to_non6_reference_exact_escapes",
            "fvro_selected_groups", "fvro_saved_bytes",
            "second_stage_selected_groups", "second_stage_saved_bytes",
        }
        if "adaptive_packed_groups" not in native_fields:
            if self.repair_optimizer in {"ssbb_value", "ssbb"}:
                raise RuntimeError(
                    "QPET SSBB backend omitted adaptive-format accounting"
                )
            # Frozen pre-v13 binaries cannot emit format 7 and therefore have
            # an exact, unambiguous zero for this newly added counter.
            native_fields["adaptive_packed_groups"] = "0"
        format7_diagnostic_fields = (
            "fvro_final_format7_groups",
            "fvro_final_format7_exact_escapes",
        )
        missing_format7_diagnostics = {
            field for field in format7_diagnostic_fields if field not in native_fields
        }
        if missing_format7_diagnostics:
            if self.repair_optimizer in {"ssbb_value", "ssbb"}:
                raise RuntimeError(
                    "QPET SSBB backend omitted format-7 escape attribution"
                )
            # The pre-v13 production binaries cannot emit format 7.
            native_fields.update(
                {field: "0" for field in missing_format7_diagnostics}
            )
        format8_diagnostic_fields = (
            "factorized_packed_groups",
            "fvro_final_format8_groups",
            "fvro_final_format8_exact_escapes",
        )
        missing_format8_diagnostics = {
            field for field in format8_diagnostic_fields if field not in native_fields
        }
        if missing_format8_diagnostics:
            if self.repair_optimizer in {"ssbb_value", "ssbb"}:
                raise RuntimeError(
                    "QPET SSBB backend omitted format-8 accounting"
                )
            native_fields.update(
                {field: "0" for field in missing_format8_diagnostics}
            )
        format9_diagnostic_fields = (
            "subset_rank_packed_groups",
            "fvro_final_format9_groups",
            "fvro_final_format9_exact_escapes",
        )
        missing_format9_diagnostics = {
            field for field in format9_diagnostic_fields if field not in native_fields
        }
        if missing_format9_diagnostics:
            present_format9_diagnostics = set(format9_diagnostic_fields).intersection(
                native_fields
            )
            if present_format9_diagnostics:
                raise RuntimeError(
                    "QPET backend emitted a partial format-9 accounting family"
                )
            if (
                self.repair_optimizer in {"ssbb_value", "ssbb"}
                and self.correction_format_ceiling >= 9
            ):
                raise RuntimeError(
                    "QPET ceiling-9 SSBB backend omitted format-9 accounting"
                )
            # Frozen ceiling-8 and non-SSBB binaries cannot emit format 9.
            native_fields.update(
                {field: "0" for field in missing_format9_diagnostics}
            )
        format10_diagnostic_fields = (
            "predictive_packed_groups",
            "fvro_final_format10_groups",
            "fvro_final_format10_exact_escapes",
        )
        missing_format10_diagnostics = {
            field for field in format10_diagnostic_fields if field not in native_fields
        }
        if missing_format10_diagnostics:
            present_format10_diagnostics = set(format10_diagnostic_fields).intersection(
                native_fields
            )
            if present_format10_diagnostics:
                raise RuntimeError(
                    "QPET backend emitted a partial format-10 accounting family"
                )
            if (
                self.repair_optimizer in {"ssbb_value", "ssbb"}
                and self.correction_format_ceiling >= 10
            ):
                raise RuntimeError(
                    "QPET ceiling-10 SSBB backend omitted format-10 accounting"
                )
            native_fields.update(
                {field: "0" for field in missing_format10_diagnostics}
            )
        if "QPET_META" not in process.stdout or not required_native_fields.issubset(
            native_fields
        ):
            raise RuntimeError("QPET backend omitted required accounting metadata")
        present_ssbb_fields = set(ssbb_native_fields).intersection(native_fields)
        all_ssbb_fields = set(ssbb_native_fields)
        if self.repair_optimizer in {"ssbb_value", "ssbb"}:
            if present_ssbb_fields != all_ssbb_fields:
                raise RuntimeError("QPET SSBB backend omitted required accounting metadata")
        elif present_ssbb_fields and present_ssbb_fields != all_ssbb_fields:
            raise RuntimeError("QPET backend emitted a partial SSBB diagnostic family")
        elif not present_ssbb_fields:
            # Compatibility with the frozen A628 FVRO-v1 binary. A partial
            # family is rejected above; only the complete absence is legacy.
            native_fields.update({field: "0" for field in ssbb_native_fields})
        cjrf_protocol_fields = {
            "base_frontier",
            "cjrf_ratio_mask",
            *cjrf_native_fields,
        }
        present_cjrf_fields = cjrf_protocol_fields.intersection(native_fields)
        if present_cjrf_fields and present_cjrf_fields != cjrf_protocol_fields:
            missing = ", ".join(sorted(cjrf_protocol_fields - present_cjrf_fields))
            raise RuntimeError(
                "QPET backend emitted a partial CJRF diagnostic family; "
                f"missing {missing}"
            )
        if not present_cjrf_fields:
            if self.base_frontier_is_explicit:
                raise RuntimeError(
                    "QPET explicitly configured base frontier omitted CJRF metadata"
                )
            # Complete absence is the only compatibility path for binaries and
            # result adapters frozen before the encoder-only base frontier was
            # introduced.
            native_fields["base_frontier"] = "incumbent_only"
            native_fields["cjrf_ratio_mask"] = "0"
            native_fields.update({field: "0" for field in cjrf_native_fields})
        try:
            native_cjrf_ratio_mask = int(native_fields["cjrf_ratio_mask"])
        except ValueError as error:
            raise RuntimeError(
                "QPET backend emitted a non-integer CJRF ratio mask"
            ) from error
        native_relaxed_values: dict[int, float | str] = {
            0: "none",
            1: 1.25,
            3: 1.5,
            7: 2.0,
        }
        if native_cjrf_ratio_mask not in native_relaxed_values:
            raise RuntimeError(
                "QPET backend emitted an invalid CJRF ratio mask"
            )
        native_relaxed_ceiling = native_relaxed_values[native_cjrf_ratio_mask]
        cfsr_protocol_fields = {
            "surrogate_frontier",
            "cfsr_ratio_mask",
            *cfsr_native_fields,
        }
        present_cfsr_fields = cfsr_protocol_fields.intersection(native_fields)
        if present_cfsr_fields and present_cfsr_fields != cfsr_protocol_fields:
            missing = ", ".join(
                sorted(cfsr_protocol_fields - present_cfsr_fields)
            )
            raise RuntimeError(
                "QPET backend emitted a partial CFSR diagnostic family; "
                f"missing {missing}"
            )
        if not present_cfsr_fields:
            if self.surrogate_frontier_is_explicit:
                raise RuntimeError(
                    "QPET explicitly configured surrogate frontier omitted "
                    "CFSR metadata"
                )
            # A complete absence is the compatibility path for binaries
            # published before certified feasible-surrogate re-encoding.
            native_fields["surrogate_frontier"] = "none"
            native_fields["cfsr_ratio_mask"] = "0"
            native_fields.update({field: "0" for field in cfsr_native_fields})
        try:
            native_cfsr_ratio_mask = int(native_fields["cfsr_ratio_mask"])
        except ValueError as error:
            raise RuntimeError(
                "QPET backend emitted a non-integer CFSR ratio mask"
            ) from error
        native_cfsr_values: dict[int, float | str] = {
            0: "none",
            1: 0.25,
            3: 0.5,
            7: 1.0,
        }
        if native_cfsr_ratio_mask not in native_cfsr_values:
            raise RuntimeError(
                "QPET backend emitted an invalid CFSR ratio mask"
            )
        native_cfsr_ceiling = native_cfsr_values[native_cfsr_ratio_mask]
        cpp_max_abs = float(native_fields["max_abs"])
        cpp_max_acf = float(native_fields["max_acf"])
        cpp_min_correlation = float(native_fields["min_corr"])
        for field, value in (
            ("max_abs", cpp_max_abs),
            ("max_acf", cpp_max_acf),
            ("min_corr", cpp_min_correlation),
        ):
            if not np.isfinite(value):
                raise RuntimeError(f"QPET backend reported a non-finite {field}")
        if (
            int(native_fields["corrections"]) != metadata["corrections"]
            or int(native_fields["base_bytes"]) != metadata["base_payload_bytes"]
            or int(native_fields["correction_bytes"])
            != metadata["correction_payload_bytes"]
            or int(native_fields["strict_constant_groups"])
            != metadata["semantic_strict_constant_groups"]
            or int(native_fields["packed_quantized_groups"])
            != metadata["packed_quantized_correction_groups"]
            or int(native_fields["adaptive_packed_groups"])
            != metadata["adaptive_packed_correction_groups"]
            or int(native_fields["factorized_packed_groups"])
            != metadata["factorized_packed_correction_groups"]
            or int(native_fields["subset_rank_packed_groups"])
            != metadata["subset_rank_packed_correction_groups"]
            or int(native_fields["predictive_packed_groups"])
            != metadata["predictive_packed_correction_groups"]
        ):
            raise RuntimeError("QPET stdout accounting disagrees with its stream bytes")
        metadata["candidates_tested"] = int(native_fields["candidates"])
        metadata["online_corrections"] = int(native_fields["online_corrections"])
        metadata["shared_base_search_groups"] = int(
            native_fields["shared_base_search_groups"]
        )
        metadata["candidate_failures"] = int(native_fields["candidate_failures"])
        metadata["candidate_profile"] = native_fields["candidate_profile"]
        metadata["repair_optimizer"] = native_fields["repair_optimizer"]
        metadata["correction_format_ceiling"] = int(
            native_fields["correction_format_ceiling"]
        )
        metadata["base_frontier"] = native_fields["base_frontier"]
        metadata["relaxed_bound_ratio_ceiling"] = native_relaxed_ceiling
        metadata["cjrf_ratio_mask"] = native_cjrf_ratio_mask
        metadata["surrogate_frontier"] = native_fields[
            "surrogate_frontier"
        ]
        metadata["cfsr_bound_ratio_ceiling"] = native_cfsr_ceiling
        metadata["cfsr_ratio_mask"] = native_cfsr_ratio_mask
        for field in (
            "zrcar_base_records",
            "zrcar_legacy_seeds",
            "zrcar_fvro_seeds",
            "zrcar_states_generated",
            "zrcar_infeasible_states",
            "zrcar_target_states_pruned",
            "zrcar_infeasible_frontier_expansions",
            "zrcar_infeasible_frontier_recoveries",
            "zrcar_candidates",
            "zrcar_prefix_candidates",
            "zrcar_local_candidates",
            "zrcar_selected_groups",
            "zrcar_saved_bytes",
            "zrcar_from_fvro_seed_groups",
            "fvro_base_records",
            "fvro_global_generated",
            "fvro_global_infeasible",
            "fvro_global_serialized",
            "fvro_channel_generated",
            "fvro_channel_infeasible",
            "fvro_channel_serialized",
            "fvro_local_generated",
            "fvro_local_infeasible",
            "fvro_selected_channels",
            "fvro_strict_channel_steps",
            "fvro_seeds_retained",
            "fvro_reference_quantized_groups",
            "fvro_final_quantized_groups",
            "fvro_reference_format4_groups",
            "fvro_reference_format6_groups",
            "fvro_final_format4_groups",
            "fvro_final_format6_groups",
            "fvro_final_format7_groups",
            "fvro_final_format8_groups",
            "fvro_final_format9_groups",
            "fvro_final_format10_groups",
            "fvro_reference_exact_escapes",
            "fvro_final_exact_escapes",
            "fvro_reference_format4_exact_escapes",
            "fvro_reference_format6_exact_escapes",
            "fvro_final_format4_exact_escapes",
            "fvro_final_format6_exact_escapes",
            "fvro_final_format7_exact_escapes",
            "fvro_final_format8_exact_escapes",
            "fvro_final_format9_exact_escapes",
            "fvro_final_format10_exact_escapes",
            "fvro_ref6_to_final6_groups",
            "fvro_ref6_to_non6_groups",
            "fvro_ref6_to_final6_reference_exact_escapes",
            "fvro_ref6_to_final6_final_exact_escapes",
            "fvro_ref6_to_non6_reference_exact_escapes",
            "fvro_selected_groups",
            "fvro_saved_bytes",
            "second_stage_selected_groups",
            "second_stage_saved_bytes",
            *ssbb_native_fields,
            *cjrf_native_fields,
            *cfsr_native_fields,
        ):
            metadata[field] = int(native_fields[field])
        if metadata["candidate_failures"] < 0:
            raise RuntimeError("QPET backend reported a negative candidate failure count")
        if metadata["candidate_profile"] != self.candidate_profile:
            raise RuntimeError("QPET backend used a different candidate profile")
        if metadata["repair_optimizer"] != self.repair_optimizer:
            raise RuntimeError("QPET backend used a different repair optimizer")
        if metadata["correction_format_ceiling"] != self.correction_format_ceiling:
            raise RuntimeError("QPET backend used a different correction-format ceiling")
        if metadata["base_frontier"] != self.base_frontier:
            raise RuntimeError("QPET backend used a different base frontier")
        if (
            metadata["relaxed_bound_ratio_ceiling"]
            != self.relaxed_bound_ratio_ceiling
            or metadata["cjrf_ratio_mask"] != self.cjrf_ratio_mask
        ):
            raise RuntimeError(
                "QPET backend used a different relaxed-bound ratio ceiling"
            )
        if metadata["surrogate_frontier"] != self.surrogate_frontier:
            raise RuntimeError(
                "QPET backend used a different surrogate frontier"
            )
        if (
            metadata["cfsr_bound_ratio_ceiling"]
            != self.cfsr_bound_ratio_ceiling
            or metadata["cfsr_ratio_mask"] != self.cfsr_ratio_mask
        ):
            raise RuntimeError(
                "QPET backend used a different CFSR bound-ratio ceiling"
            )
        cjrf_count_fields = tuple(cjrf_native_fields)
        if any(metadata[field] < 0 for field in cjrf_count_fields):
            raise RuntimeError("QPET backend reported a negative CJRF diagnostic")
        cjrf_attempts = metadata["cjrf_relaxed_base_attempts"]
        cjrf_failures = metadata["cjrf_relaxed_base_failures"]
        cjrf_pruned = metadata["cjrf_relaxed_base_pruned"]
        cjrf_duplicates = metadata["cjrf_relaxed_base_duplicates"]
        cjrf_records = metadata["cjrf_relaxed_base_records"]
        cjrf_outside = metadata["cjrf_relaxed_base_outside_pointwise"]
        cjrf_repaired = metadata["cjrf_relaxed_legacy_repaired"]
        cjrf_repair_candidates = metadata[
            "cjrf_relaxed_legacy_repair_candidates"
        ]
        cjrf_selected = metadata["cjrf_selected_groups"]
        cjrf_saved = metadata["cjrf_saved_bytes"]
        cjrf_winner_groups = sum(
            metadata[field]
            for field in (
                "cjrf_legacy_winner_groups",
                "cjrf_fvro_winner_groups",
                "cjrf_zrcar_winner_groups",
                "cjrf_ssbb_winner_groups",
            )
        )
        if (
            cjrf_attempts
            != cjrf_failures + cjrf_pruned + cjrf_duplicates + cjrf_records
            or cjrf_records != cjrf_repaired
            or cjrf_repair_candidates < cjrf_repaired
            or cjrf_outside > cjrf_records
            or cjrf_winner_groups != cjrf_selected
            or cjrf_selected > int(metadata["batch"])
            or (cjrf_selected == 0) != (cjrf_saved == 0)
            or cjrf_failures > metadata["candidate_failures"]
        ):
            raise RuntimeError("QPET backend reported inconsistent CJRF accounting")
        if self.base_frontier == "incumbent_only":
            if any(metadata[field] != 0 for field in cjrf_count_fields):
                raise RuntimeError(
                    "QPET incumbent-only frontier emitted nonzero CJRF diagnostics"
                )
        else:
            ratio_count = {1: 1, 3: 2, 7: 3}[self.cjrf_ratio_mask]
            arms_per_shell = 3 if source.shape[1] == 1 else 4
            expected_attempts = source.shape[0] * ratio_count * arms_per_shell
            if cjrf_attempts != expected_attempts:
                raise RuntimeError(
                    "QPET backend did not execute the complete fixed CJRF shell"
                )
        cfsr_count_fields = tuple(cfsr_native_fields)
        if any(metadata[field] < 0 for field in cfsr_count_fields):
            raise RuntimeError("QPET backend reported a negative CFSR diagnostic")
        cfsr_attempts = metadata["cfsr_surrogate_base_attempts"]
        cfsr_failures = metadata["cfsr_surrogate_base_failures"]
        cfsr_pruned = metadata["cfsr_surrogate_base_pruned"]
        cfsr_duplicates = metadata["cfsr_surrogate_base_duplicates"]
        cfsr_records = metadata["cfsr_surrogate_base_records"]
        cfsr_outside = metadata[
            "cfsr_surrogate_base_outside_original_pointwise"
        ]
        cfsr_repaired = metadata["cfsr_surrogate_legacy_repaired"]
        cfsr_repair_candidates = metadata[
            "cfsr_surrogate_legacy_repair_candidates"
        ]
        cfsr_selected = metadata["cfsr_selected_groups"]
        cfsr_saved = metadata["cfsr_saved_bytes"]
        cfsr_sources = metadata["cfsr_surrogate_sources"]
        cfsr_winner_groups = sum(
            metadata[field]
            for field in (
                "cfsr_legacy_winner_groups",
                "cfsr_fvro_winner_groups",
                "cfsr_zrcar_winner_groups",
                "cfsr_ssbb_winner_groups",
            )
        )
        cfsr_mode_winner_groups = sum(
            metadata[field]
            for field in (
                "cfsr_mode1_winner_groups",
                "cfsr_mode3_winner_groups",
                "cfsr_mode4_winner_groups",
                "cfsr_mode5_winner_groups",
                "cfsr_mode6_winner_groups",
            )
        )
        if (
            cfsr_attempts
            != cfsr_failures + cfsr_pruned + cfsr_duplicates + cfsr_records
            or cfsr_records != cfsr_repaired
            or cfsr_repair_candidates < cfsr_repaired
            or cfsr_outside > cfsr_records
            or cfsr_winner_groups != cfsr_selected
            or cfsr_mode_winner_groups != cfsr_selected
            or cfsr_selected > int(metadata["batch"])
            or (cfsr_selected == 0) != (cfsr_saved == 0)
            or cfsr_failures > metadata["candidate_failures"]
            or (
                source.shape[1] == 1
                and (
                    metadata["cfsr_mode1_winner_groups"] != 0
                    or metadata["cfsr_mode3_winner_groups"] != 0
                )
            )
        ):
            raise RuntimeError("QPET backend reported inconsistent CFSR accounting")
        if self.surrogate_frontier == "none":
            if any(metadata[field] != 0 for field in cfsr_count_fields):
                raise RuntimeError(
                    "QPET disabled surrogate frontier emitted nonzero CFSR diagnostics"
                )
        else:
            shell_count = {1: 1, 3: 2, 7: 3}[self.cfsr_ratio_mask]
            # Both official 2D HPEZ mode 1 and SPERR mode 3 are undefined for
            # a single-channel tensor; C=1 schedules only Serf modes 5 and 6.
            official_arms = 2 if source.shape[1] == 1 else 4
            expected_attempts = source.shape[0] * (
                1 + shell_count * official_arms
            )
            if cfsr_attempts != expected_attempts:
                raise RuntimeError(
                    "QPET backend did not execute the complete fixed CFSR shell"
                )
            if cfsr_sources != int(metadata["batch"]):
                raise RuntimeError(
                    "QPET backend did not certify exactly one CFSR surrogate "
                    "source per window"
                )
        maximum_second_stage_base_records = (
            int(metadata["batch"]) * 4 + cjrf_records + cfsr_records
        )
        if (
            metadata["adaptive_packed_correction_groups"] > 0
            and self.correction_format_ceiling < 7
        ) or (
            metadata["factorized_packed_correction_groups"] > 0
            and self.correction_format_ceiling < 8
        ) or (
            metadata["subset_rank_packed_correction_groups"] > 0
            and self.correction_format_ceiling < 9
        ) or (
            metadata["predictive_packed_correction_groups"] > 0
            and self.correction_format_ceiling < 10
        ):
            raise RuntimeError(
                "QPET stream used a correction format above the configured ceiling"
            )
        zrcar_count_fields = (
            "zrcar_base_records",
            "zrcar_legacy_seeds",
            "zrcar_fvro_seeds",
            "zrcar_states_generated",
            "zrcar_infeasible_states",
            "zrcar_target_states_pruned",
            "zrcar_infeasible_frontier_expansions",
            "zrcar_infeasible_frontier_recoveries",
            "zrcar_candidates",
            "zrcar_prefix_candidates",
            "zrcar_local_candidates",
            "zrcar_selected_groups",
            "zrcar_saved_bytes",
            "zrcar_from_fvro_seed_groups",
        )
        if any(metadata[field] < 0 for field in zrcar_count_fields):
            raise RuntimeError("QPET backend reported a negative ZRCAR diagnostic")
        if (
            metadata["zrcar_prefix_candidates"]
            + metadata["zrcar_local_candidates"]
            != metadata["zrcar_candidates"]
        ):
            raise RuntimeError("QPET backend reported inconsistent ZRCAR candidate counts")
        if (
            metadata["zrcar_target_states_pruned"]
            != metadata["zrcar_infeasible_states"]
            or metadata["zrcar_states_generated"]
            != metadata["zrcar_candidates"] + metadata["zrcar_infeasible_states"]
        ):
            raise RuntimeError("QPET backend reported inconsistent ZRCAR state accounting")
        if (
            metadata["zrcar_infeasible_frontier_expansions"]
            > metadata["zrcar_infeasible_states"]
            or metadata["zrcar_infeasible_frontier_recoveries"]
            > metadata["zrcar_candidates"]
        ):
            raise RuntimeError("QPET backend reported invalid ZRCAR frontier accounting")
        selected_zrcar_seeds = (
            metadata["zrcar_legacy_seeds"] + metadata["zrcar_fvro_seeds"]
        )
        if (
            metadata["zrcar_base_records"] > maximum_second_stage_base_records
            or metadata["zrcar_legacy_seeds"]
            > metadata["zrcar_base_records"] * 2
            or metadata["zrcar_fvro_seeds"]
            > metadata["zrcar_base_records"]
            * (1 if self.repair_optimizer in {"fvro", "ssbb"} else 0)
            or metadata["zrcar_candidates"] > selected_zrcar_seeds * 16
            or metadata["zrcar_states_generated"] > selected_zrcar_seeds * 106
            or metadata["zrcar_infeasible_frontier_expansions"]
            > selected_zrcar_seeds * 8
        ):
            raise RuntimeError("QPET backend exceeded a fixed ZRCAR search budget")
        native_record_groups = sum(
            int(metadata[field])
            for field in (
                "variable_eb_groups",
                "global_eb_groups",
                "sperr_groups",
                "lossless_zstd_groups",
                "serf_xor_groups",
                "serf_qt_groups",
                "serf_smooth_groups",
                "semantic_independent_groups",
                "semantic_shared_groups",
            )
        )
        if metadata["zrcar_selected_groups"] > native_record_groups:
            raise RuntimeError("QPET backend selected more ZRCAR groups than stream records")
        if (metadata["zrcar_selected_groups"] == 0) != (
            metadata["zrcar_saved_bytes"] == 0
        ):
            raise RuntimeError("QPET backend reported inconsistent ZRCAR byte savings")
        if metadata["zrcar_from_fvro_seed_groups"] > metadata["zrcar_selected_groups"]:
            raise RuntimeError("QPET backend reported invalid FVRO-to-ZRCAR provenance")
        if self.repair_optimizer not in {"zrcar", "fvro", "ssbb"} and any(
            metadata[field] != 0 for field in zrcar_count_fields
        ):
            raise RuntimeError("QPET optimizer unexpectedly executed ZRCAR search")
        fvro_count_fields = (
            "fvro_base_records",
            "fvro_global_generated",
            "fvro_global_infeasible",
            "fvro_global_serialized",
            "fvro_channel_generated",
            "fvro_channel_infeasible",
            "fvro_channel_serialized",
            "fvro_local_generated",
            "fvro_local_infeasible",
            "fvro_selected_channels",
            "fvro_strict_channel_steps",
            "fvro_seeds_retained",
            "fvro_reference_quantized_groups",
            "fvro_final_quantized_groups",
            "fvro_reference_format4_groups",
            "fvro_reference_format6_groups",
            "fvro_final_format4_groups",
            "fvro_final_format6_groups",
            "fvro_final_format7_groups",
            "fvro_final_format8_groups",
            "fvro_final_format9_groups",
            "fvro_final_format10_groups",
            "fvro_reference_exact_escapes",
            "fvro_final_exact_escapes",
            "fvro_reference_format4_exact_escapes",
            "fvro_reference_format6_exact_escapes",
            "fvro_final_format4_exact_escapes",
            "fvro_final_format6_exact_escapes",
            "fvro_final_format7_exact_escapes",
            "fvro_final_format8_exact_escapes",
            "fvro_final_format9_exact_escapes",
            "fvro_final_format10_exact_escapes",
            "fvro_ref6_to_final6_groups",
            "fvro_ref6_to_non6_groups",
            "fvro_ref6_to_final6_reference_exact_escapes",
            "fvro_ref6_to_final6_final_exact_escapes",
            "fvro_ref6_to_non6_reference_exact_escapes",
            "fvro_selected_groups",
            "fvro_saved_bytes",
        )
        if any(metadata[field] < 0 for field in fvro_count_fields):
            raise RuntimeError("QPET backend reported a negative FVRO diagnostic")
        if (
            metadata["fvro_global_generated"]
            != metadata["fvro_global_serialized"]
            + metadata["fvro_global_infeasible"]
            or metadata["fvro_channel_generated"]
            != metadata["fvro_channel_serialized"]
            + metadata["fvro_channel_infeasible"]
            or metadata["fvro_local_infeasible"]
            > metadata["fvro_local_generated"]
            or metadata["fvro_strict_channel_steps"]
            < metadata["fvro_selected_channels"]
            or metadata["fvro_strict_channel_steps"]
            > metadata["fvro_channel_serialized"]
            or (metadata["fvro_strict_channel_steps"] == 0)
            != (metadata["fvro_selected_channels"] == 0)
        ):
            raise RuntimeError("QPET backend reported inconsistent FVRO accounting")
        fvro_base_records = metadata["fvro_base_records"]
        if (
            fvro_base_records > maximum_second_stage_base_records
            or metadata["fvro_global_generated"] > fvro_base_records * 8
            or metadata["fvro_global_serialized"] > fvro_base_records * 8
            or metadata["fvro_channel_generated"] > fvro_base_records * 12
            or metadata["fvro_channel_serialized"] > fvro_base_records * 12
            or metadata["fvro_local_generated"] > fvro_base_records * 60
            or metadata["fvro_selected_channels"] > fvro_base_records * 4
            or metadata["fvro_seeds_retained"]
            > metadata["fvro_global_serialized"]
            + metadata["fvro_channel_serialized"]
        ):
            raise RuntimeError("QPET backend exceeded a fixed FVRO v1 search budget")
        if metadata["fvro_selected_groups"] > native_record_groups:
            raise RuntimeError("QPET backend selected more FVRO groups than stream records")
        if (
            metadata["fvro_reference_quantized_groups"] > native_record_groups
            or metadata["fvro_final_quantized_groups"] > native_record_groups
            or metadata["fvro_reference_quantized_groups"]
            != metadata["fvro_reference_format4_groups"]
            + metadata["fvro_reference_format6_groups"]
            or metadata["fvro_final_quantized_groups"]
            != metadata["fvro_final_format4_groups"]
            + metadata["fvro_final_format6_groups"]
            + metadata["fvro_final_format7_groups"]
            + metadata["fvro_final_format8_groups"]
            + metadata["fvro_final_format9_groups"]
            + metadata["fvro_final_format10_groups"]
            or metadata["fvro_reference_exact_escapes"]
            != metadata["fvro_reference_format4_exact_escapes"]
            + metadata["fvro_reference_format6_exact_escapes"]
            or metadata["fvro_final_exact_escapes"]
            != metadata["fvro_final_format4_exact_escapes"]
            + metadata["fvro_final_format6_exact_escapes"]
            + metadata["fvro_final_format7_exact_escapes"]
            + metadata["fvro_final_format8_exact_escapes"]
            + metadata["fvro_final_format9_exact_escapes"]
            + metadata["fvro_final_format10_exact_escapes"]
            or metadata["fvro_reference_format6_groups"]
            != metadata["fvro_ref6_to_final6_groups"]
            + metadata["fvro_ref6_to_non6_groups"]
            or metadata["fvro_reference_format6_exact_escapes"]
            != metadata["fvro_ref6_to_final6_reference_exact_escapes"]
            + metadata["fvro_ref6_to_non6_reference_exact_escapes"]
            or metadata["fvro_ref6_to_final6_groups"]
            > metadata["fvro_final_format6_groups"]
            or metadata["fvro_ref6_to_final6_final_exact_escapes"]
            > metadata["fvro_final_format6_exact_escapes"]
            or (
                metadata["fvro_reference_format4_groups"] == 0
                and metadata["fvro_reference_format4_exact_escapes"] != 0
            )
            or (
                metadata["fvro_reference_format6_groups"] == 0
                and metadata["fvro_reference_format6_exact_escapes"] != 0
            )
            or (
                metadata["fvro_final_format4_groups"] == 0
                and metadata["fvro_final_format4_exact_escapes"] != 0
            )
            or (
                metadata["fvro_final_format6_groups"] == 0
                and metadata["fvro_final_format6_exact_escapes"] != 0
            )
            or (
                metadata["fvro_final_format7_groups"] == 0
                and metadata["fvro_final_format7_exact_escapes"] != 0
            )
            or metadata["fvro_final_format7_groups"]
            != metadata["adaptive_packed_correction_groups"]
            or (
                metadata["fvro_final_format8_groups"] == 0
                and metadata["fvro_final_format8_exact_escapes"] != 0
            )
            or metadata["fvro_final_format8_groups"]
            != metadata["factorized_packed_correction_groups"]
            or (
                metadata["fvro_final_format9_groups"] == 0
                and metadata["fvro_final_format9_exact_escapes"] != 0
            )
            or metadata["fvro_final_format9_groups"]
            != metadata["subset_rank_packed_correction_groups"]
            or (
                metadata["fvro_final_format10_groups"] == 0
                and metadata["fvro_final_format10_exact_escapes"] != 0
            )
            or metadata["fvro_final_format10_groups"]
            != metadata["predictive_packed_correction_groups"]
            or (
                metadata["fvro_ref6_to_final6_groups"] == 0
                and (
                    metadata["fvro_ref6_to_final6_reference_exact_escapes"] != 0
                    or metadata["fvro_ref6_to_final6_final_exact_escapes"] != 0
                )
            )
            or (
                metadata["fvro_ref6_to_non6_groups"] == 0
                and metadata["fvro_ref6_to_non6_reference_exact_escapes"] != 0
            )
        ):
            raise RuntimeError("QPET backend reported invalid FVRO escape accounting")
        if (metadata["fvro_selected_groups"] == 0) != (
            metadata["fvro_saved_bytes"] == 0
        ):
            raise RuntimeError("QPET backend reported inconsistent FVRO byte savings")
        ssbb_count_fields = tuple(ssbb_native_fields)
        if any(metadata[field] < 0 for field in ssbb_count_fields):
            raise RuntimeError("QPET backend reported a negative SSBB diagnostic")
        if (
            metadata["ssbb_states_generated"]
            != metadata["ssbb_states_deduplicated"]
            + metadata["ssbb_statistics_feasible_states"]
            + metadata["ssbb_statistics_infeasible_states"]
            or metadata["ssbb_statistics_feasible_states"]
            + metadata["ssbb_statistics_infeasible_states"]
            != metadata["ssbb_feasible_states_retained"]
            + metadata["ssbb_infeasible_states_retained"]
            + metadata["ssbb_states_pruned_by_beam"]
            or metadata["ssbb_terminal_states"]
            != metadata["ssbb_terminal_direct_feasible"]
            + metadata["ssbb_terminal_direct_infeasible"]
        ):
            raise RuntimeError("QPET backend reported inconsistent SSBB partitions")
        # A larger correction-format ceiling runs every complete lower-ceiling
        # SSBB path and retains all of their final winners.  This is what turns
        # the serializer portfolio into a method-level nested ablation even
        # when format-dependent byte rankings change the beam trajectory.
        ceiling_paths = self.correction_format_ceiling - 5
        composition_starts_per_base = (
            (2 if self.repair_optimizer == "ssbb" else 1) * ceiling_paths
        )
        if (
            metadata["ssbb_composition_starts"]
            != metadata["ssbb_base_records"] * composition_starts_per_base
            or metadata["ssbb_representation_candidates_serialized"]
            != metadata["ssbb_composition_starts"]
            or metadata["ssbb_representation_improved_starts"]
            > metadata["ssbb_representation_candidates_serialized"]
            or (metadata["ssbb_representation_improved_starts"] == 0)
            != (metadata["ssbb_representation_gross_saved_bytes"] == 0)
            or metadata["ssbb_composition_transitions_generated"]
            != metadata["ssbb_composition_transitions_deduplicated"]
            + metadata["ssbb_composition_direct_feasible"]
            + metadata["ssbb_composition_direct_infeasible"]
            or metadata["ssbb_composition_direct_feasible"]
            != metadata["ssbb_composition_candidates_serialized"]
            or metadata["ssbb_composition_singletons_serialized"]
            != metadata["ssbb_composition_local_options"]
            or metadata["ssbb_composition_frontier_carries"]
            + metadata["ssbb_composition_direct_feasible"]
            != metadata["ssbb_composition_anchor_retention_events"]
            + metadata["ssbb_composition_format_retention_events"]
            + metadata["ssbb_composition_rate_retention_events"]
            + metadata["ssbb_composition_states_pruned"]
        ):
            raise RuntimeError(
                "QPET backend reported inconsistent SSBB composition partitions"
            )
        if (
            metadata["ssbb_base_records"] > maximum_second_stage_base_records
            or metadata["ssbb_channels_searched"]
            > metadata["ssbb_base_records"]
            * (16 if self.repair_optimizer == "ssbb" else 8)
            * ceiling_paths
            or metadata["ssbb_positions_processed"]
            != metadata["ssbb_channels_searched"] * int(metadata["time"])
            or metadata["ssbb_actions_generated"]
            < metadata["ssbb_positions_processed"]
            or metadata["ssbb_actions_generated"]
            > metadata["ssbb_positions_processed"] * 5
            or metadata["ssbb_feasible_states_retained"]
            + metadata["ssbb_infeasible_states_retained"]
            > metadata["ssbb_positions_processed"] * 64
            or metadata["ssbb_terminal_states"]
            > metadata["ssbb_channels_searched"] * 64
            or metadata["ssbb_outputs_returned"]
            > metadata["ssbb_channels_searched"] * 3
            or metadata["ssbb_candidates_serialized"]
            > metadata["ssbb_outputs_returned"]
            or metadata["ssbb_candidates_serialized"]
            > metadata["ssbb_base_records"]
            * (48 if self.repair_optimizer == "ssbb" else 24)
            * ceiling_paths
            or metadata["ssbb_composition_candidates_serialized"]
            > metadata["ssbb_composition_starts"] * 1368
            or metadata["ssbb_composition_pairs_serialized"]
            > metadata["ssbb_composition_starts"] * 252
            or metadata["ssbb_composition_max_frontier"] > 57
            or metadata["ssbb_seeds_retained"]
            > metadata["ssbb_base_records"]
            * (2 if self.repair_optimizer == "ssbb" else 1)
            * ceiling_paths
            or metadata["ssbb_selected_groups"] > native_record_groups
            or (metadata["ssbb_selected_groups"] == 0)
            != (metadata["ssbb_saved_bytes"] == 0)
            or metadata["ssbb_composition_selected_groups"]
            > metadata["ssbb_selected_groups"]
            or metadata["ssbb_composition_selected_channels"]
            > metadata["ssbb_composition_selected_groups"] * 8
            or (metadata["ssbb_composition_selected_groups"] == 0)
            != (metadata["ssbb_composition_saved_bytes"] == 0)
            or metadata["ssbb_representation_selected_groups"]
            > metadata["ssbb_selected_groups"]
            or metadata["ssbb_representation_selected_groups"]
            > metadata["ssbb_representation_improved_starts"]
            or metadata["ssbb_search_selected_groups"]
            > metadata["ssbb_selected_groups"]
            or (metadata["ssbb_representation_selected_groups"] == 0)
            != (metadata["ssbb_selected_representation_saved_bytes"] == 0)
            or (metadata["ssbb_search_selected_groups"] == 0)
            != (metadata["ssbb_selected_search_saved_bytes"] == 0)
            or metadata["ssbb_incremental_saved_bytes"]
            != metadata["ssbb_selected_representation_saved_bytes"]
            + metadata["ssbb_selected_search_saved_bytes"]
            or (metadata["ssbb_selected_groups"] == 0)
            != (metadata["ssbb_incremental_saved_bytes"] == 0)
            or metadata["ssbb_saved_bytes"]
            < metadata["ssbb_incremental_saved_bytes"]
            or metadata["ssbb_selected_representation_saved_bytes"]
            > metadata["ssbb_representation_gross_saved_bytes"]
        ):
            raise RuntimeError("QPET backend exceeded a fixed SSBB v2 budget")
        if self.base_frontier == "cjrf_v1" and (
            metadata["zrcar_base_records"] != metadata["fvro_base_records"]
            or metadata["fvro_base_records"] != metadata["ssbb_base_records"]
            or metadata["ssbb_base_records"] < cjrf_records
        ):
            raise RuntimeError(
                "QPET CJRF did not run the complete full-SSBB stack per retained base"
            )
        if self.surrogate_frontier == "cfsr_v1" and (
            metadata["zrcar_base_records"] != metadata["fvro_base_records"]
            or metadata["fvro_base_records"] != metadata["ssbb_base_records"]
            or metadata["ssbb_base_records"] < cfsr_records
        ):
            raise RuntimeError(
                "QPET CFSR did not run the complete full-SSBB stack per retained base"
            )
        if metadata["second_stage_selected_groups"] < 0 or metadata[
            "second_stage_saved_bytes"
        ] < 0:
            raise RuntimeError("QPET backend reported a negative second-stage diagnostic")
        if (
            metadata["second_stage_selected_groups"]
            != metadata["zrcar_selected_groups"]
            + metadata["fvro_selected_groups"]
            + metadata["ssbb_selected_groups"]
            or metadata["second_stage_saved_bytes"]
            != metadata["zrcar_saved_bytes"]
            + metadata["fvro_saved_bytes"]
            + metadata["ssbb_saved_bytes"]
            or metadata["second_stage_selected_groups"] > native_record_groups
            or (metadata["second_stage_selected_groups"] == 0)
            != (metadata["second_stage_saved_bytes"] == 0)
        ):
            raise RuntimeError("QPET backend reported inconsistent second-stage attribution")
        if self.repair_optimizer not in {
            "fvro_value",
            "fvro",
            "ssbb_value",
            "ssbb",
        } and any(
            metadata[field] != 0 for field in fvro_count_fields
        ):
            raise RuntimeError("QPET optimizer unexpectedly executed FVRO search")
        if self.repair_optimizer not in {"ssbb_value", "ssbb"} and any(
            metadata[field] != 0 for field in ssbb_count_fields
        ):
            raise RuntimeError("QPET optimizer unexpectedly executed SSBB search")
        if (
            self.repair_optimizer not in {"fvro", "ssbb"}
            and metadata["zrcar_from_fvro_seed_groups"] != 0
        ):
            raise RuntimeError("QPET optimizer reported an impossible composed provenance")
        if self.repair_optimizer == "legacy" and (
            metadata["second_stage_selected_groups"] != 0
            or metadata["second_stage_saved_bytes"] != 0
        ):
            raise RuntimeError("legacy QPET unexpectedly selected a second-stage result")
        if not (
            0
            <= metadata["shared_base_search_groups"]
            <= metadata["semantic_shared_groups"]
        ):
            raise RuntimeError("QPET backend reported an invalid shared-base selection count")
        if self.candidate_profile == "generic" and any(
            int(metadata[field]) != 0
            for field in (
                "semantic_independent_groups",
                "semantic_shared_groups",
                "semantic_strict_constant_groups",
                "shared_base_search_groups",
            )
        ):
            raise RuntimeError("generic QPET stream selected a vehicle-semantic candidate")
        if (metadata["batch"], metadata["channels"], metadata["time"]) != source.shape:
            raise RuntimeError("QPET correlation stream shape does not match its input")
        if not np.isfinite(reconstructed).all():
            raise RuntimeError("QPET correlation decoded a non-finite reconstruction")
        observed_max = float(np.max(np.abs(source.astype(np.float64) - reconstructed)))
        if observed_max > epsilon + 1e-7:
            raise RuntimeError(
                f"QPET violated the requested point-wise bound: {observed_max} > {epsilon}"
            )

        observed_acf = 0.0
        observed_min_correlation = 1.0
        for original_window, reconstructed_window in zip(source, reconstructed):
            for original_series, reconstructed_series in zip(
                original_window, reconstructed_window
            ):
                if acf_epsilon >= 0:
                    deviation = np.abs(
                        _acf_vector(original_series, self.max_lag) -
                        _acf_vector(reconstructed_series, self.max_lag)
                    )
                    observed_acf = max(observed_acf, float(np.max(deviation)))
                if minimum_correlation >= 0:
                    observed_min_correlation = min(
                        observed_min_correlation,
                        _series_correlation(original_series, reconstructed_series),
                    )
        if acf_epsilon >= 0 and observed_acf > acf_epsilon + 1e-7:
            raise RuntimeError(
                f"QPET violated the strict ACF bound: {observed_acf} > {acf_epsilon}"
            )
        if minimum_correlation >= 0 and observed_min_correlation < minimum_correlation - 1e-7:
            raise RuntimeError(
                "QPET violated the per-series Pearson lower bound: "
                f"{observed_min_correlation} < {minimum_correlation}"
            )
        if abs(cpp_max_abs - observed_max) > 1e-6:
            raise RuntimeError(
                "QPET C++/Python point-error verification disagrees: "
                f"{cpp_max_abs} != {observed_max}"
            )
        if acf_epsilon >= 0 and abs(cpp_max_acf - observed_acf) > 2e-6:
            raise RuntimeError(
                "QPET C++/Python ACF verification disagrees: "
                f"{cpp_max_acf} != {observed_acf}"
            )
        if (
            minimum_correlation >= 0
            and abs(cpp_min_correlation - observed_min_correlation) > 2e-6
        ):
            raise RuntimeError(
                "QPET C++/Python Pearson verification disagrees: "
                f"{cpp_min_correlation} != {observed_min_correlation}"
            )
        metadata.update({
            "standardized_max_abs_error": observed_max,
            "strict_acf_max_abs": observed_acf,
            "correlation_series_min": observed_min_correlation,
            "cpp_standardized_max_abs_error": cpp_max_abs,
            "cpp_strict_acf_max_abs": cpp_max_acf,
            "cpp_correlation_series_min": cpp_min_correlation,
        })
        self.last_metadata = metadata
        result = torch.from_numpy(reconstructed).to(device=x.device, dtype=x.dtype)
        return result, payload

    def decompress_payload(self, payload: bytes) -> np.ndarray:
        """Decode one persisted QPC batch with the native outer-stream decoder."""
        metadata = _qpet_stream_metadata(payload)
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(
            prefix="neural_compression_qpet_persisted_decode_"
        ) as directory:
            temp = Path(directory)
            stream_path = temp / "stream.qpc"
            reconstruction_path = temp / "reconstruction.f32"
            stream_path.write_bytes(payload)
            process = _run_native(
                [
                    wsl,
                    "--exec",
                    _wsl_path(self.binary),
                    "decompress",
                    _wsl_path(stream_path),
                    _wsl_path(reconstruction_path),
                ],
                timeout=300,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
                raise RuntimeError(f"persisted QPET stream failed to decode: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype="<f4")
        expected = int(metadata["batch"] * metadata["channels"] * metadata["time"])
        if reconstructed.size != expected or not np.isfinite(reconstructed).all():
            raise RuntimeError("persisted QPET stream decoded invalid float32 values")
        return reconstructed.reshape(
            int(metadata["batch"]), int(metadata["channels"]), int(metadata["time"])
        )


class LFZipNLMSJointCodec(NativeClassicalCodec):
    """Official LFZip-NLMS base followed by the shared deterministic hard repair."""

    method_name = "lfzip_nlms_joint"
    upstream_commit = "14b2e9f79b1d9bc3f60d5cb69e20eb1529c709b4"
    entropy_coder = "official-LFZip-NLMS-libbsc-3.1.0+joint-repair"
    _header = struct.Struct("<4sIIIIIddd")
    _record = struct.Struct("<III")

    def __init__(
        self,
        max_lag: int = 16,
        correction_mode: str = "posthoc_greedy",
        nlms_orders: tuple[int, ...] | list[int] = (0, 32),
        quantization_bytes: tuple[int, ...] | list[int] = (1, 2),
        base_tolerance_scales: tuple[float, ...] | list[float] = (0.5, 1.0, 1.5, 2.0),
        repair_denominators: tuple[int, ...] | list[int] = (7, 15, 31, 63, 127),
        mu: float = 0.5,
    ) -> None:
        super().__init__()
        if max_lag < 1 or correction_mode not in {"posthoc_sequential", "posthoc_greedy"}:
            raise ValueError("invalid LFZip-NLMS joint repair settings")
        orders = tuple(int(value) for value in nlms_orders)
        qbytes = tuple(int(value) for value in quantization_bytes)
        scales = tuple(float(value) for value in base_tolerance_scales)
        denominators = tuple(int(value) for value in repair_denominators)
        if not orders or any(value < 0 for value in orders):
            raise ValueError("LFZip NLMS orders must be nonnegative")
        if not qbytes or any(value not in {1, 2} for value in qbytes):
            raise ValueError("LFZip quantization bytes must be 1 or 2")
        if not scales or any(not np.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("LFZip base tolerance scales must be finite and positive")
        if any(value < 1 or value > 127 for value in denominators):
            raise ValueError("LFZip repair denominators must be in [1,127]")
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("LFZip NLMS mu must be finite and positive")
        root = Path(__file__).resolve().parents[2]
        self.script = root / "tmp" / "upstreams" / "LFZip-official" / "src" / "nlms_compress.py"
        self.repair_binary = root / "tools" / "qpet_correlation" / "bin" / "qpet_correlation_batch"
        self.max_lag = int(max_lag)
        self.correction_mode = correction_mode
        self.nlms_orders = orders
        self.quantization_bytes = qbytes
        self.base_tolerance_scales = scales
        self.repair_denominators = denominators
        self.mu = float(mu)
        self.last_metadata: dict[str, object] = {}

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError("official LFZip-NLMS joint codec requires WSL")
        if not self.script.is_file():
            raise RuntimeError(
                "official LFZip checkout is missing; run "
                "`wsl bash ./tools/lfzip_official/build_wsl.sh`"
            )
        if not self.repair_binary.is_file():
            raise RuntimeError(
                "joint repair backend is missing; run "
                "`wsl bash ./tools/qpet_correlation/build_wsl.sh`"
            )
        return wsl

    def operating_points(self, spec: dict) -> list[dict[str, object]]:
        points: list[dict[str, object]] = []
        for item in spec["operating_points"]:
            point = {
                "epsilon": float(item["epsilon"]),
                "acf_epsilon": float(item.get("acf_epsilon", -1.0)),
                "min_correlation": float(item.get("min_correlation", -1.0)),
            }
            if point["epsilon"] <= 0 or (
                point["acf_epsilon"] < 0 and point["min_correlation"] < 0
            ):
                raise ValueError("invalid LFZip-NLMS joint operating point")
            point["label"] = item.get(
                "label",
                f"epsilon={point['epsilon']:g},acf={point['acf_epsilon']:g},"
                f"corr={point['min_correlation']:g}",
            )
            points.append(point)
        return points

    def _run_official(self, wsl: str, arguments: list[str], timeout: int = 300) -> None:
        process = _run_native(
            [
                wsl,
                "--exec",
                "env",
                "PYTHONHASHSEED=0",
                "python3",
                _wsl_path(self.script),
                *arguments,
            ],
            timeout=timeout,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "unknown official error"
            raise RuntimeError(f"official LFZip-NLMS failed: {detail}")

    @staticmethod
    def _metadata_fields(stdout: str) -> dict[str, str]:
        return {
            key: value
            for token in stdout.split()
            if "=" in token
            for key, value in [token.split("=", 1)]
        }

    def _decode_container(self, payload: bytes, directory: Path, wsl: str) -> np.ndarray:
        if len(payload) < self._header.size:
            raise ValueError("truncated LFZip-NLMS joint header")
        (
            magic, version, batch, channels, time_points, max_lag,
            _, _, _,
        ) = self._header.unpack_from(payload)
        if magic != b"LFC1" or version != 1 or min(batch, channels, time_points) < 1:
            raise ValueError("invalid LFZip-NLMS joint header")
        if max_lag > time_points - 2:
            raise ValueError("invalid LFZip-NLMS joint max lag")
        position = self._header.size
        windows: list[np.ndarray] = []
        for index in range(batch):
            if position + self._record.size > len(payload):
                raise ValueError("truncated LFZip-NLMS joint record")
            correction_count, base_size, correction_size = self._record.unpack_from(
                payload, position
            )
            position += self._record.size
            if base_size < 1 or position + base_size + correction_size > len(payload):
                raise ValueError("truncated LFZip-NLMS joint payload")
            base_stream = payload[position:position + base_size]
            position += base_size
            correction = payload[position:position + correction_size]
            position += correction_size
            base_bsc = directory / f"decode-{index}.bsc"
            base_npy = directory / f"decode-{index}.npy"
            base_f32 = directory / f"decode-{index}.f32"
            correction_path = directory / f"decode-{index}.corr"
            output_f32 = directory / f"decode-{index}-reconstruction.f32"
            base_bsc.write_bytes(base_stream)
            correction_path.write_bytes(correction)
            self._run_official(
                wsl,
                ["-m", "d", "-i", _wsl_path(base_bsc), "-o", _wsl_path(base_npy)],
            )
            base = np.load(base_npy, allow_pickle=False)
            if base.dtype != np.float32 or base.shape != (channels, time_points):
                raise RuntimeError("official LFZip-NLMS decoded an invalid tensor")
            np.ascontiguousarray(base, dtype="<f4").tofile(base_f32)
            process = _run_native(
                [
                    wsl,
                    "--exec",
                    _wsl_path(self.repair_binary),
                    "apply-external",
                    _wsl_path(base_f32),
                    _wsl_path(correction_path),
                    _wsl_path(output_f32),
                    str(correction_count),
                    str(channels),
                    str(time_points),
                ],
                timeout=300,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown repair error"
                raise RuntimeError(f"LFZip-NLMS correction decode failed: {detail}")
            reconstructed = np.fromfile(output_f32, dtype="<f4")
            if reconstructed.size != channels * time_points:
                raise RuntimeError("LFZip-NLMS correction decoded an invalid value count")
            windows.append(reconstructed.reshape(channels, time_points))
        if position != len(payload):
            raise ValueError("LFZip-NLMS joint stream has trailing bytes")
        return np.stack(windows)

    @torch.no_grad()
    def compress_reconstruct_point(
        self, x: torch.Tensor, point: dict[str, object]
    ) -> tuple[torch.Tensor, bytes]:
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4")
        if source.ndim != 3 or not np.isfinite(source).all():
            raise ValueError("LFZip-NLMS joint codec expects finite [batch,channel,time]")
        if source.shape[1] >= 256 or self.max_lag > source.shape[2] - 2:
            raise ValueError("LFZip-NLMS shape or max lag is invalid")
        epsilon = float(point["epsilon"])
        acf_epsilon = float(point.get("acf_epsilon", -1.0))
        minimum_correlation = float(point.get("min_correlation", -1.0))
        wsl = self._validate_backend()
        records: list[bytes] = []
        total_base_bytes = 0
        total_correction_bytes = 0
        total_corrections = 0
        candidates_tested = 0
        selected_orders: list[int] = []
        selected_qbytes: list[int] = []
        selected_scales: list[float] = []
        with tempfile.TemporaryDirectory(prefix="nclab_lfzip_nlms_joint_") as directory:
            temp = Path(directory)
            for window_index, window in enumerate(source):
                original_npy = temp / f"original-{window_index}.npy"
                original_f32 = temp / f"original-{window_index}.f32"
                np.save(original_npy, window, allow_pickle=False)
                window.tofile(original_f32)
                best: tuple[int, bytes, bytes, int, int, int, float] | None = None
                candidate_index = 0
                for scale in self.base_tolerance_scales:
                    for order in self.nlms_orders:
                        for qbytes in self.quantization_bytes:
                            candidate_dir = temp / f"candidate-{window_index}-{candidate_index}"
                            candidate_dir.mkdir()
                            base_bsc = candidate_dir / "base.bsc"
                            decoded_npy = candidate_dir / "base-decoded.npy"
                            base_f32 = candidate_dir / "base.f32"
                            correction_path = candidate_dir / "repair.bin"
                            repaired_f32 = candidate_dir / "repaired.f32"
                            self._run_official(
                                wsl,
                                [
                                    "-m", "c", "-i", _wsl_path(original_npy),
                                    "-o", _wsl_path(base_bsc), "-n", str(order),
                                    "--mu", f"{self.mu:.9g}",
                                    "-a", f"{epsilon * scale:.9g}",
                                    "-q", str(qbytes),
                                ],
                            )
                            self._run_official(
                                wsl,
                                [
                                    "-m", "d", "-i", _wsl_path(base_bsc),
                                    "-o", _wsl_path(decoded_npy),
                                ],
                            )
                            base = np.load(decoded_npy, allow_pickle=False)
                            if base.dtype != np.float32 or base.shape != window.shape:
                                raise RuntimeError("official LFZip-NLMS candidate shape changed")
                            np.ascontiguousarray(base, dtype="<f4").tofile(base_f32)
                            process = _run_native(
                                [
                                    wsl,
                                    "--exec",
                                    _wsl_path(self.repair_binary),
                                    "repair-external",
                                    _wsl_path(original_f32),
                                    _wsl_path(base_f32),
                                    _wsl_path(correction_path),
                                    _wsl_path(repaired_f32),
                                    str(source.shape[1]),
                                    str(source.shape[2]),
                                    f"{epsilon:.17g}",
                                    f"{acf_epsilon:.17g}",
                                    f"{minimum_correlation:.17g}",
                                    str(self.max_lag),
                                    self.correction_mode,
                                    (
                                        ",".join(map(str, self.repair_denominators))
                                        if self.repair_denominators else "none"
                                    ),
                                ],
                                timeout=300,
                            )
                            if process.returncode != 0:
                                detail = process.stderr.strip() or process.stdout.strip() or "unknown repair error"
                                raise RuntimeError(f"LFZip-NLMS candidate repair failed: {detail}")
                            fields = self._metadata_fields(process.stdout)
                            correction_count = int(fields["corrections"])
                            base_bytes = base_bsc.read_bytes()
                            correction_bytes = correction_path.read_bytes()
                            record_size = self._record.size + len(base_bytes) + len(correction_bytes)
                            if best is None or record_size < best[0]:
                                best = (
                                    record_size, base_bytes, correction_bytes,
                                    correction_count, order, qbytes, scale,
                                )
                            candidate_index += 1
                            candidates_tested += 1
                if best is None:
                    raise RuntimeError("LFZip-NLMS produced no candidate")
                (
                    _, base_bytes, correction_bytes, correction_count,
                    selected_order, selected_qbyte, selected_scale,
                ) = best
                records.append(
                    self._record.pack(
                        correction_count, len(base_bytes), len(correction_bytes)
                    ) + base_bytes + correction_bytes
                )
                total_base_bytes += len(base_bytes)
                total_correction_bytes += len(correction_bytes)
                total_corrections += correction_count
                selected_orders.append(selected_order)
                selected_qbytes.append(selected_qbyte)
                selected_scales.append(selected_scale)
            payload = self._header.pack(
                b"LFC1", 1, source.shape[0], source.shape[1], source.shape[2],
                self.max_lag, epsilon, acf_epsilon, minimum_correlation,
            ) + b"".join(records)
            reconstructed = self._decode_container(payload, temp, wsl)

        observed_max = float(np.max(np.abs(source.astype(np.float64) - reconstructed)))
        observed_acf = 0.0
        observed_min_correlation = 1.0
        for original_window, reconstructed_window in zip(source, reconstructed):
            for original_series, reconstructed_series in zip(
                original_window, reconstructed_window
            ):
                if acf_epsilon >= 0:
                    observed_acf = max(
                        observed_acf,
                        float(np.max(np.abs(
                            _acf_vector(original_series, self.max_lag)
                            - _acf_vector(reconstructed_series, self.max_lag)
                        ))),
                    )
                if minimum_correlation >= 0:
                    observed_min_correlation = min(
                        observed_min_correlation,
                        _series_correlation(original_series, reconstructed_series),
                    )
        if observed_max > epsilon + 1e-7:
            raise RuntimeError("LFZip-NLMS joint stream violated the point-wise bound")
        if acf_epsilon >= 0 and observed_acf > acf_epsilon + 1e-7:
            raise RuntimeError("LFZip-NLMS joint stream violated the ACF bound")
        if minimum_correlation >= 0 and observed_min_correlation < minimum_correlation - 1e-7:
            raise RuntimeError("LFZip-NLMS joint stream violated the Pearson bound")
        self.last_metadata = {
            "batch": int(source.shape[0]),
            "corrections": total_corrections,
            "candidates_tested": candidates_tested,
            "base_payload_bytes": total_base_bytes,
            "correction_payload_bytes": total_correction_bytes,
            "standardized_max_abs_error": observed_max,
            "strict_acf_max_abs": observed_acf,
            "correlation_series_min": observed_min_correlation,
            "lfzip_order0_windows": sum(value == 0 for value in selected_orders),
            "lfzip_order32_windows": sum(value == 32 for value in selected_orders),
            "lfzip_qbyte1_windows": sum(value == 1 for value in selected_qbytes),
            "lfzip_qbyte2_windows": sum(value == 2 for value in selected_qbytes),
            "lfzip_base_scale_min": min(selected_scales),
            "lfzip_base_scale_max": max(selected_scales),
        }
        return torch.from_numpy(reconstructed).to(x.device, x.dtype), payload

    @torch.no_grad()
    def compress_reconstruct(
        self, x: torch.Tensor, quality: float | None
    ) -> tuple[torch.Tensor, bytes]:
        raise TypeError("LFZip-NLMS joint codec requires a full operating point")


def _cameo_retained_points(payload: bytes) -> tuple[int, int]:
    header = struct.Struct("<4sIIIIdd")
    if len(payload) < header.size:
        raise ValueError("truncated CAMEO stream header")
    magic, batch, channels, time_points, _, _, _ = header.unpack_from(payload)
    if magic != b"CAM1":
        raise ValueError("invalid CAMEO stream magic")
    position = header.size
    retained = 0
    for _ in range(batch * channels):
        if position + 2 > len(payload):
            raise ValueError("truncated CAMEO point count")
        count = struct.unpack_from("<H", payload, position)[0]
        position += 2
        retained += count
        for _ in range(count):
            while True:
                if position >= len(payload):
                    raise ValueError("truncated CAMEO index stream")
                byte = payload[position]
                position += 1
                if byte < 0x80:
                    break
        position += count * 4
        if position > len(payload):
            raise ValueError("truncated CAMEO value stream")
    if position != len(payload):
        raise ValueError("CAMEO stream has trailing data")
    return retained, batch * channels * time_points


def _acf_vector(series: np.ndarray, max_lag: int) -> np.ndarray:
    values = []
    series = np.asarray(series, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        left, right = series[:-lag], series[lag:]
        left = left - left.mean()
        right = right - right.mean()
        denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        values.append(float(np.dot(left, right) / denominator) if denominator > 1e-15 else 0.0)
    return np.asarray(values)


class CAMEOCodec(NativeClassicalCodec):
    """Official CAMEO ACF-preserving line simplification, independently per channel."""

    method_name = "cameo"
    upstream_commit = "97d146b4d06662107fe6ac5aaf7b24f55a45d862"
    quality_key = "acf_epsilon"
    quality_spec_key = "acf_epsilons"
    entropy_coder = "CAMEO-retained-index-f32"

    def __init__(self, max_lag: int = 16, blocking_factor: float = 10.0) -> None:
        super().__init__()
        if max_lag < 1 or blocking_factor <= 0:
            raise ValueError("CAMEO max_lag and blocking_factor must be positive")
        root = Path(__file__).resolve().parents[2]
        self.adapter = root / "tools" / "cameo" / "cameo_batch.py"
        self.source = root / "tmp" / "upstreams" / "cameo"
        self.max_lag = int(max_lag)
        self.blocking_factor = float(blocking_factor)
        self.last_metadata: dict[str, int] = {}

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError("CAMEO uses its official Cython code and requires WSL on Windows")
        if not self.adapter.is_file():
            raise RuntimeError(f"CAMEO adapter is missing: {self.adapter}")
        if not self.source.is_dir() or not list((self.source / "compression" / "lpc").glob("cameo*.so")):
            raise RuntimeError(
                "CAMEO official extension is missing; run `wsl bash ./tools/cameo/build_wsl.sh`"
            )
        return wsl

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, quality: float | None) -> tuple[torch.Tensor, bytes]:
        if quality is None:
            raise ValueError("CAMEO requires an ACF error bound")
        acf_epsilon = float(quality)
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if source.ndim != 3:
            raise ValueError(f"CAMEO expects [batch, channel, time], got {source.shape}")
        if source.shape[-1] < 4:
            raise ValueError("CAMEO requires at least four time points")
        if not np.isfinite(source).all():
            raise ValueError("CAMEO input contains NaN or infinity")
        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix="neural_compression_cameo_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "streams.cameo"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            command = [
                wsl,
                "--exec",
                "python3",
                _wsl_path(self.adapter),
                _wsl_path(input_path),
                _wsl_path(stream_path),
                _wsl_path(reconstruction_path),
                str(source.shape[0]),
                str(source.shape[1]),
                str(source.shape[2]),
                f"{acf_epsilon:.17g}",
                str(self.max_lag),
                f"{self.blocking_factor:.17g}",
                _wsl_path(self.source),
            ]
            process = _run_native(command, timeout=300)
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown native error"
                raise RuntimeError(f"CAMEO official backend failed: {detail}")
            reconstructed = np.fromfile(reconstruction_path, dtype=np.float32)
            if reconstructed.size != source.size:
                raise RuntimeError(f"CAMEO decoded {reconstructed.size} values, expected {source.size}")
            reconstructed = reconstructed.reshape(source.shape)
            payload = stream_path.read_bytes()

        max_lag = min(self.max_lag, source.shape[-1] - 2)
        deviations = []
        for original_window, reconstructed_window in zip(source, reconstructed):
            for original_series, reconstructed_series in zip(original_window, reconstructed_window):
                deviations.append(float(np.mean(np.abs(
                    _acf_vector(original_series, max_lag) -
                    _acf_vector(reconstructed_series, max_lag)
                ))))
        observed = max(deviations, default=0.0)
        if observed >= acf_epsilon + 1e-7:
            raise RuntimeError(
                f"CAMEO violated the requested mean ACF error bound: {observed} >= {acf_epsilon}"
            )
        retained, logical_values = _cameo_retained_points(payload)
        self.last_metadata = {"retained_points": retained, "logical_values": logical_values}
        result = torch.from_numpy(reconstructed).to(device=x.device, dtype=x.dtype)
        return result, payload


_CAMEO_GUARD_HEADER = struct.Struct("<4sIIIIdddd")


def _decode_cameo_guard_payload(payload: bytes) -> tuple[np.ndarray, dict[str, object]]:
    """Independently decode a complete CMP1/CMJ1 retained-point stream."""
    if len(payload) < _CAMEO_GUARD_HEADER.size:
        raise ValueError("truncated CAMEO guard stream header")
    (
        magic,
        batch,
        channels,
        time_points,
        max_lag,
        epsilon,
        acf_epsilon,
        minimum_correlation,
        blocking_factor,
    ) = _CAMEO_GUARD_HEADER.unpack_from(payload)
    if magic not in {b"CMP1", b"CMJ1"}:
        raise ValueError("invalid CAMEO guard stream magic")
    if min(batch, channels) < 1 or time_points < 4 or not 1 <= max_lag <= time_points - 2:
        raise ValueError("invalid CAMEO guard tensor shape")
    if (
        not np.isfinite([epsilon, acf_epsilon, blocking_factor]).all()
        or epsilon <= 0
        or acf_epsilon <= 0
        or blocking_factor <= 0
    ):
        raise ValueError("invalid CAMEO guard stream parameters")
    if magic == b"CMJ1" and (
        not np.isfinite(minimum_correlation)
        or not 0 < minimum_correlation <= 1
    ):
        raise ValueError("invalid CAMEO joint correlation bound")

    decoded = np.empty((batch, channels, time_points), dtype=np.float32)
    position = _CAMEO_GUARD_HEADER.size
    retained_points = 0
    for batch_index in range(batch):
        for channel_index in range(channels):
            if position + 2 > len(payload):
                raise ValueError("truncated CAMEO guard retained-point count")
            count = struct.unpack_from("<H", payload, position)[0]
            position += 2
            if count < 2 or count > time_points:
                raise ValueError("invalid CAMEO guard retained-point count")
            indices = np.empty(count, dtype=np.int64)
            current = 0
            for item in range(count):
                value = 0
                shift = 0
                while True:
                    if position >= len(payload) or shift > 63:
                        raise ValueError("invalid CAMEO guard index varint")
                    byte = payload[position]
                    position += 1
                    value |= (byte & 0x7F) << shift
                    if byte < 0x80:
                        break
                    shift += 7
                current += value
                indices[item] = current
            value_bytes = count * np.dtype("<f4").itemsize
            if position + value_bytes > len(payload):
                raise ValueError("truncated CAMEO guard retained values")
            values = np.frombuffer(
                payload, dtype="<f4", count=count, offset=position
            ).astype(np.float64)
            position += value_bytes
            if (
                indices[0] != 0
                or indices[-1] != time_points - 1
                or np.any(np.diff(indices) <= 0)
            ):
                raise ValueError("invalid CAMEO guard retained indices")
            decoded[batch_index, channel_index] = np.interp(
                np.arange(time_points), indices, values
            ).astype(np.float32)
            retained_points += count
    if position != len(payload):
        raise ValueError("CAMEO guard stream has trailing bytes")
    return decoded, {
        "magic": magic.decode("ascii"),
        "batch": batch,
        "channels": channels,
        "time": time_points,
        "max_lag": max_lag,
        "epsilon": epsilon,
        "acf_epsilon": acf_epsilon,
        "minimum_correlation": minimum_correlation,
        "blocking_factor": blocking_factor,
        "retained_points": retained_points,
        "logical_values": batch * channels * time_points,
    }


class _CAMEOGuardCodec(NativeClassicalCodec):
    """Shared harness for explicitly local CAMEO deletion-guard extensions."""

    upstream_commit = CAMEOCodec.upstream_commit
    constraint_mode: str

    def __init__(self, max_lag: int = 16, blocking_factor: float = 10.0) -> None:
        super().__init__()
        if max_lag < 1 or blocking_factor <= 0:
            raise ValueError("CAMEO guard max_lag and blocking_factor must be positive")
        root = Path(__file__).resolve().parents[2]
        self.adapter = root / "tools" / "cameo" / "cameo_pointwise_batch.py"
        self.source = root / "tmp" / "upstreams" / "cameo"
        self.max_lag = int(max_lag)
        self.blocking_factor = float(blocking_factor)
        self.last_metadata: dict[str, object] = {}

    def _validate_backend(self) -> str:
        wsl = shutil.which("wsl")
        if wsl is None:
            raise RuntimeError(f"{self.method_name} requires WSL")
        if not self.adapter.is_file():
            raise RuntimeError(f"CAMEO guard adapter is missing: {self.adapter}")
        if not self.source.is_dir() or not list(
            (self.source / "compression" / "lpc").glob("cameo*.so")
        ):
            raise RuntimeError(
                "CAMEO guard extension is missing; run `wsl bash ./tools/cameo/build_wsl.sh`"
            )
        return wsl

    def operating_points(self, spec: dict) -> list[dict[str, object]]:
        points = spec.get("operating_points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{self.method_name} requires non-empty operating_points")
        return [dict(point) for point in points]

    def decompress_payload(self, payload: bytes) -> np.ndarray:
        decoded, metadata = _decode_cameo_guard_payload(payload)
        expected_magic = "CMP1" if self.constraint_mode == "pointguard" else "CMJ1"
        if metadata["magic"] != expected_magic:
            raise ValueError(
                f"{self.method_name} expected {expected_magic}, got {metadata['magic']}"
            )
        return decoded

    @torch.no_grad()
    def compress_reconstruct_point(
        self, x: torch.Tensor, point: dict[str, object]
    ) -> tuple[torch.Tensor, bytes]:
        epsilon = float(point["epsilon"])
        acf_epsilon = float(point["acf_epsilon"])
        minimum_correlation = (
            float(point["min_correlation"])
            if self.constraint_mode == "joint"
            else -1.0
        )
        if not np.isfinite([epsilon, acf_epsilon]).all() or min(epsilon, acf_epsilon) <= 0:
            raise ValueError("CAMEO guard bounds must be finite and positive")
        if self.constraint_mode == "joint" and not 0 < minimum_correlation <= 1:
            raise ValueError("CAMEO joint minimum correlation must be in (0,1]")
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4")
        if source.ndim != 3 or source.shape[-1] < 4:
            raise ValueError(f"{self.method_name} expects [batch,channel,time>=4]")
        if not np.isfinite(source).all():
            raise ValueError(f"{self.method_name} input contains NaN or infinity")

        wsl = self._validate_backend()
        with tempfile.TemporaryDirectory(prefix=f"nclab_{self.method_name}_") as directory:
            temp = Path(directory)
            input_path = temp / "input.f32"
            stream_path = temp / "stream.cameo_guard"
            reconstruction_path = temp / "reconstruction.f32"
            source.tofile(input_path)
            process = _run_native(
                [
                    wsl,
                    "--exec",
                    "python3",
                    _wsl_path(self.adapter),
                    _wsl_path(input_path),
                    _wsl_path(stream_path),
                    _wsl_path(reconstruction_path),
                    str(source.shape[0]),
                    str(source.shape[1]),
                    str(source.shape[2]),
                    f"{epsilon:.17g}",
                    f"{acf_epsilon:.17g}",
                    f"{minimum_correlation:.17g}",
                    str(self.max_lag),
                    f"{self.blocking_factor:.17g}",
                    _wsl_path(self.source),
                    self.constraint_mode,
                ],
                timeout=300,
            )
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
                raise RuntimeError(f"{self.method_name} backend failed: {detail}")
            adapter_reconstruction = np.fromfile(reconstruction_path, dtype="<f4")
            payload = stream_path.read_bytes()

        if adapter_reconstruction.size != source.size:
            raise RuntimeError(f"{self.method_name} decoded an invalid value count")
        adapter_reconstruction = adapter_reconstruction.reshape(source.shape)
        decoded, stream_metadata = _decode_cameo_guard_payload(payload)
        if decoded.shape != source.shape or not np.array_equal(
            adapter_reconstruction.view(np.uint32), decoded.view(np.uint32)
        ):
            raise RuntimeError(f"{self.method_name} independent stream decode disagrees with adapter")
        expected_magic = "CMP1" if self.constraint_mode == "pointguard" else "CMJ1"
        if (
            stream_metadata["magic"] != expected_magic
            or stream_metadata["epsilon"] != epsilon
            or stream_metadata["acf_epsilon"] != acf_epsilon
            or stream_metadata["minimum_correlation"] != minimum_correlation
        ):
            raise RuntimeError(f"{self.method_name} stream header changed its requested bounds")

        observed_point = 0.0
        observed_mean_acf = 0.0
        observed_max_acf = 0.0
        observed_correlation = 1.0
        max_lag = min(self.max_lag, source.shape[-1] - 2)
        for original_window, decoded_window in zip(source, decoded):
            for original_series, decoded_series in zip(original_window, decoded_window):
                observed_point = max(
                    observed_point,
                    float(np.max(np.abs(
                        original_series.astype(np.float64)
                        - decoded_series.astype(np.float64)
                    ))),
                )
                acf_errors = np.abs(
                    _acf_vector(original_series, max_lag)
                    - _acf_vector(decoded_series, max_lag)
                )
                observed_mean_acf = max(observed_mean_acf, float(np.mean(acf_errors)))
                observed_max_acf = max(observed_max_acf, float(np.max(acf_errors)))
                observed_correlation = min(
                    observed_correlation,
                    _series_correlation(original_series, decoded_series),
                )
        if observed_point > epsilon + 1e-7:
            raise RuntimeError(f"{self.method_name} violated its point-wise bound")
        if self.constraint_mode == "pointguard":
            if observed_mean_acf >= acf_epsilon + 1e-7:
                raise RuntimeError(f"{self.method_name} violated official CAMEO mean-ACF bound")
        else:
            if observed_max_acf > acf_epsilon + 1e-7:
                raise RuntimeError(f"{self.method_name} violated its worst-lag ACF bound")
            if observed_correlation < minimum_correlation - 1e-7:
                raise RuntimeError(f"{self.method_name} violated its Pearson bound")

        self.last_metadata = {
            **stream_metadata,
            "constraint_mode": self.constraint_mode,
            "standardized_max_abs_error": observed_point,
            "cameo_mean_acf_max": observed_mean_acf,
            "strict_acf_max_abs": observed_max_acf,
            "correlation_series_min": observed_correlation,
            "decoded_numpy_verified": True,
        }
        return torch.from_numpy(decoded).to(x.device, x.dtype), payload

    @torch.no_grad()
    def compress_reconstruct(
        self, x: torch.Tensor, quality: float | None
    ) -> tuple[torch.Tensor, bytes]:
        raise TypeError(f"{self.method_name} requires a full operating point")


class CAMEOPointGuardCodec(_CAMEOGuardCodec):
    """Local CAMEO extension: deletion-time L-infinity plus official mean-ACF."""

    method_name = "cameo_point_guard"
    constraint_mode = "pointguard"
    entropy_coder = "CAMEO-SIP+local-point-guard-retained-index-f32"


class CAMEOJointGuardCodec(_CAMEOGuardCodec):
    """Fair joint baseline: PointGuard plus decoded worst-ACF and Pearson guards."""

    method_name = "cameo_joint_guard"
    constraint_mode = "joint"
    entropy_coder = "CAMEO-SIP+local-joint-guard-retained-index-f32"


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.accumulator = 0
        self.used = 0

    def write(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or value >= (1 << width):
            raise ValueError(f"cannot write value={value} with width={width}")
        remaining = width
        while remaining:
            take = min(8 - self.used, remaining)
            shift = remaining - take
            chunk = (value >> shift) & ((1 << take) - 1)
            self.accumulator = (self.accumulator << take) | chunk
            self.used += take
            remaining -= take
            if self.used == 8:
                self.output.append(self.accumulator)
                self.accumulator = 0
                self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.output.append(self.accumulator << (8 - self.used))
            self.accumulator = 0
            self.used = 0
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def read(self, width: int) -> int:
        if self.position + width > len(self.payload) * 8:
            raise ValueError("truncated Gorilla bitstream")
        value = 0
        remaining = width
        while remaining:
            byte_index, bit_offset = divmod(self.position, 8)
            take = min(8 - bit_offset, remaining)
            shift = 8 - bit_offset - take
            chunk = (self.payload[byte_index] >> shift) & ((1 << take) - 1)
            value = (value << take) | chunk
            self.position += take
            remaining -= take
        return value

    def require_canonical_padding(self) -> None:
        """Accept only the encoder's zero padding to the next byte boundary."""
        remaining = len(self.payload) * 8 - self.position
        if remaining >= 8:
            raise ValueError("Gorilla bitstream has trailing bytes")
        if remaining and self.read(remaining) != 0:
            raise ValueError("Gorilla bitstream has nonzero padding bits")


def _encode_gorilla32(window: np.ndarray) -> bytes:
    """Encode one [channel, time] float32 window using Gorilla section 4.1.2."""
    bits = np.ascontiguousarray(window, dtype=np.float32).view(np.uint32)
    writer = _BitWriter()
    for channel in bits:
        previous = int(channel[0])
        writer.write(previous, 32)
        previous_leading = 0
        previous_trailing = 0
        has_previous_block = False
        for raw_value in channel[1:]:
            current = int(raw_value)
            xor = previous ^ current
            if xor == 0:
                writer.write(0, 1)
            else:
                writer.write(1, 1)
                leading = 32 - xor.bit_length()
                trailing = (xor & -xor).bit_length() - 1
                if (
                    has_previous_block
                    and leading >= previous_leading
                    and trailing >= previous_trailing
                ):
                    writer.write(0, 1)
                    meaningful = 32 - previous_leading - previous_trailing
                    writer.write(xor >> previous_trailing, meaningful)
                else:
                    writer.write(1, 1)
                    meaningful = 32 - leading - trailing
                    writer.write(leading, 5)
                    writer.write(meaningful, 6)
                    writer.write(xor >> trailing, meaningful)
                    previous_leading = leading
                    previous_trailing = trailing
                    has_previous_block = True
            previous = current
    return writer.finish()


def _decode_gorilla32(payload: bytes, channels: int, time_points: int) -> np.ndarray:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Gorilla window stream must be nonempty bytes")
    minimum_bits = channels * (time_points + 31)
    if len(payload) * 8 < minimum_bits:
        raise ValueError("truncated Gorilla window stream")
    reader = _BitReader(payload)
    bits = np.empty((channels, time_points), dtype=np.uint32)
    for channel_index in range(channels):
        previous = reader.read(32)
        bits[channel_index, 0] = previous
        previous_leading = 0
        previous_trailing = 0
        has_previous_block = False
        for time_index in range(1, time_points):
            if reader.read(1) == 0:
                current = previous
            else:
                if reader.read(1) == 0:
                    if not has_previous_block:
                        raise ValueError("Gorilla stream reuses a missing meaningful-bit block")
                    leading = previous_leading
                    trailing = previous_trailing
                    meaningful = 32 - leading - trailing
                else:
                    leading = reader.read(5)
                    meaningful = reader.read(6)
                    trailing = 32 - leading - meaningful
                    if meaningful < 1 or trailing < 0:
                        raise ValueError("invalid Gorilla meaningful-bit block")
                    previous_leading = leading
                    previous_trailing = trailing
                    has_previous_block = True
                xor = reader.read(meaningful) << trailing
                current = previous ^ xor
            bits[channel_index, time_index] = current
            previous = current
    reader.require_canonical_padding()
    return bits.view(np.float32)


def _decode_gorilla_payload(payload: bytes) -> np.ndarray:
    batch, channels, time_points, _ = _parse_framed_float32_shape(
        payload, b"GOR1", "Gorilla"
    )
    position = struct.calcsize("<4sIII")
    windows: list[np.ndarray] = []
    for _ in range(batch):
        if position + 4 > len(payload):
            raise ValueError("truncated Gorilla frame length")
        frame_size = struct.unpack_from("<I", payload, position)[0]
        position += 4
        if frame_size < 1 or frame_size > len(payload) - position:
            raise ValueError("truncated Gorilla frame")
        windows.append(
            _decode_gorilla32(
                payload[position:position + frame_size], channels, time_points
            )
        )
        position += frame_size
    if position != len(payload):
        raise ValueError("Gorilla stream has trailing bytes")
    return np.stack(windows)


class GorillaCodec(NativeClassicalCodec):
    """Local paper-aligned Gorilla XOR codec adapted from float64 to float32."""

    method_name = "gorilla"
    requires_epsilon = False
    entropy_coder = "local-paper-aligned-gorilla-xor-float32"

    @torch.no_grad()
    def compress_reconstruct(self, x: torch.Tensor, epsilon: float | None = None) -> tuple[torch.Tensor, bytes]:
        if x.dtype != torch.float32:
            raise ValueError("Gorilla requires a float32 tensor")
        source = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if source.ndim != 3 or min(source.shape) < 1:
            raise ValueError(f"Gorilla expects [batch, channel, time], got {source.shape}")
        if source.shape[-1] < 1:
            raise ValueError("Gorilla requires at least one time point")
        reconstructed = np.empty_like(source)
        streams: list[bytes] = []
        for index, window in enumerate(source):
            compressed = _encode_gorilla32(window)
            decoded = _decode_gorilla32(compressed, window.shape[0], window.shape[1])
            reconstructed[index] = decoded
            streams.append(compressed)
        if not np.array_equal(source.view(np.uint32), reconstructed.view(np.uint32)):
            raise RuntimeError("Gorilla failed its lossless bitwise roundtrip check")
        result = torch.from_numpy(reconstructed).to(device=x.device, dtype=x.dtype)
        header = struct.pack(
            "<4sIII", b"GOR1", source.shape[0], source.shape[1], source.shape[2]
        )
        return result, header + _frame_streams(streams)

    def decompress_payload(self, payload: bytes) -> np.ndarray:
        """Decode a complete counted GOR1 artifact with canonical padding checks."""
        return _decode_gorilla_payload(payload)
