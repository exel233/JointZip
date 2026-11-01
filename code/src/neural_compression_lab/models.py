from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from vector_quantize_pytorch import ResidualVQ

from .classical import (
    CAMEOCodec,
    CAMEOJointGuardCodec,
    CAMEOPointGuardCodec,
    FPZipLosslessCodec,
    FPcompressLosslessCodec,
    GorillaCodec,
    LFZipNLMSJointCodec,
    QPETCorrelationCodec,
    SPERRCodec,
    SZ2016Codec,
    ZstdLosslessCodec,
)


def _ste_binary(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hard = (probabilities >= 0.5).to(probabilities.dtype)
    return probabilities + (hard - probabilities).detach(), hard


class CausalTCNBlock(nn.Module):
    """Two-layer, weight-normalized causal TCN residual block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
        self.padding = padding
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def _causal(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return layer(F.pad(x, (self.padding, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.dropout(F.relu(self._causal(self.conv1, x)))
        x = self.dropout(F.relu(self._causal(self.conv2, x)))
        return F.relu(x + residual)


class TCNRNNCodec(nn.Module):
    """Zheng & Zhang TCN-RNN: continuous 1-channel bottleneck and reverse GRU decoder."""

    method_name = "tcn_rnn"
    symbol_kind = "float32"

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 32,
        decoder_hidden: int | None = None,
        kernel_size: int = 4,
        num_blocks: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(num_blocks):
            blocks.append(
                CausalTCNBlock(
                    channels if index == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*blocks)
        self.bottleneck = nn.Conv1d(hidden_dim, 1, 1)
        decoder_hidden = decoder_hidden or channels
        self.decoder = nn.GRU(1, decoder_hidden, batch_first=True)
        self.output = nn.Linear(decoder_hidden, channels)

    def forward(self, x: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        latent = self.bottleneck(self.encoder(x))
        reversed_latent = latent.flip(-1).transpose(1, 2)
        decoded, _ = self.decoder(reversed_latent)
        reconstruction = self.output(decoded).transpose(1, 2).flip(-1)
        return {"reconstruction": reconstruction, "symbols": latent}


class SameConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        total = dilation * (kernel_size - 1)
        self.left = total // 2
        self.right = total - self.left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, self.right)))


class EdgeResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            SameConv1d(channels, channels, 5, dilation=dilation),
            nn.PReLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EdgeEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            EdgeResidualUnit(in_channels, dilation=3),
            nn.PReLU(),
            SameConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EdgeDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride)
        self.units = nn.Sequential(
            nn.ELU(), EdgeResidualUnit(out_channels, 1),
            nn.ELU(), EdgeResidualUnit(out_channels, 3),
            nn.ELU(), EdgeResidualUnit(out_channels, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = x.shape[-1] * self.up.stride[0]
        x = self.up(x)
        if x.shape[-1] != expected:
            x = F.interpolate(x, size=expected, mode="linear", align_corners=False)
        return self.units(x)


class ChannelWiseLinear(nn.Module):
    def __init__(self, channels: int, length: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(length, length) for _ in range(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([layer(x[:, index]) for index, layer in enumerate(self.layers)], dim=1)


class EdgeEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        latent_channels: int,
        stem_width: int,
        channel_progression: Sequence[int],
        temporal_strides: Sequence[int],
    ) -> None:
        super().__init__()
        if len(channel_progression) != 4 or len(temporal_strides) != 4:
            raise ValueError("EdgeCodec requires four encoder blocks")
        layers: list[nn.Module] = [SameConv1d(channels, stem_width, 7), nn.PReLU()]
        inputs = [stem_width, *channel_progression[:-1]]
        for in_width, out_width, stride in zip(inputs, channel_progression, temporal_strides):
            layers.extend([EdgeEncoderBlock(int(in_width), int(out_width), int(stride)), nn.PReLU()])
        layers.append(nn.Conv1d(int(channel_progression[-1]), latent_channels, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EdgeDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        latent_channels: int,
        latent_length: int,
        channel_progression: Sequence[int],
        temporal_strides: Sequence[int],
        middle_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        reversed_widths = list(reversed([int(v) for v in channel_progression[:-1]])) + [channels]
        reversed_strides = list(reversed([int(v) for v in temporal_strides]))
        first_width = int(channel_progression[-1])
        self.start = nn.Sequential(
            ChannelWiseLinear(latent_channels, latent_length),
            nn.ELU(),
            nn.Conv1d(latent_channels, first_width, 1),
            nn.ELU(),
        )
        blocks = []
        in_width = first_width
        middle_length = latent_length
        for index, (out_width, stride) in enumerate(zip(reversed_widths, reversed_strides)):
            blocks.append(EdgeDecoderBlock(in_width, out_width, stride))
            middle_length *= stride
            if index == 1:
                blocks.extend([ChannelWiseLinear(out_width, middle_length), nn.ELU(), nn.Dropout(middle_dropout)])
            in_width = out_width
        self.blocks = nn.Sequential(*blocks)
        self.final = SameConv1d(channels, channels, 3)

    def forward(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        x = self.final(self.blocks(self.start(z)))
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x


def _edge_layout(channels: int, architecture: str) -> tuple[int, int, list[int], list[int]]:
    if architecture == "original":
        return 9, 36, [24, 20, 16, 12], [1, 2, 2, 2]
    if architecture == "thesis":
        return channels, channels, [channels] * 4, [1, 1, 1, 2]
    raise ValueError(f"Unknown EdgeCodec architecture: {architecture}")


class EdgeCodec(nn.Module):
    """EdgeCodec architecture with its original ResidualVQ implementation and quantization axis."""

    method_name = "edgecodec"
    symbol_kind = "codebook"

    def __init__(
        self,
        channels: int,
        window: int,
        architecture: str = "original",
        codebook_size: int = 768,
        num_quantizers: int = 4,
        kmeans_iters: int = 20,
        middle_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        latent_channels, stem_width, progression, strides = _edge_layout(channels, architecture)
        factor = math.prod(strides)
        if window % factor:
            raise ValueError(f"Window {window} must be divisible by EdgeCodec temporal factor {factor}")
        self.latent_length = window // factor
        self.encoder = EdgeEncoder(channels, latent_channels, stem_width, progression, strides)
        self.decoder = EdgeDecoder(
            channels, latent_channels, self.latent_length, progression, strides, middle_dropout=middle_dropout
        )
        self.quantizer = ResidualVQ(
            dim=self.latent_length,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            threshold_ema_dead_code=2,
            kmeans_init=True,
            kmeans_iters=kmeans_iters,
            quantize_dropout=True,
        )
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

    def forward(
        self,
        x: torch.Tensor,
        n_quantizers: int | None = None,
        bypass_quantizer: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        if bypass_quantizer:
            symbols = torch.empty(z.shape[0], z.shape[1], 0, dtype=torch.long, device=z.device)
            return {
                "reconstruction": self.decoder(z, x.shape[-1]),
                "symbols": symbols,
                "vq_loss": z.new_zeros(()),
            }
        zq, indices, commitment = self.quantizer(z)
        if n_quantizers is not None and n_quantizers < self.num_quantizers:
            indices = indices[..., :n_quantizers]
            zq = self.quantizer.get_output_from_indices(indices)
            commitment = commitment[..., :n_quantizers]
        return {
            "reconstruction": self.decoder(zq, x.shape[-1]),
            "symbols": indices,
            "vq_loss": commitment.mean(),
        }


class TaskProbe(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x).squeeze(-1)).squeeze(-1)


class TAEdgeCodec(EdgeCodec):
    """Thesis TAEdgeCodec: EdgeCodec reconstruction is jointly optimized through a task head."""

    method_name = "taedgecodec"

    def __init__(self, channels: int, window: int, task_hidden: int = 128, **kwargs: object) -> None:
        super().__init__(channels=channels, window=window, **kwargs)
        self.task_head = TaskProbe(channels, task_hidden)

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        output = super().forward(x, **kwargs)
        output["task_logit"] = self.task_head(output["reconstruction"])
        return output


class UniEdgeCodec(nn.Module):
    """Thesis adapter: shared Edge encoder/decoder plus disjoint RVQ partitions per signal type."""

    method_name = "uniedgecodec"
    symbol_kind = "codebook"

    def __init__(
        self,
        channels: int,
        window: int,
        groups: dict[str, list[int]],
        partition_sizes: Sequence[int],
        num_quantizers: int = 2,
        kmeans_iters: int = 20,
        middle_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(groups) != len(partition_sizes):
            raise ValueError("partition_sizes must match the configured signal groups")
        covered = sorted(index for indices in groups.values() for index in indices)
        if covered != list(range(channels)):
            raise ValueError("UniEdgeCodec signal groups must partition all channels")
        latent_channels, stem_width, progression, strides = _edge_layout(channels, "thesis")
        self.latent_length = window // math.prod(strides)
        self.encoder = EdgeEncoder(channels, latent_channels, stem_width, progression, strides)
        self.decoder = EdgeDecoder(
            channels, latent_channels, self.latent_length, progression, strides, middle_dropout=middle_dropout
        )
        self.groups = {name: list(indices) for name, indices in groups.items()}
        self.offsets: dict[str, int] = {}
        offset = 0
        quantizers: dict[str, nn.Module] = {}
        for (name, indices), size in zip(self.groups.items(), partition_sizes):
            self.offsets[name] = offset
            if indices:
                quantizers[name] = ResidualVQ(
                    dim=self.latent_length,
                    num_quantizers=num_quantizers,
                    codebook_size=int(size),
                    threshold_ema_dead_code=2,
                    kmeans_init=True,
                    kmeans_iters=kmeans_iters,
                    quantize_dropout=True,
                )
            offset += int(size)
        self.quantizers = nn.ModuleDict(quantizers)
        self.codebook_size = offset
        self.num_quantizers = num_quantizers

    def forward(
        self,
        x: torch.Tensor,
        n_quantizers: int | None = None,
        bypass_quantizer: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        if bypass_quantizer:
            symbols = torch.empty(z.shape[0], z.shape[1], 0, dtype=torch.long, device=z.device)
            return {
                "reconstruction": self.decoder(z, x.shape[-1]),
                "symbols": symbols,
                "vq_loss": z.new_zeros(()),
                "signal_groups": self.groups,
            }
        zq = torch.zeros_like(z)
        levels = n_quantizers or self.num_quantizers
        symbols = torch.zeros(z.shape[0], z.shape[1], levels, dtype=torch.long, device=z.device)
        commitments = []
        for name, indices_list in self.groups.items():
            if not indices_list:
                continue
            quantized, local_indices, commitment = self.quantizers[name](z[:, indices_list])
            if levels < self.num_quantizers:
                local_indices = local_indices[..., :levels]
                quantized = self.quantizers[name].get_output_from_indices(local_indices)
                commitment = commitment[..., :levels]
            zq[:, indices_list] = quantized
            symbols[:, indices_list] = local_indices + self.offsets[name]
            commitments.append(commitment.mean())
        return {
            "reconstruction": self.decoder(zq, x.shape[-1]),
            "symbols": symbols,
            "vq_loss": torch.stack(commitments).mean(),
            "signal_groups": self.groups,
        }


class DeepDictCodec(nn.Module):
    """Paper-sized Bernoulli Transformer autoencoder; residual correction is serialized by the runner."""

    method_name = "deepdict"
    symbol_kind = "binary"

    def __init__(
        self,
        channels: int,
        window: int,
        latent_bits: int = 64,
        hidden_dim: int = 64,
        d_model: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = channels * window
        self.channels = channels
        self.window = window
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_bits),
        )
        self.augment = nn.Sequential(
            nn.Linear(latent_bits, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, window * d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=64,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, channels))
        self.d_model = d_model

    def forward(self, x: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        logits = self.encoder(x.flatten(1))
        probabilities = torch.sigmoid(logits)
        bits, hard = _ste_binary(probabilities)
        sequence = self.augment(bits).view(x.shape[0], self.window, self.d_model)
        reconstruction = self.output(self.transformer(sequence)).transpose(1, 2)
        return {"reconstruction": reconstruction, "symbols": hard.to(torch.uint8)}


class LFZipCodec(nn.Module):
    """PyTorch port of LFZip's reported 4x128 FFN predictor, shared across Tesla channels."""

    method_name = "lfzip"
    symbol_kind = "residual"

    def __init__(
        self,
        channels: int,
        context: int = 32,
        hidden_dim: int = 128,
        hidden_layers: int = 4,
        training_epsilon: float = 0.25,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = context
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_dim), nn.ReLU()])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.predictor = nn.Sequential(*layers)
        self.channels = channels
        self.context = context
        self.training_epsilon = training_epsilon

    def forward(self, x: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        noisy = x
        if self.training and self.training_epsilon > 0:
            noisy = x + torch.empty_like(x).uniform_(-self.training_epsilon, self.training_epsilon)
        if x.shape[-1] <= self.context:
            raise ValueError("LFZip context must be shorter than the data window")
        windows = noisy.unfold(-1, self.context, 1)[..., :-1, :]
        predicted_delta = self.predictor(windows).squeeze(-1)
        target_delta = x[..., self.context:] - x[..., self.context - 1:-1]
        prediction_residual = target_delta - predicted_delta
        prediction_loss = prediction_residual.pow(2).mean()
        predicted = x[..., self.context - 1:-1] + predicted_delta
        reconstruction = torch.cat([x[..., :self.context], predicted], dim=-1)
        return {
            "reconstruction": reconstruction,
            "prediction_loss": prediction_loss,
            "prediction_residual": prediction_residual,
        }

    @torch.no_grad()
    def error_bounded_reconstruct(self, x: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed = torch.zeros_like(x)
        symbols = torch.zeros_like(x, dtype=torch.int32)
        step = 2.0 * epsilon
        for time_index in range(x.shape[-1]):
            if time_index < self.context:
                prediction = torch.zeros_like(x[..., time_index])
            else:
                history = reconstructed[..., time_index - self.context:time_index]
                delta = self.predictor(history).squeeze(-1)
                prediction = reconstructed[..., time_index - 1] + delta
            q = torch.round((x[..., time_index] - prediction) / step)
            reconstructed[..., time_index] = prediction + q * step
            symbols[..., time_index] = q.to(torch.int32)
        return reconstructed, symbols


class LFZipQELCodec(LFZipCodec):
    """LFZip predictor trained on quantized entropy loss instead of plain MSE."""

    method_name = "lfzip_qel"


def build_model(spec: dict, channels: int, window: int, groups: dict[str, list[int]]) -> nn.Module:
    name = spec["name"]
    ignored = {
        "name", "tv_weight", "peak_weight", "peak_quantile", "residual_epsilons",
        "task_weight", "warmup_epochs", "qel_weight", "reconstruction_weight",
        "qel_epsilon", "qel_exponent", "qel_max_bin",
        "learning_rate", "weight_decay", "optimizer_betas", "scheduler", "eta_min",
        "train_per_epsilon",
        "acf_epsilons", "acf_metric_lags", "operating_points",
        "collection_frontier",
    }
    common = {key: value for key, value in spec.items() if key not in ignored}
    if name == "tcn_rnn":
        return TCNRNNCodec(channels=channels, **common)
    if name == "edgecodec":
        return EdgeCodec(channels=channels, window=window, **common)
    if name == "uniedgecodec":
        return UniEdgeCodec(channels=channels, window=window, groups=groups, **common)
    if name == "taedgecodec":
        return TAEdgeCodec(channels=channels, window=window, **common)
    if name == "deepdict":
        return DeepDictCodec(channels=channels, window=window, **common)
    if name == "lfzip":
        return LFZipCodec(channels=channels, **common)
    if name == "lfzip_qel":
        return LFZipQELCodec(channels=channels, **common)
    if name == "lfzip_nlms_joint":
        return LFZipNLMSJointCodec(**common)
    if name == "zstd_lossless":
        return ZstdLosslessCodec(**common)
    if name == "fpzip_lossless":
        return FPZipLosslessCodec()
    if name == "fpcompress_lossless":
        return FPcompressLosslessCodec()
    if name == "sz2016":
        return SZ2016Codec()
    if name == "gorilla":
        return GorillaCodec()
    if name == "sperr":
        return SPERRCodec()
    if name == "cameo":
        return CAMEOCodec(**common)
    if name == "cameo_point_guard":
        return CAMEOPointGuardCodec(**common)
    if name == "cameo_joint_guard":
        return CAMEOJointGuardCodec(**common)
    if name == "qpet_correlation":
        return QPETCorrelationCodec(**common)
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
