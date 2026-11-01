from __future__ import annotations

import torch
import torch.nn.functional as F


def total_variation(reconstructed: torch.Tensor) -> torch.Tensor:
    return (reconstructed[..., 1:] - reconstructed[..., :-1]).abs().mean()


def peak_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    quantile: float = 0.9,
) -> torch.Tensor:
    gradient = (original[..., 1:] - original[..., :-1]).abs()
    threshold = torch.quantile(gradient.detach().flatten(1), quantile, dim=1, keepdim=True)
    mask = gradient >= threshold.unsqueeze(-1)
    point_error = (reconstructed[..., 1:] - original[..., 1:]).pow(2)
    return (point_error * mask).sum() / mask.sum().clamp_min(1)


class _QuantizedEntropy(torch.autograd.Function):
    """Deep Dict QEL: hard entropy forward with the paper's surrogate gradient."""

    @staticmethod
    def forward(ctx, residual: torch.Tensor, epsilon: float, exponent: float) -> torch.Tensor:
        if epsilon <= 0:
            raise ValueError("QEL epsilon must be positive")
        flat = residual.reshape(-1)
        quantized = torch.round(flat / (2.0 * epsilon)) * (2.0 * epsilon)
        symbols, counts = torch.unique(quantized, return_counts=True)
        probabilities = counts.to(flat.dtype) / flat.numel()
        entropy = -(probabilities * torch.log(probabilities)).sum()
        ctx.save_for_backward(flat.detach(), symbols.detach(), probabilities.detach())
        ctx.input_shape = residual.shape
        ctx.epsilon = float(epsilon)
        ctx.exponent = float(exponent)
        return entropy

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        residual, symbols, probabilities = ctx.saved_tensors
        epsilon = ctx.epsilon
        exponent = ctx.exponent
        sample_count = residual.numel()
        coefficient = 1.0 + torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
        gradient = torch.zeros_like(residual)

        # Equation (4) is evaluated in a numerically stable form.  For |u|>1,
        # u^(b-1)/(u^b+1)^2 is rewritten with negative powers.
        for symbol, weight in zip(symbols, coefficient):
            u = (residual - symbol) / epsilon
            magnitude = u.abs()
            sign = u.sign()
            near = magnitude <= 1.0
            response = torch.empty_like(u)
            near_magnitude = magnitude[near]
            near_power = near_magnitude.pow(exponent)
            response[near] = (
                sign[near]
                * near_magnitude.pow(exponent - 1.0)
                / (near_power + 1.0).pow(2)
            )
            far_magnitude = magnitude[~near]
            inverse_power = far_magnitude.pow(-exponent)
            response[~near] = (
                sign[~near]
                * far_magnitude.pow(-exponent - 1.0)
                / (inverse_power + 1.0).pow(2)
            )
            gradient.add_(weight * response)
        gradient.mul_(exponent / (sample_count * epsilon))
        return grad_output * gradient.reshape(ctx.input_shape), None, None


def quantized_entropy_loss(
    residual: torch.Tensor,
    epsilon: float,
    exponent: float = 10.0,
    max_bin: int | None = None,
) -> torch.Tensor:
    """Entropy of hard ``2*epsilon`` symbols with Deep Dict's QEL backward.

    ``max_bin`` remains an accepted no-op for old configs; unlike the previous
    soft histogram, the paper's QEL uses every symbol actually present.
    """
    del max_bin
    return _QuantizedEntropy.apply(residual, float(epsilon), float(exponent))


def _uniedge_losses(output: dict, original: torch.Tensor, reconstructed: torch.Tensor, spec: dict):
    groups: dict[str, list[int]] = output["signal_groups"]
    tv_terms = []
    peak_terms = []
    for name, indices in groups.items():
        if not indices:
            continue
        if "discontinuous" in name:
            tv_terms.append(total_variation(reconstructed[:, indices]))
        if "irregular" in name:
            peak_terms.append(
                peak_loss(
                    original[:, indices],
                    reconstructed[:, indices],
                    quantile=float(spec.get("peak_quantile", 0.9)),
                )
            )
    tv = torch.stack(tv_terms).mean() if tv_terms else original.new_zeros(())
    peaks = torch.stack(peak_terms).mean() if peak_terms else original.new_zeros(())
    return tv, peaks


def compression_loss(
    name: str,
    output: dict,
    original: torch.Tensor,
    labels: torch.Tensor,
    spec: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstructed = output["reconstruction"]

    if name == "lfzip":
        loss = output["prediction_loss"]
        return loss, {"prediction_mse": float(loss.detach()), "total": float(loss.detach())}

    if name == "lfzip_qel":
        prediction_mse = output["prediction_loss"]
        epsilon = float(spec.get("qel_epsilon", spec.get("training_epsilon", 0.25)))
        qel = quantized_entropy_loss(
            output["prediction_residual"],
            epsilon,
            exponent=float(spec.get("qel_exponent", 10.0)),
            max_bin=int(spec.get("qel_max_bin", 32)),
        )
        loss = float(spec.get("qel_weight", 1.0)) * qel
        loss = loss + float(spec.get("reconstruction_weight", 0.01)) * prediction_mse
        return loss, {
            "prediction_mse": float(prediction_mse.detach()),
            "qel": float(qel.detach()),
            "total": float(loss.detach()),
        }

    mse = F.mse_loss(reconstructed, original)
    if name == "tcn_rnn":
        # The paper sums across channels and time, divides by sequence length,
        # and averages over the batch (rather than using element-wise mean MSE).
        loss = (reconstructed - original).pow(2).sum(dim=(1, 2)).div(original.shape[-1]).mean()
        return loss, {"summed_mse": float(loss.detach()), "mse": float(mse.detach()), "total": float(loss.detach())}

    smooth = F.smooth_l1_loss(reconstructed, original)
    if name == "deepdict":
        epsilon = float(spec.get("qel_epsilon", spec["residual_epsilons"][0]))
        qel = quantized_entropy_loss(
            original - reconstructed,
            epsilon,
            exponent=float(spec.get("qel_exponent", 10.0)),
            max_bin=int(spec.get("qel_max_bin", 32)),
        )
        loss = float(spec.get("qel_weight", 1.0)) * qel + float(spec.get("reconstruction_weight", 0.0)) * mse
        parts = {"mse": float(mse.detach()), "qel": float(qel.detach())}
    else:
        loss = 0.8 * mse + 0.2 * smooth
        parts = {"mse": float(mse.detach()), "smooth_l1": float(smooth.detach())}

    if "vq_loss" in output:
        loss = loss + output["vq_loss"]
        parts["vq"] = float(output["vq_loss"].detach())
    if name == "uniedgecodec":
        tv, peaks = _uniedge_losses(output, original, reconstructed, spec)
        loss = loss + float(spec.get("tv_weight", 0.1)) * tv
        loss = loss + float(spec.get("peak_weight", 1.0)) * peaks
        parts["tv"] = float(tv.detach())
        parts["peak"] = float(peaks.detach())
    if "task_logit" in output:
        positives = labels.sum().clamp_min(1)
        negatives = (1 - labels).sum().clamp_min(1)
        task = F.binary_cross_entropy_with_logits(output["task_logit"], labels, pos_weight=negatives / positives)
        loss = loss + float(spec.get("task_weight", 1e-2)) * task
        parts["task"] = float(task.detach())
    parts["total"] = float(loss.detach())
    return loss, parts
