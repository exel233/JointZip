"""Independent, frozen official DLinear forecasting on main-benchmark artifacts.

This is a new six-source evaluation, not a reproduction of every EDBT setting.
Training never receives TEST arrays. Evaluation decodes persisted SCS2 only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.vldb.main_benchmark import tasks_data as data

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_MODEL = ROOT / "tmp/upstreams/EvalImpLSTS-audit/forecasting/DLinear/components/model.py"
OFFICIAL_SHA256 = "06659056434eedf125b0740073c2c35a80830db1a9a58281d95d6d161f8abb64"
DATASETS = ("ETTm1", "ETTm2", "Weather", "ElecDem", "Wind", "Solar")
DEFAULT_RUN = ROOT / "experiments/results/vldb_edbt_dlinear_20260907_v1"
CONFIG = {
    "context": 96, "horizon": 24, "kernel": 25, "individual": False,
    "epochs": 10, "batch_size": 32, "learning_rate": 0.001,
    "patience": 3, "seed": 20260907, "train_stride": 1, "drop_last": False,
    "optimizer": "Adam", "loss": "MSE", "cpu_threads": 1,
    "lr_schedule": "official type1: set base_lr*0.5**(epoch-1) after 1-based epoch",
    "selection": "minimum raw VAL MSE; exact ties keep first epoch; patience=3",
    "training": "all contiguous TRAIN-only 96+24 pairs; no TEST access or compressor retraining",
    "solar": "one shared-parameter model trained jointly across all 137 train-standardized variates",
    "evaluation": "last 96 points of each frozen 128-point source; same raw 24-point future",
    "aggregation": "series within dataset, then six-dataset macro; Solar is one source domain",
    "claim": "official DLinear architecture with declared evaluation changes; not strict EDBT replication",
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_bytes(canonical(value))
    temp.replace(path)


def source_identity() -> dict[str, str]:
    if data.file_sha256(OFFICIAL_MODEL) != OFFICIAL_SHA256:
        raise RuntimeError("official DLinear source hash does not match audited snapshot")
    return {"dlinear_evaluator_sha256": data.file_sha256(__file__),
            "tasks_data_sha256": data.file_sha256(data.__file__),
            "official_model_sha256": OFFICIAL_SHA256,
            "official_commit": data.UPSTREAM_COMMIT,
            "official_repository": "https://github.com/cmcuza/EvalImpLSTS"}


def new_model(channels: int, *, context: int = 96, horizon: int = 24):
    """Load the unmodified SHA-bound official implementation directly."""
    source_identity()
    spec = importlib.util.spec_from_file_location("edbt_audited_dlinear_model", OFFICIAL_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model(SimpleNamespace(seq_len=context, pred_len=horizon,
                                        individual=False, enc_in=channels)).float()


def configure_runtime(device: str) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA training requested but CUDA unavailable")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    return {"device": device, "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "deterministic_algorithms": True, "cpu_threads": torch.get_num_threads(),
            "tf32": False, "cudnn_benchmark": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def bundle_identity(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: bundle[key] for key in (
        "dataset", "series_id", "channel_name", "source_file_sha256", "physical_source_sha256",
        "train_standardized_sha256", "source_standardized_sha256", "future_standardized_sha256",
        "val_source_standardized_sha256", "val_future_standardized_sha256", "mean", "std",
        "split_boundaries", "normalization", "selection",
    )} | {"starts": bundle["starts"].tolist(), "val_starts": bundle["val_starts"].tolist()}


def freeze(run: Path, *, device: str = "cuda") -> dict[str, Any]:
    if (run / "freeze.json").exists():
        raise FileExistsError("independent DLinear freeze is immutable; choose a new run")
    identity = source_identity()
    record = {"schema": "nclab.edbt-dlinear-freeze.v1", "config": CONFIG,
              "identity": identity, "runtime": configure_runtime(device),
              "series": [bundle_identity(b) for name in DATASETS for b in data.prepare_dataset(name)],
              "created_utc": datetime.now(timezone.utc).isoformat()}
    if identity != source_identity():
        raise RuntimeError("source changed during DLinear freeze")
    write_json(run / "freeze.json", record)
    snapshot = run / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, path in (("dlinear.py", Path(__file__)), ("tasks_data.py", Path(data.__file__)),
                       ("official_model.py", OFFICIAL_MODEL)):
        (snapshot / name).write_bytes(path.read_bytes())
    return {"series": len(record["series"]), "freeze_sha256": data.file_sha256(run / "freeze.json")}


def verify_run(run: Path) -> dict[str, Any]:
    record = json.loads((run / "freeze.json").read_text())
    if record["config"] != CONFIG or record["identity"] != source_identity():
        raise RuntimeError("DLinear protocol/source changed since freeze")
    if record["runtime"] != configure_runtime(record["runtime"]["device"]):
        raise RuntimeError("DLinear runtime changed since freeze")
    return record


def group_arrays(bundles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TRAIN [T,C], raw validation context [B,96,C], future [B,24,C]."""
    if not bundles:
        raise ValueError("empty dataset group")
    for bundle in bundles[1:]:
        if not np.array_equal(bundle["val_starts"], bundles[0]["val_starts"]):
            raise ValueError("joint-channel model requires aligned validation origins")
    train = np.stack([b["train_1d"] for b in bundles], axis=-1)
    context = np.stack([b["val_source"][:, 0, -96:] for b in bundles], axis=-1)
    future = np.stack([b["val_future"] for b in bundles], axis=-1)
    return train, context, future


def predict(model: Any, source: np.ndarray, *, device: str = "cpu", context: int = 96) -> np.ndarray:
    """Input [B,1,T] for one series or [B,T,C] for a joint-channel group."""
    import torch
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("prediction inputs must be finite rank-three arrays")
    if values.shape[1] == 1:
        values = values.transpose(0, 2, 1)
    if values.shape[1] < context:
        raise ValueError("insufficient forecast context")
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(values[:, -context:, :].copy(), device=device)).cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite DLinear prediction")
    return output


def fit_raw(train: np.ndarray, val_context: np.ndarray, val_future: np.ndarray, *,
            device: str, config: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """The training API deliberately accepts no TEST arrays or labels."""
    import torch
    cfg = CONFIG if config is None else config
    configure_runtime(device)
    torch.manual_seed(int(cfg["seed"]))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    training = np.asarray(train, dtype=np.float32)
    vx, vy = np.asarray(val_context, dtype=np.float32), np.asarray(val_future, dtype=np.float32)
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    if training.ndim != 2 or vx.ndim != 3 or vy.ndim != 3:
        raise ValueError("TRAIN must be [T,C] and validation must be [B,T,C]")
    if vx.shape[1:] != (context, training.shape[1]) or vy.shape != (len(vx), horizon, training.shape[1]):
        raise ValueError("incompatible validation context/future shape")
    if not all(np.isfinite(a).all() for a in (training, vx, vy)):
        raise ValueError("training/validation must be finite")
    count = len(training) - context - horizon + 1
    if count < 1:
        raise ValueError("insufficient training history")
    model = new_model(training.shape[1], context=context, horizon=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    windows = torch.as_tensor(training.copy(), device=device).unfold(0, context + horizon, 1)
    validation = torch.as_tensor(vx.copy(), device=device)
    target = torch.as_tensor(vy.copy(), device=device)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(count, device=device)
        total_loss = torch.zeros((), device=device)
        for begin in range(0, count, int(cfg["batch_size"])):
            chunk = windows[order[begin:begin + int(cfg["batch_size"])]]
            x = chunk[:, :, :context].transpose(1, 2)
            y = chunk[:, :, context:].transpose(1, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(chunk)
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(validation) - target) ** 2).item())
        if not np.isfinite(val_loss):
            raise RuntimeError("nonfinite DLinear validation loss")
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_mse": float((total_loss / count).item()),
                        "validation_mse": val_loss, "seconds": time.perf_counter() - epoch_started,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(json.dumps({"dlinear_epoch": history[-1], "channels": training.shape[1]}), flush=True)
        if stale >= int(cfg["patience"]):
            break
        for group in optimizer.param_groups:
            group["lr"] = float(cfg["learning_rate"]) * 0.5 ** (epoch - 1)
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch, "validation_mse": best_loss,
                   "training_pairs": count, "channels": training.shape[1], "epochs": history,
                   "training_seconds": time.perf_counter() - started, "config": cfg}


def save_model(run: Path, dataset: str, model: Any, metadata: dict[str, Any]) -> None:
    folder = run / "models"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f"{dataset}.npz"
    if checkpoint.exists() or checkpoint.with_suffix(".json").exists():
        raise FileExistsError(f"model already exists: {dataset}")
    with checkpoint.open("xb") as stream:
        np.savez_compressed(stream, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    write_json(checkpoint.with_suffix(".json"), {**metadata,
               "checkpoint_sha256": data.file_sha256(checkpoint),
               "freeze_sha256": data.file_sha256(run / "freeze.json")})


def load_model(run: Path, dataset: str, *, device: str):
    import torch
    checkpoint = run / "models" / f"{dataset}.npz"
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    if metadata["freeze_sha256"] != data.file_sha256(run / "freeze.json"):
        raise RuntimeError("model belongs to a different DLinear freeze")
    if metadata["checkpoint_sha256"] != data.file_sha256(checkpoint):
        raise RuntimeError("DLinear checkpoint hash mismatch")
    model = new_model(metadata["channels"]).to(device)
    with np.load(checkpoint, allow_pickle=False) as checkpoint_arrays:
        model.load_state_dict({key: torch.from_numpy(checkpoint_arrays[key]).to(device) for key in checkpoint_arrays.files})
    model.eval()
    return model, metadata


def train(run: Path, *, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    record = verify_run(run)
    complete = []
    for dataset in datasets:
        bundles = data.prepare_dataset(dataset)
        for bundle in bundles:
            expected = next(x for x in record["series"] if x["series_id"] == bundle["series_id"])
            if expected != bundle_identity(bundle):
                raise RuntimeError("training data differ from DLinear freeze")
        if (run / "models" / f"{dataset}.json").exists():
            load_model(run, dataset, device=record["runtime"]["device"])
            complete.append(dataset)
            continue
        model, metadata = fit_raw(*group_arrays(bundles), device=record["runtime"]["device"])
        verify_run(run)
        save_model(run, dataset, model, {**metadata, "dataset": dataset,
                   "series_ids": [b["series_id"] for b in bundles]})
        complete.append(dataset)
        write_json(run / "TRAIN_STATUS.json", {"verified_complete_datasets": complete})
    return {"verified_complete_datasets": complete}


def evaluate(run: Path, main_run: Path, *, datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    """Replay final persisted TEST artifacts; no encoding or model fitting."""
    record = verify_run(run)
    from experiments.vldb.main_benchmark import native_bridge
    from experiments.vldb.main_benchmark.benchmark import identity as main_identity
    from experiments.vldb.utility.artifacts import parse_scs2
    native_bridge.install_linux_transport()
    main_freeze = json.loads((main_run / "freeze.json").read_text())
    if main_freeze["identity"] != main_identity():
        raise RuntimeError("compression decoder implementation/runtime differs from its freeze")
    main_sha = data.file_sha256(main_run / "freeze.json")
    freeze_sha = data.file_sha256(run / "freeze.json")
    binding = {"main_freeze_sha256": main_sha, "dlinear_freeze_sha256": freeze_sha}
    binding_path = run / "evaluation_binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError("DLinear evaluation was already bound to a different compression freeze")
    write_json(binding_path, binding)
    counts: dict[str, int] = {}
    for dataset in datasets:
        model, model_metadata = load_model(run, dataset, device=record["runtime"]["device"])
        for bundle in data.prepare_dataset(dataset):
            expected = next(x for x in record["series"] if x["series_id"] == bundle["series_id"])
            if expected != bundle_identity(bundle):
                raise RuntimeError("task source differs from DLinear freeze")
            main_source = next(x for x in main_freeze["series"] if x["series_id"] == bundle["series_id"])
            if main_source["source_file_sha256"] != bundle["source_file_sha256"]:
                raise RuntimeError("compression and DLinear use different physical source files")
            for key in ("source", "future", "val_source", "val_future"):
                detail = main_source["arrays"][key]
                if detail["sha256"] != data.array_sha256(bundle[key]) or detail["shape"] != list(bundle[key].shape):
                    raise RuntimeError(f"compression and DLinear differ: {bundle['series_id']}/{key}")
            raw_prediction = predict(model, bundle["source"], device=record["runtime"]["device"])[:, :, 0]
            for row_path in sorted((main_run / "rows" / dataset / bundle["series_id"]).glob("*.json")):
                row = json.loads(row_path.read_text())
                if row["comparison"] == "calibration":
                    continue
                output = run / "rows" / dataset / bundle["series_id"] / row_path.name
                row_sha = data.file_sha256(row_path)
                if output.exists():
                    if json.loads(output.read_text())["compression_row_sha256"] != row_sha:
                        raise RuntimeError("completed compression row changed during DLinear resume")
                    continue
                result = {"dataset": dataset, "series_id": bundle["series_id"],
                          "method": row["method"], "point": row["point"], "comparison": row["comparison"],
                          "compression_row_sha256": row_sha, "model_sha256": model_metadata["checkpoint_sha256"],
                          **binding}
                try:
                    if row.get("freeze_sha256") != main_sha:
                        raise RuntimeError("compression row freeze mismatch")
                    if row["status"] != "complete":
                        result.update(status="upstream_" + row["status"], upstream_reason=row.get("error", row.get("reason")))
                    else:
                        parsed = parse_scs2(main_run / row["artifact"])
                        if parsed.sha256 != row["artifact_sha256"] or len(parsed.payloads) != 1:
                            raise RuntimeError("persisted compressed artifact identity mismatch")
                        if (parsed.channels != (bundle["channel_name"],) or float(parsed.mean[0]) != bundle["mean"]
                                or float(parsed.std[0]) != bundle["std"]):
                            raise RuntimeError("persisted artifact normalizer/channel differs from frozen task")
                        decoded = native_bridge.decode_payload(parsed.codec_id, parsed.payloads[0], bundle["source"].shape)
                        if decoded.shape != bundle["source"].shape or decoded.dtype != np.float32:
                            raise RuntimeError("persisted decode shape/dtype differs from frozen encoder input")
                        prediction = predict(model, decoded, device=record["runtime"]["device"])[:, :, 0]
                        metrics = data.forecast_metrics(bundle["future"], raw_prediction, prediction,
                                                       bundle["train_1d"], seasonal_period=bundle["seasonal_period"])
                        result.update(status="complete", tasks=metrics, artifact_sha256=parsed.sha256,
                                      artifact_bytes=parsed.byte_size, raw_bytes=int(bundle["source"].nbytes),
                                      predictions={"raw": raw_prediction.tolist(), "decoded": prediction.tolist()})
                except Exception as exc:
                    result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                write_json(output, result)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
    verify_run(run)
    write_json(run / "EVALUATION_STATUS.json", {"new_rows": counts, **binding})
    return {"new_rows": counts, **binding}


def watch(run: Path, main_run: Path, *, poll_seconds: float = 30, max_hours: float = 48) -> dict[str, Any]:
    """Bounded evaluation follower; exits after a final drain or an explicit timeout."""
    if not 1 <= poll_seconds <= 60 or not 0 < max_hours <= 168:
        raise ValueError("poll must be 1..60 seconds and max_hours must be >0 and <=168")
    verify_run(run)
    for dataset in DATASETS:
        if not (run / "models" / f"{dataset}.json").exists():
            raise RuntimeError(f"missing frozen trained model: {dataset}")
    started = time.monotonic()
    while True:
        if (main_run / "freeze.json").exists():
            latest = evaluate(run, main_run)
        else:
            latest = {"waiting_for_compression_freeze": True}
        state = {"schema": "nclab.edbt-dlinear-watch-status.v1", "linux_pid": os.getpid(),
                 "main_run": str(main_run.resolve()), "poll_seconds": poll_seconds,
                 "max_hours": max_hours, "elapsed_seconds": time.monotonic() - started,
                 "updated_utc": datetime.now(timezone.utc).isoformat(), "latest": latest}
        if (main_run / "FINISHED.json").exists():
            # Evaluate once more after observing completion: a row can arrive
            # after the preceding glob scan and before FINISHED is published.
            state["final_drain"] = evaluate(run, main_run)
            state["status"] = "finished_after_main"
            state["main_finished_sha256"] = data.file_sha256(main_run / "FINISHED.json")
            main_final = json.loads((main_run / "FINISHED.json").read_text())
            state["main_all_rows_passed"] = main_final.get("all_rows_passed")
            state["main_source_failures"] = main_final.get("source_failures", [])
            totals: dict[str, int] = {}
            for path in (run / "rows").glob("*/*/*.json"):
                status = json.loads(path.read_text())["status"]
                totals[status] = totals.get(status, 0) + 1
            state["row_counts"] = totals
            write_json(run / "FINISHED.json", state)
            return state
        if time.monotonic() - started >= max_hours * 3600:
            state["status"] = "timeout_waiting_for_main"
            write_json(run / "WATCH_TIMEOUT.json", state)
            return state
        state["status"] = "watching"
        write_json(run / "WATCH_STATUS.json", state)
        print(json.dumps({"dlinear_watch": state}, ensure_ascii=False), flush=True)
        time.sleep(poll_seconds)


def smoke() -> dict[str, Any]:
    """Synthetic identity, save/load-equivalent state, and no-core model smoke."""
    import torch
    config = dict(CONFIG, epochs=1, batch_size=32)
    samples = np.arange(480, dtype=np.float32)
    series = np.stack((np.sin(samples / 11), np.cos(samples / 17)), axis=-1)
    model, info = fit_raw(series[:360], series[240:336][None], series[336:360][None],
                          device="cpu", config=config)
    query = series[320:448][None]
    prediction = predict(model, query)
    assert prediction.shape == (1, 24, 2) and np.isfinite(prediction).all()
    # Shared weights must give the same result jointly or one variate at a time.
    single = np.concatenate([predict(model, query[:, :, i:i+1]) for i in range(2)], axis=2)
    np.testing.assert_allclose(single, prediction, rtol=2e-6, atol=2e-6)
    clone = new_model(2)
    clone.load_state_dict({key: value.clone() for key, value in model.state_dict().items()})
    np.testing.assert_array_equal(predict(clone, query), prediction)
    metrics = data.forecast_metrics(series[448:472, 0][None], prediction[:, :, 0], prediction[:, :, 0], series[:360, 0])
    assert metrics["forecast_excess_mse"] == 0
    json.dumps(info, allow_nan=False)
    return {"synthetic_smoke": "passed", "official_sha256": OFFICIAL_SHA256,
            "prediction_shape": list(prediction.shape), "training_pairs": info["training_pairs"],
            "torch_version": torch.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "evaluate", "watch", "train-watch", "smoke"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--main-run", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=48)
    args = parser.parse_args()
    if args.command in ("evaluate", "watch", "train-watch") and args.main_run is None:
        parser.error(f"{args.command} requires --main-run")
    if args.command in ("watch", "train-watch") and tuple(args.datasets) != DATASETS:
        parser.error("watch/train-watch require the complete frozen six-dataset model set")
    try:
        if args.command == "freeze":
            result = freeze(args.run, device=args.device)
        elif args.command == "train":
            result = train(args.run, datasets=tuple(args.datasets))
        elif args.command == "evaluate":
            result = evaluate(args.run, args.main_run, datasets=tuple(args.datasets))
        elif args.command in ("watch", "train-watch"):
            write_json(args.run / "RUNNING.json", {"linux_pid": os.getpid(), "command": args.command,
                       "started_utc": datetime.now(timezone.utc).isoformat(),
                       "main_run": str(args.main_run.resolve()), "max_hours": args.max_hours,
                       "poll_seconds": args.poll_seconds})
            if args.command == "train-watch":
                train(args.run)
            result = watch(args.run, args.main_run, poll_seconds=args.poll_seconds, max_hours=args.max_hours)
        else:
            result = smoke()
    except Exception as exc:
        if args.command != "smoke":
            write_json(args.run / "FAILED.json", {"status": "failed", "linux_pid": os.getpid(),
                       "command": args.command, "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc(), "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
