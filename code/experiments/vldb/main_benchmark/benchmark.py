"""Resumable, complete-artifact EDBT-source compression evaluation.

Run with the dedicated Linux environment; ``freeze`` must precede ``run``.
The protocol is a new evaluation on the EDBT authors' data, not a reproduction
of their entire model/relative-error protocol. No test-based parameter tuning.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import traceback
from typing import Any
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "experiments/results/vldb_edbt_main_20260907_v1"
DATASETS = ["ETTm1", "ETTm2", "Weather", "ElecDem", "Wind", "Solar"]
METHODS = ["qpet_joint", "qpet_point", "qpet_pearson", "qpet_acf", "qpet_cf9",
           "qpet_legacy", "sperr", "sz2016", "sz3", "zfp", "cameo", "cameo_joint", "zstd"]
BASELINES = ["sperr", "sz2016", "sz3", "zfp", "qpet_point"]
BUNDLE_ARRAYS = ["source", "future", "val_source", "val_future", "starts", "val_starts", "train_1d", "train_raw_1d"]
BUNDLE_FIELDS = ["source_file_sha256", "physical_source_sha256", "train_standardized_sha256",
                 "channel_name", "mean", "std", "split_boundaries", "sampling_interval_seconds",
                 "upstream_commit", "selection", "normalization"]
POINTS = [dict(label="strict", epsilon=0.1, tau=0.01, rho=0.999),
          dict(label="medium", epsilon=0.25, tau=0.02, rho=0.995),
          dict(label="loose", epsilon=0.5, tau=0.05, rho=0.99)]
CALIBRATION_EPSILONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
PROTOCOL = {
    "schema": "nclab.edbt-source-main-protocol.v1", "datasets": DATASETS,
    "methods": METHODS, "points": POINTS, "calibration_epsilons": CALIBRATION_EPSILONS,
    "validation_selected_baselines": BASELINES, "input_window": 128,
    "task_context": 96, "forecast_horizon": 24, "test_windows_per_series": 32,
    "validation_windows_per_series": 8, "solar_series": "all_137",
    "normalization": "per-series TRAIN-only float32 mean/std; independently billed NOR2",
    "window_selection": "length-only deterministic spread; disjoint input windows; no score selection",
    "task_models": "raw-TRAIN Ridge, validation-only p/alpha selection; persistence reference",
    "same_rate_selector": "min abs(log(validation complete SCS2 bytes / joint bytes)); tie smaller epsilon",
    "same_rate_test_tolerance_fraction": 0.05,
    "same_constraint_selector": "smallest validation full SCS2 among tested epsilons passing all three target gates",
    "float_acceptance_tolerance": 1e-7,
    "timing": "one adapter encode including immediate reconstruction, excluding outer SCS2 wrapping; three read+unpack+decode repetitions with warm OS caches; native /usr/bin/time RSS",
    "failure_policy": "record exception/timeout/OOM/contract failure; never silently remove a row",
    "downstream_failure_policy": "forecast usefulness: raw Ridge MSE < raw persistence MSE but reconstructed Ridge MSE >= raw persistence MSE; descriptive threshold, not codec crash; report prevalence and ties, no posthoc sample removal",
    "concurrency": 2, "native_threads": 1,
    "aggregation": "series within dataset, then six-dataset macro; Solar plants not independent domains",
    "rate_scope": "selected disjoint float32 history windows; not complete continuous streams or timestamps",
    "evidence": "predeclared independent evaluation on previously public datasets",
    "claims_not_established": ["state of the art", "arbitrary downstream guarantee", "all-corpus performance"],
}
_LOCK = threading.Lock()


def normalize(value: Any) -> Any:
    if is_dataclass(value):
        return normalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value: Any) -> bytes:
    return (json.dumps(normalize(value), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_bytes(canonical(value))
    temp.replace(path)


def get(bundle: Any, key: str) -> Any:
    return bundle[key] if isinstance(bundle, dict) else getattr(bundle, key)


def identity() -> dict[str, Any]:
    from . import modern_codecs
    paths = [p for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".sh", ".cpp")]
    paths += [ROOT / relative for relative in [
        "src/neural_compression_lab/classical.py", "src/neural_compression_lab/runner.py",
        "src/neural_compression_lab/metrics.py", "src/neural_compression_lab/data.py",
        "src/neural_compression_lab/models.py", "src/neural_compression_lab/bitstream.py",
        "experiments/vldb/utility/artifacts.py", "tools/cameo/cameo_batch.py",
        "tools/cameo/cameo_pointwise_batch.py", "tools/qpet_correlation/bin/qpet_correlation_batch",
        "tools/sperr/bin/sperr_batch", "tools/sz2016/bin/sz2016_batch",
        "tools/sz2016/sz2016.config",
    ]]
    paths += [p for p in (ROOT / "tmp/upstreams/cameo").rglob("*")
              if p.is_file() and p.suffix in (".py", ".so", ".pyx", ".pxd")]
    for relative in ["tools/qpet_correlation/CorrelationQoI.hpp",
                     "tools/qpet_correlation/SequenceBeamSearch.hpp",
                     "tools/qpet_correlation/qpet_correlation_batch.cpp"]:
        paths.append(ROOT / relative)
    versions = {}
    for package in ["numpy", "pandas", "pyarrow", "torch", "scipy", "scikit-learn", "zstandard"]:
        versions[package] = importlib.metadata.version(package)
    result = {"files": {str(p.relative_to(ROOT)): digest_file(p) for p in sorted(set(paths))},
              "python": platform.python_version(), "packages": versions,
              "modern_codecs": modern_codecs.available_codecs()}
    if set(result["modern_codecs"]) != {"sz3", "zfp"}:
        raise RuntimeError("Both frozen modern baselines must be built and verified before freeze/run")
    # Dynamic libraries are part of the executable identity, not just ldd text.
    dependencies = {}
    for relative in ["tools/qpet_correlation/bin/qpet_correlation_batch",
                     "tools/sperr/bin/sperr_batch", "tools/sz2016/bin/sz2016_batch"]:
        output = subprocess.check_output(["ldd", str(ROOT / relative)], text=True)
        for line in output.splitlines():
            for token in line.split():
                if token.startswith("/") and Path(token).is_file():
                    dependencies[token] = digest_file(Path(token).resolve())
    result["native_dynamic_libraries"] = dependencies
    return result


def freeze(run: Path) -> dict[str, Any]:
    from . import tasks_data
    if (run / "freeze.json").exists():
        raise FileExistsError("Choose a new run directory; existing freeze is immutable")
    frozen_identity = identity()
    bundles = []
    for dataset in DATASETS:
        for bundle in tasks_data.prepare_dataset(dataset):
            arrays = {}
            for name in BUNDLE_ARRAYS:
                a = np.ascontiguousarray(get(bundle, name))
                arrays[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
            bundles.append({"dataset": dataset, "series_id": get(bundle, "series_id"),
                            **{name: normalize(get(bundle, name)) for name in BUNDLE_FIELDS},
                            "starts": normalize(get(bundle, "starts")),
                            "val_starts": normalize(get(bundle, "val_starts")), "arrays": arrays})
    record = {"schema": "nclab.edbt-source-main-freeze.v1", "protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(canonical(PROTOCOL)).hexdigest(),
              "identity": frozen_identity, "series": bundles,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                           "affinity": sorted(os.sched_getaffinity(0)),
                           "lscpu": subprocess.check_output(["lscpu"], text=True)},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_status": "new evaluator source individually SHA-bound; core codec unchanged"}
    if frozen_identity != identity():
        raise RuntimeError("Implementation changed during freeze")
    write_json(run / "freeze.json", record)
    for relative in frozen_identity["files"]:
        path = ROOT / relative
        if path.suffix in (".py", ".sh", ".hpp", ".cpp"):
            dst = run / "source_snapshot" / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    return {"series": len(bundles), "test_windows": sum(x["arrays"]["source"]["shape"][0] for x in bundles),
            "freeze_sha256": digest_file(run / "freeze.json")}


def gates(metrics: dict[str, float], epsilon: float, tau: float, rho: float) -> dict[str, bool]:
    t = float(PROTOCOL["float_acceptance_tolerance"])
    return {"point": metrics["max_abs_error"] <= epsilon + t,
            "acf": metrics["acf_max_abs"] <= tau + t,
            "pearson": metrics["correlation_series_min"] >= rho - t}


def run_one(run: Path, bundle: Any, dataset: str, fitted: Any, method: str,
            point: dict[str, Any], mode: str, ref: dict[str, Any] | None,
            freeze_record: dict[str, Any], freeze_sha: str) -> dict[str, Any]:
    from . import native_bridge, tasks_data
    series_id = str(get(bundle, "series_id"))
    task_key = f"{method}-{point['label']}-{mode}"
    row_path = run / "rows" / dataset / series_id / (task_key + ".json")
    if row_path.exists():
        existing = json.loads(row_path.read_text())
        if existing.get("freeze_sha256") != freeze_sha:
            raise RuntimeError("Resume row belongs to a different freeze")
        if existing["status"] == "complete":
            artifact_path = run / existing["artifact"]
            if digest_file(artifact_path) != existing["artifact_sha256"]:
                raise RuntimeError("Resume artifact has changed")
        return existing
    row: dict[str, Any] = {"dataset": dataset, "series_id": series_id, "method": method,
                           "point": point, "comparison": mode, "status": "running",
                           "freeze_sha256": freeze_sha, "reference": ref}
    started = time.perf_counter()
    try:
        source = get(bundle, "val_source" if mode == "calibration" else "source")
        encoded = native_bridge.encode(method, source, point["epsilon"], point["tau"], point["rho"])
        artifact = run / "artifacts" / dataset / series_id / (task_key + "-" + uuid.uuid4().hex[:10] + ".scs2")
        decoded, info = native_bridge.persist_replay(
            artifact, encoded, source, channel=str(get(bundle, "channel_name")),
            mean=float(get(bundle, "mean")), std=float(get(bundle, "std")),
            protocol_sha=freeze_record["protocol_sha256"], freeze_sha=freeze_sha,
            method_index=METHODS.index(method),
            point_index=int.from_bytes(hashlib.sha256(canonical(point)).digest()[:4], "little"),
        )
        check = gates(info["metrics"], point["epsilon"], point["tau"], point["rho"])
        own_ok = (check["point"] if method != "cameo" else True)
        if method in ("qpet_joint", "qpet_cf9", "qpet_legacy", "cameo_joint"):
            own_ok = all(check.values())
        elif method == "qpet_pearson":
            own_ok = check["point"] and check["pearson"]
        elif method == "qpet_acf":
            own_ok = check["point"] and check["acf"]
        row.update(status="complete", artifact=str(artifact.relative_to(run)), **info,
                   joint_gates=check, joint_feasible=all(check.values()), metadata=encoded.metadata)
        if not own_ok:
            raise RuntimeError(f"codec violated its enabled contract: {check}")
        if mode != "calibration":
            row["tasks"] = tasks_data.task_metrics(bundle, decoded, fitted)
        if mode == "matched_rate" and ref:
            ratio = info["artifact_bytes"] / ref["test_artifact_bytes"]
            row["test_byte_ratio_to_joint"] = ratio
            row["achieved_same_rate"] = abs(ratio - 1) <= PROTOCOL["same_rate_test_tolerance_fraction"]
        if mode == "matched_constraint" and ref:
            row["target_joint_gates"] = gates(info["metrics"], ref["epsilon"], ref["tau"], ref["rho"])
            row["target_joint_feasible"] = all(row["target_joint_gates"].values())
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                   traceback=traceback.format_exc())
    row["job_wall_seconds"] = time.perf_counter() - started
    write_json(row_path, row)
    with _LOCK:
        print(json.dumps({k: row[k] for k in ["dataset", "series_id", "method", "comparison", "status"]},
                         ensure_ascii=False), flush=True)
    return row


def series_worker(run: Path, dataset: str, bundle: Any, record: dict[str, Any], freeze_sha: str) -> None:
    from . import tasks_data
    if identity() != record["identity"]:
        raise RuntimeError("Frozen implementation/runtime changed")
    series_id = str(get(bundle, "series_id"))
    expected = next(x for x in record["series"] if x["dataset"] == dataset and str(x["series_id"]) == series_id)
    for key in BUNDLE_FIELDS:
        if normalize(get(bundle, key)) != expected[key]:
            raise RuntimeError(f"Frozen dataset metadata changed: {dataset}/{series_id}/{key}")
    for key, detail in expected["arrays"].items():
        a = np.ascontiguousarray(get(bundle, key))
        if hashlib.sha256(a.tobytes()).hexdigest() != detail["sha256"]:
            raise RuntimeError(f"Frozen data changed: {dataset}/{series_id}/{key}")
    fitted = tasks_data.select_ridge(get(bundle, "train_1d"), get(bundle, "val_source"),
                                     get(bundle, "val_future"))
    model_path = run / "models" / dataset / (series_id + ".json")
    model_record = normalize(fitted)
    if model_path.exists() and json.loads(model_path.read_text()) != model_record:
        raise RuntimeError("Validation-only model selection changed on resume")
    write_json(model_path, model_record)
    # Calibrate before decoding any TEST row for this source. The selection rule,
    # candidate grid and all inputs were fixed before any run started.
    calibration: dict[str, list[dict[str, Any]]] = {}
    for method in BASELINES:
        calibration[method] = []
        for epsilon in CALIBRATION_EPSILONS:
            point = dict(label=f"e{epsilon:g}", epsilon=epsilon, tau=0.02, rho=0.995)
            calibration[method].append(run_one(run, bundle, dataset, fitted, method, point,
                                              "calibration", None, record, freeze_sha))
    joint_validation = {}
    for point in POINTS:
        joint_validation[point["label"]] = run_one(run, bundle, dataset, fitted, "qpet_joint", point,
                                                   "calibration", None, record, freeze_sha)
    selection = {}
    for point in POINTS:
        choices = {}
        target = joint_validation[point["label"]]
        for method, rows in calibration.items():
            valid = [x for x in rows if x["status"] == "complete"]
            rate = (min(valid, key=lambda x: (abs(math.log(x["artifact_bytes"] / target["artifact_bytes"])),
                                             x["point"]["epsilon"]))
                    if valid and target["status"] == "complete" else None)
            feasible = [x for x in valid if all(gates(x["metrics"], point["epsilon"], point["tau"], point["rho"]).values())]
            contract = min(feasible, key=lambda x: (x["artifact_bytes"], x["point"]["epsilon"])) if feasible else None
            choices[method] = {"matched_rate_epsilon": rate["point"]["epsilon"] if rate else None,
                               "matched_constraint_epsilon": contract["point"]["epsilon"] if contract else None}
        selection[point["label"]] = choices
    write_json(run / "selections" / dataset / (series_id + ".json"), selection)
    for point in POINTS:
        reference = run_one(run, bundle, dataset, fitted, "qpet_joint", point, "same_epsilon", None, record, freeze_sha)
        for method in METHODS:
            if method == "qpet_joint" or (method == "zstd" and point != POINTS[0]):
                continue
            run_one(run, bundle, dataset, fitted, method, point, "same_epsilon", None, record, freeze_sha)
        for method in BASELINES:
            for mode in ["matched_rate", "matched_constraint"]:
                epsilon = selection[point["label"]][method][mode + "_epsilon"]
                if epsilon is None or reference["status"] != "complete":
                    write_json(run / "rows" / dataset / series_id / f"{method}-{point['label']}-{mode}.json",
                               {"dataset": dataset, "series_id": series_id, "method": method,
                                "comparison": mode, "point": point, "status": "unavailable",
                                "reason": "No feasible validation grid point or joint reference failed",
                                "freeze_sha256": freeze_sha})
                    continue
                run_one(run, bundle, dataset, fitted, method, dict(point, epsilon=epsilon), mode,
                        dict(point, test_artifact_bytes=reference["artifact_bytes"]), record, freeze_sha)
    if identity() != record["identity"]:
        raise RuntimeError("Frozen implementation/runtime changed during source execution")


def status(run: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    datasets: dict[str, dict[str, int]] = {}
    rows = []
    for path in sorted((run / "rows").glob("*/*/*.json")):
        row = json.loads(path.read_text())
        rows.append(row)
        key = row["status"]
        counts[key] = counts.get(key, 0) + 1
        dst = datasets.setdefault(row["dataset"], {})
        dst[key] = dst.get(key, 0) + 1
    result = {"updated_utc": datetime.now(timezone.utc).isoformat(), "rows": len(rows),
              "counts": counts, "datasets": datasets}
    write_json(run / "STATUS.json", result)
    return result


def run_all(run: Path, workers: int) -> None:
    from . import native_bridge, tasks_data
    native_bridge.install_linux_transport()
    record = json.loads((run / "freeze.json").read_text())
    if record["protocol"] != PROTOCOL or record["identity"] != identity():
        raise RuntimeError("Protocol/implementation/runtime does not match frozen run")
    freeze_sha = digest_file(run / "freeze.json")
    write_json(run / "RUNNING.json", {"pid": os.getpid(), "workers": workers,
                                      "started_utc": datetime.now(timezone.utc).isoformat(),
                                      "freeze_sha256": freeze_sha})
    jobs = [(name, bundle) for name in DATASETS for bundle in tasks_data.prepare_dataset(name)]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(series_worker, run, name, bundle, record, freeze_sha):
                   (name, str(get(bundle, "series_id"))) for name, bundle in jobs}
        for future in as_completed(futures):
            name, series_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failure = {"dataset": name, "series_id": series_id, "error": str(exc),
                           "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(run / "source_failures" / name / (series_id + ".json"), failure)
            snapshot = status(run)
            print(json.dumps(snapshot), flush=True)
    write_json(run / "FINISHED.json", {**status(run), "source_failures": failures,
                                       "all_sources_processed": True,
                                       "all_rows_passed": not failures and not status(run)["counts"].get("failed", 0)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.output), indent=2))
    elif args.command == "run":
        if args.workers != 2:
            raise ValueError("Concurrency is fixed at two workers by this protocol")
        run_all(args.output, args.workers)
    else:
        print(json.dumps(status(args.output), indent=2))


if __name__ == "__main__":
    main()
