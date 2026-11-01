"""Inference-only foundation-model extension; existing experiments remain read-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PATCH = ROOT / "experiments/results/vldb_patchtst_full_20260915_v1"
MAIN = ROOT / "experiments/results/vldb_edbt_main_20260907_v1"
DEFAULT_RUN = ROOT / "experiments/results/vldb_foundation_20260922_v1"
LOCK = "/tmp/nclab_downstream_gpu_20260915.lock"
LEVELS = [i / 10 for i in range(1, 10)]
PROTOCOL = {
    "schema": "nclab.foundation-frozen-consumer.v1",
    "context": 96, "horizon": 24, "compression_window": 128,
    "quantile_levels": LEVELS, "point_prediction": "median (quantile 0.5), not mean",
    "scope": "Frozen pretrained univariate inference; no fitting, finetuning, new compression, covariates, or TEST selection",
    "inputs": "Identical TRAIN-standardized float32 replay as the accepted PatchTST experiment; final 96 of 128 history values only",
    "normalization": "Original per-series TRAIN mean/std retained; any model-internal normalization sees only its own supplied context",
    "targets": "Same uncompressed 24-point future, never passed to predictor",
    "coverage": "142 series, six sources, 32 origins per series; all 67 existing TEST configurations; exclude 8236 calibration rows",
    "aggregation": "Origins within series, then series within source, then six sources equally; Solar is one source with 137 series",
    "rate": "Full original SCS2 bytes including metadata and normalizer; selected disjoint histories only, not the complete continuous series",
    "matched_rate": "Original VAL-selected settings unchanged; paired TEST subset with original achieved_same_rate flag, cross-checked against plus/minus 5 percent full bytes",
    "matched_constraint": "Original VAL-selected settings unchanged; TEST eligibility uses all original joint target gates",
    "failure": "Raw-good means raw-model MSE below fixed raw persistence; compressed usefulness loss means no longer below that same reference",
    "linear_reference": "Fixed existing raw Ridge and DLinear source means; advantage retention defined only where raw foundation model beats that reference",
    "uncertainty": "Pinball averaged across nine quantiles and 24 steps; 80 percent interval uses q0.1/q0.9; empirical coverage and width reported separately",
    "fairness": "Common short context, horizon and median target; not an original foundation-model leaderboard reproduction or a claim of unseen pretraining data",
    "reproducibility": "One fixed inference seed 20260922; repeated deterministic inference is not independent training evidence",
    "batching": "Per series exact-equality dedup of complete 32-origin consumer batches; raw first; one bulk adapter call with fixed internal batch size; no cross-series learning",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4", order="C").tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def code_identity():
    return {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), Path(__file__).with_name("models.py"))}


def verify(run):
    f = read(run / "freeze.json")
    if f["protocol"] != PROTOCOL or f["code_sha256"] != code_identity():
        raise RuntimeError("frozen inference protocol/code changed; use a new run directory")
    if f["models_manifest_sha256"] != sha(run / "models.json"):
        raise RuntimeError("model manifest changed")
    if f["patch_freeze_sha256"] != sha(PATCH / "freeze.json") or f["main_freeze_sha256"] != sha(MAIN / "freeze.json"):
        raise RuntimeError("upstream experiment freeze changed")
    return f


def freeze(run):
    if (run / "freeze.json").exists():
        return verify(run)
    manifest = read(run / "models.json")
    if not isinstance(manifest.get("models"), dict) or not manifest["models"]:
        raise ValueError("models.json must contain a nonempty models object")
    pf, mf = read(PATCH / "freeze.json"), read(MAIN / "freeze.json")
    psha = sha(PATCH / "freeze.json")
    if pf["main_freeze_sha256"] != sha(MAIN / "freeze.json"):
        raise RuntimeError("upstream compression binding differs")
    if not read(PATCH / "INPUT_AUDIT.json")["passed"] or not read(PATCH / "RESULTS_AUDIT.json")["passed"]:
        raise RuntimeError("upstream audits did not pass")
    main_series = {s["series_id"]: s for s in mf["series"]}
    rows_by_path = {r["path"]: r for r in pf["row_manifest"]}
    files, series, count = {}, [], 0
    for s in pf["series"]:
        ds, sid = s["dataset"], s["series_id"]
        if any(s[k] != main_series[sid][k] for k in ("dataset", "channel_name", "mean", "std")):
            raise RuntimeError("original TRAIN normalization/series metadata changed")
        ip, cp = PATCH / "inputs" / ds / (sid + ".npz"), PATCH / "cache" / ds / (sid + ".npz")
        cm_path = cp.with_suffix(".json")
        cm = read(cm_path)
        if sha(ip) != pf["inputs_sha256"][str(ip.relative_to(PATCH))] or sha(cp) != cm["npz_sha256"] or cm["freeze_sha256"] != psha:
            raise RuntimeError("upstream input/cache binding differs: " + sid)
        with np.load(ip, allow_pickle=False) as z:
            if z["source"].shape != (32, 1, 128) or z["future"].shape != (32, 24) or z["source"].dtype != np.float32 or z["future"].dtype != np.float32:
                raise RuntimeError("unexpected evaluation support")
            if array_sha(z["source"]) != s["source_sha256"] or array_sha(z["future"]) != s["future_sha256"]:
                raise RuntimeError("source/future array hash mismatch")
            starts = z["starts"]
            ms = main_series[sid]
            if not np.array_equal(starts, ms["starts"]) or np.any(np.diff(starts) < 152):
                raise RuntimeError("origin identity or disjointness violation")
            if starts[0] + 128 < ms["split_boundaries"]["test"][0] or starts[-1] + 152 > ms["split_boundaries"]["test"][1]:
                raise RuntimeError("future outside original TEST split")
        quality = []
        with np.load(cp, allow_pickle=False) as z:
            for r in cm["rows"]:
                rp = MAIN / "rows" / ds / sid / r["row_name"]
                expected = rows_by_path[str(rp.relative_to(MAIN))]["sha256"]
                if sha(rp) != expected or r["compression_row_sha256"] != expected or r["comparison"] == "calibration":
                    raise RuntimeError("original compression row binding differs")
                old = read(rp)
                if r["status"] != "complete" or z[r["cache_key"]].shape != (32, 128) or array_sha(z[r["cache_key"]]) != r["decoded_sha256"]:
                    raise RuntimeError("incomplete or corrupt decoded replay: " + str(rp))
                q = dict(r)
                q.update(raw_bytes=old["raw_bytes"], cr=old["raw_bytes"] / old["artifact_bytes"],
                         bits_per_value=32 * old["artifact_bytes"] / old["raw_bytes"])
                target = old.get("reference") or old["point"]
                met = old["metrics"]
                q["target_joint_pass"] = bool(met["max_abs_error"] <= target["epsilon"] + 1e-7 and
                    met["correlation_series_min"] >= target["rho"] - 1e-7 and met["acf_max_abs"] <= target["tau"] + 1e-7)
                quality.append(q)
                count += 1
        if len(quality) != 67:
            raise RuntimeError("incomplete configuration panel")
        for p in (ip, cp, cm_path):
            files[str(p.relative_to(ROOT))] = sha(p)
        series.append(dict(s, input_path=str(ip.relative_to(ROOT)), cache_path=str(cp.relative_to(ROOT)),
                           cache_metadata_path=str(cm_path.relative_to(ROOT)), rows=quality))
    if count != 9514 or len(series) != 142:
        raise RuntimeError("unexpected overall panel size")
    linear = PATCH / "supplements/linear_reference_20260915/linear_raw_source.csv"
    files[str(linear.relative_to(ROOT))] = sha(linear)
    f = dict(protocol=PROTOCOL, created_utc=now(), models=list(manifest["models"]),
             models_manifest_sha256=sha(run / "models.json"), code_sha256=code_identity(),
             patch_freeze_sha256=psha, main_freeze_sha256=sha(MAIN / "freeze.json"),
             input_files_sha256=files, series=series, artifact_rows_per_model=count,
             decoded_origins_per_model=count * 32, unique_raw_origins=4544,
             linear_reference_path=str(linear.relative_to(ROOT)))
    write_json(run / "freeze.json", f)
    for rel in f["code_sha256"]:
        dest = run / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    shutil.copyfile(run / "models.json", run / "source_snapshot/models.json")
    return dict(freeze_sha256=sha(run / "freeze.json"), rows=count, series=len(series))


@contextmanager
def gpu_lock():
    import fcntl
    with open(LOCK, "a") as f:
        print(json.dumps({"waiting_for_gpu_lock": LOCK, "utc": now()}), flush=True)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def checked_predict(model, values):
    x = np.ascontiguousarray(values[:, -96:], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 96 or not np.isfinite(x).all():
        raise ValueError("invalid inference contexts")
    start = time.perf_counter()
    out = model.predict(x)
    point, quantiles = np.asarray(out["point"]), np.asarray(out["quantiles"])
    if point.shape != (len(x), 24) or quantiles.shape != (len(x), 24, 9):
        raise ValueError(f"invalid prediction shapes: {point.shape}, {quantiles.shape}")
    if not np.allclose(out["quantile_levels"], LEVELS, atol=1e-10, rtol=0) or not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise ValueError("invalid quantile levels or nonfinite output")
    if not np.allclose(point, quantiles[:, :, 4], atol=1e-6, rtol=1e-6):
        raise ValueError("point prediction must be the q0.5 median")
    return dict(point=point, quantiles=quantiles, seconds=time.perf_counter() - start)


def metrics(future, raw, pred, persistence):
    y, p, q = [np.asarray(v, dtype=np.float64) for v in (future, pred["point"], pred["quantiles"])]
    rmse = np.mean((np.asarray(raw["point"], dtype=np.float64) - y) ** 2, axis=1)
    mse = np.mean((p - y) ** 2, axis=1)
    naive = np.mean((persistence.astype(np.float64) - y) ** 2, axis=1)
    errors = y[:, :, None] - q
    pinball = np.maximum(np.asarray(LEVELS) * errors, (np.asarray(LEVELS) - 1) * errors)
    covered = (y >= q[:, :, 0]) & (y <= q[:, :, -1])
    good, failed = rmse < naive, (rmse < naive) & (mse >= naive)
    return dict(mse=float(mse.mean()), mae=float(np.abs(p-y).mean()), raw_mse=float(rmse.mean()),
        excess_mse=float(np.mean(mse-rmse)), prediction_mse=float(np.mean((p-raw["point"]) ** 2)),
        pinball=float(pinball.mean()), coverage80=float(covered.mean()), width80=float(np.mean(q[:, :, -1]-q[:, :, 0])),
        quantile_crossing_fraction=float(np.mean(np.any(np.diff(q, axis=-1) < 0, axis=-1))),
        naive_mse=float(naive.mean()), raw_good_count=int(good.sum()), usefulness_failure_count=int(failed.sum()),
        catastrophic_count=int(((mse >= 10*np.maximum(rmse, 1e-12)) & (mse >= naive)).sum()), origins=len(y),
        per_origin_mse=mse.tolist(), per_origin_raw_mse=rmse.tolist(), per_origin_naive_mse=naive.tolist(),
        per_origin_pinball=pinball.mean(axis=(1,2)).tolist())


def checked_inputs(frozen, s):
    for name in ("input_path", "cache_path", "cache_metadata_path"):
        rel = s[name]
        if sha(ROOT / rel) != frozen["input_files_sha256"][rel]:
            raise RuntimeError("frozen input changed: " + rel)
    with np.load(ROOT / s["input_path"], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    return a


def evaluate(run, name, smoke=False):
    frozen = verify(run)
    if name not in frozen["models"]:
        raise ValueError("model not in frozen roster")
    from .models import load_model
    with gpu_lock():
        model = load_model(name, read(run / "models.json")["models"][name])
        metadata = dict(model.metadata)
        fsha = sha(run / "freeze.json")
        if smoke:
            s = frozen["series"][0]
            a = checked_inputs(frozen, s)
            out = checked_predict(model, a["source"][:2, 0, :])
            result = dict(model=name, shape=list(out["point"].shape), quantile_shape=list(out["quantiles"].shape),
                          metadata=metadata, seconds=out["seconds"], passed=True, freeze_sha256=fsha)
            write_json(run / "smoke" / (name + ".json"), result)
            return result
        total, errors = 0, 0
        for s in frozen["series"]:
            verify(run)
            output = run / "evaluation" / name / s["dataset"] / (s["series_id"] + ".json")
            if output.exists():
                done = read(output)
                if done["freeze_sha256"] != fsha:
                    raise RuntimeError("existing result belongs to a different freeze")
                if done.get("status") == "complete":
                    if sha(output.with_suffix(".npz")) != done["predictions_sha256"]:
                        raise RuntimeError("resumed prediction hash mismatch")
                    total += len(done["rows"])
                    continue
            started = time.perf_counter()
            a = checked_inputs(frozen, s)
            try:
                raw_x = np.ascontiguousarray(a["source"][:, 0, -96:], dtype=np.float32)
                unique = [raw_x]
                prediction_cache = {array_sha(raw_x): (raw_x, 0, "raw")}
                row_mapping = {}
                with np.load(ROOT / s["cache_path"], allow_pickle=False) as cache:
                    for entry in s["rows"]:
                        xx = np.ascontiguousarray(cache[entry["cache_key"]][:, -96:], dtype=np.float32)
                        input_hash = array_sha(xx)
                        saved = prediction_cache.get(input_hash)
                        if saved is not None:
                            if not np.array_equal(xx, saved[0]):
                                raise RuntimeError("input digest collision")
                            row_mapping[entry["cache_key"]] = (saved[1], input_hash, saved[2])
                        else:
                            index = len(unique)
                            unique.append(xx)
                            prediction_cache[input_hash] = (xx, index, entry["cache_key"])
                            row_mapping[entry["cache_key"]] = (index, input_hash, None)
                bulk = checked_predict(model, np.concatenate(unique))
                def part(index):
                    return dict(point=bulk["point"][index*32:(index+1)*32], quantiles=bulk["quantiles"][index*32:(index+1)*32])
                raw = part(0)
                naive = np.repeat(a["source"][:, 0, -1:], 24, axis=1)
                raw_metrics = metrics(a["future"], raw, raw, naive)
                predictions = dict(raw_point=raw["point"], raw_quantiles=raw["quantiles"], future=a["future"], starts=a["starts"], persistence=naive)
                rows = []
                with np.load(ROOT / s["cache_path"], allow_pickle=False) as cache:
                    for entry in s["rows"]:
                        row = dict(entry)
                        try:
                            index, input_hash, reused = row_mapping[row["cache_key"]]
                            pred = part(index)
                            if reused is not None:
                                row["reused_from"] = reused
                                row["reused_from_input_sha256"] = input_hash
                            row["consumer_input_sha256"] = input_hash
                            predictions[row["cache_key"] + "_point"] = pred["point"]
                            predictions[row["cache_key"] + "_quantiles"] = pred["quantiles"]
                            row.update(metrics=metrics(a["future"], raw, pred, naive), inference_seconds=None, status="complete")
                        except Exception as exc:
                            row.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
                            errors += 1
                        rows.append(row)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".npz.tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, **predictions)
                temporary.replace(output.with_suffix(".npz"))
                complete = all(r["status"] == "complete" for r in rows)
                record = dict(model=name, dataset=s["dataset"], series_id=s["series_id"], status="complete" if complete else "partial",
                    rows=rows, raw_metrics=raw_metrics, raw_inference_seconds=None, bulk_inference_seconds=bulk["seconds"],
                    unique_consumer_batches=len(unique), evaluated_consumer_windows=len(unique)*32, metadata=metadata,
                    freeze_sha256=fsha, predictions_sha256=sha(output.with_suffix(".npz")),
                    wall_seconds=time.perf_counter()-started, finished_utc=now())
                write_json(output, record)
                total += sum(r["status"] == "complete" for r in rows)
            except Exception as exc:
                errors += 67
                write_json(output, dict(model=name, dataset=s["dataset"], series_id=s["series_id"], status="failed",
                    error=repr(exc), traceback=traceback.format_exc(), rows=[], freeze_sha256=fsha, metadata=metadata))
            state = dict(model=name, last_series=s["series_id"], complete_rows=total, failed_rows=errors, utc=now())
            write_json(run / "status" / (name + ".json"), state)
            print(json.dumps(state), flush=True)
        result = dict(model=name, complete_rows=total, failed_rows=errors, expected_rows=9514,
                      all_complete=total == 9514 and errors == 0, metadata=metadata, freeze_sha256=fsha, finished_utc=now())
        write_json(run / "finished" / (name + ".json"), result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "smoke", "evaluate", "all"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == "freeze":
        result = freeze(run)
    elif args.command == "all":
        freeze(run)
        result = [evaluate(run, name) for name in verify(run)["models"]]
    else:
        if not args.model:
            parser.error("--model is required for smoke/evaluate")
        result = evaluate(run, args.model, smoke=args.command == "smoke")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
