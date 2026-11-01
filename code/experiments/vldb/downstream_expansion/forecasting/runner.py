"""Official PatchTST architecture, TRAIN/VAL-only fitting, persisted TEST replay.

All CUDA work is dispatched through the shared WSL flock. No compression runs.
This declared-budget experiment is not a reproduction of the original SOTA table.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from experiments.vldb.main_benchmark import tasks_data as data

ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "experiments/results/vldb_edbt_main_20260907_v1"
DEFAULT_RUN = ROOT / "experiments/results/vldb_patchtst_full_20260915_v1"
OFFICIAL = ROOT / "tmp/upstreams/PatchTST-20260915/PatchTST_supervised"
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
LOCK = "/tmp/nclab_downstream_gpu_20260915.lock"
DATASETS = ("ETTm1", "ETTm2", "Weather", "ElecDem", "Wind", "Solar")
SEEDS = (20260915, 20260916, 20260917)
MODEL = dict(enc_in=1, seq_len=96, pred_len=24, e_layers=3, n_heads=4,
             d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3, head_dropout=0.,
             individual=False, patch_len=16, stride=8, padding_patch="end",
             revin=True, affine=False, subtract_last=False, decomposition=False,
             kernel_size=25)
CONFIG = dict(model=MODEL, seeds=list(SEEDS), epochs=20, patience=5,
              batch_size=128, learning_rate=0.0001, train_pair_cap_per_source=32768,
              optimizer="Adam", loss="MSE", lr_schedule="constant, declared adaptation",
              train_sampling="Allocate cap equally across all source series (first series receive remainder); each series uses integer-equidistant starts across all TRAIN-only 96+24 pairs; deterministic and value-independent",
              solar="One shared univariate model, all 137 series represented in TRAIN and VAL; no per-series fitting",
              validation="Same frozen eight raw VAL origins per series; minimum series-equal raw MSE, ties earliest epoch; patience five",
              evaluation="All 9514 original TEST artifacts, 142 series, 4544 unique origins, all 13 methods and all three comparison modes; calibration excluded",
              failure="Raw-good means raw-model MSE < fixed raw persistence MSE; usefulness loss means decoded-model MSE >= that persistence MSE on raw-good origins; report both rescue and new failure",
              catastrophic="Decoded MSE >= 10 * max(raw MSE,1e-12) AND decoded MSE >= fixed raw persistence; also report actual and excess MSE",
              aggregation="Mean over origins then series within each source, then six sources; seeds reported individually and arithmetic mean/sample SD; counts remain pooled",
              case_selection="Exploratory only: greatest dual-minus-triple MSE among raw-good qpet_pearson failures rescued by qpet_joint, alongside reverse counts; no task or hyperparameter tuning",
              compute="WSL shared flock per dataset/seed train or inference job; no CUDA model allocation before lock; deterministic torch FP32, no TF32; 18 models maximum, 360 epochs maximum",
              claim="Unmodified official PatchTST architecture under declared short-context and bounded-training protocol; not original paper SOTA reproduction")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity():
    paths = [Path(__file__), Path(data.__file__),
             ROOT / "experiments/vldb/main_benchmark/native_bridge.py",
             ROOT / "experiments/vldb/utility/artifacts.py"]
    paths += [OFFICIAL / p for p in ("models/PatchTST.py", "layers/PatchTST_backbone.py",
                                    "layers/PatchTST_layers.py", "layers/RevIN.py")]
    return {str(p.relative_to(ROOT)): data.file_sha256(p) for p in paths}


def verify(run):
    freeze = read(run / "freeze.json")
    if freeze["config"] != CONFIG or freeze["code_sha256"] != identity():
        raise RuntimeError("frozen forecasting code or protocol changed")
    if data.file_sha256(MAIN / "freeze.json") != freeze["main_freeze_sha256"]:
        raise RuntimeError("main compression freeze changed")
    return freeze


def model():
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    spec = importlib.util.spec_from_file_location("frozen_official_patchtst", OFFICIAL / "models/PatchTST.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model(SimpleNamespace(**MODEL)).float()


def runtime():
    if os.environ.get("NCLAB_GPU_LOCK_HELD") != "1":
        raise RuntimeError("GPU entry must be invoked by WSL shared flock dispatcher")
    import torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), deterministic=True, tf32=False)


def freeze_run(run):
    if (run / "freeze.json").exists():
        return verify(run)
    main = read(MAIN / "freeze.json")
    ident = identity()
    actual_commit = subprocess.check_output(["git", "-C", str(OFFICIAL.parent), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != COMMIT:
        raise RuntimeError("official checkout commit mismatch")
    all_series, row_manifest, input_hashes = [], [], {}
    for dataset in DATASETS:
        bundles = data.prepare_dataset(dataset)
        train_pairs, train_indices, val_x, val_y = [], [], [], []
        nseries = len(bundles)
        for j, b in enumerate(bundles):
            old = next(s for s in main["series"] if s["series_id"] == b["series_id"])
            for key in ("source", "future", "val_source", "val_future"):
                if old["arrays"][key]["sha256"] != data.array_sha256(b[key]):
                    raise RuntimeError(f"original main array differs: {b['series_id']} {key}")
            count = len(b["train_1d"]) - 119
            budget = CONFIG["train_pair_cap_per_source"] // nseries + (j < CONFIG["train_pair_cap_per_source"] % nseries)
            chosen = min(count, budget)
            starts = np.arange(chosen, dtype=np.int64) * (count - 1) // max(chosen - 1, 1)
            train_pairs.append(np.stack([b["train_1d"][k:k+120] for k in starts]))
            train_indices += [(j, int(k)) for k in starts]
            val_x.append(b["val_source"][:, 0, -96:])
            val_y.append(b["val_future"])
            series_path = run / "inputs" / dataset / (b["series_id"] + ".npz")
            series_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(series_path, source=b["source"], future=b["future"], starts=b["starts"])
            input_hashes[str(series_path.relative_to(run))] = data.file_sha256(series_path)
            all_series.append({"dataset": dataset, "series_id": b["series_id"], "mean": b["mean"], "std": b["std"],
                               "channel_name": b["channel_name"], "source_sha256": data.array_sha256(b["source"]),
                               "future_sha256": data.array_sha256(b["future"]), "train_sha256": data.array_sha256(b["train_1d"]),
                               "train_pairs_available": count, "train_pairs_selected": chosen})
            for rp in sorted((MAIN / "rows" / dataset / b["series_id"]).glob("*.json")):
                row = read(rp)
                if row["comparison"] != "calibration":
                    row_manifest.append({"path": str(rp.relative_to(MAIN)), "sha256": data.file_sha256(rp),
                                         "dataset": dataset, "series_id": b["series_id"], "status": row["status"]})
        train_path = run / "inputs" / dataset / "TRAIN_VAL.npz"
        np.savez(train_path, pairs=np.concatenate(train_pairs), selected_indices=np.asarray(train_indices),
                 val_x=np.concatenate(val_x), val_y=np.concatenate(val_y))
        input_hashes[str(train_path.relative_to(run))] = data.file_sha256(train_path)
        print(json.dumps({"prepared": dataset, "series": nseries, "training_pairs": len(train_indices)}), flush=True)
    if len(row_manifest) != 9514 or len(all_series) != 142:
        raise RuntimeError("unexpected original TEST coverage")
    if ident != identity():
        raise RuntimeError("code changed while freezing")
    record = dict(schema="nclab.patchtst-full.v1", created_utc=now(), config=CONFIG,
                  main_freeze_sha256=data.file_sha256(MAIN / "freeze.json"),
                  official_repository="https://github.com/yuqinie98/PatchTST", official_commit=COMMIT,
                  code_sha256=ident, inputs_sha256=input_hashes, series=all_series, row_manifest=row_manifest)
    write_json(run / "freeze.json", record)
    for rel in ident:
        target = run / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return {"freeze_sha256": data.file_sha256(run / "freeze.json"), "test_rows": 9514, "unique_test_windows": 4544}


def inputs(run, relative):
    expected = verify(run)["inputs_sha256"][relative]
    path = run / relative
    if data.file_sha256(path) != expected:
        raise RuntimeError("frozen input file changed: " + relative)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cache(run):
    frozen = verify(run)
    from experiments.vldb.main_benchmark import native_bridge
    from experiments.vldb.main_benchmark.benchmark import identity as main_identity
    from experiments.vldb.utility.artifacts import parse_scs2
    if main_identity() != read(MAIN / "freeze.json")["identity"]:
        raise RuntimeError("native decoder differs from original compression freeze")
    native_bridge.install_linux_transport()
    total = Counter()
    for s in frozen["series"]:
        path = run / "cache" / s["dataset"] / (s["series_id"] + ".npz")
        if path.with_suffix(".json").exists():
            meta = read(path.with_suffix(".json"))
            if data.file_sha256(path) != meta["npz_sha256"]:
                raise RuntimeError("replay cache hash mismatch")
            total.update(x["status"] for x in meta["rows"])
            continue
        rows, arrays, start = [], {}, time.perf_counter()
        for ent in frozen["row_manifest"]:
            if ent["series_id"] != s["series_id"]:
                continue
            rp = MAIN / ent["path"]
            if data.file_sha256(rp) != ent["sha256"]:
                raise RuntimeError("compression row changed")
            old = read(rp)
            row = dict(dataset=s["dataset"], series_id=s["series_id"], method=old["method"],
                       point=old["point"], comparison=old["comparison"], row_name=rp.name,
                       compression_row_sha256=ent["sha256"], status=old["status"],
                       artifact_bytes=old.get("artifact_bytes"), artifact_sha256=old.get("artifact_sha256"),
                       original_reference=old.get("reference"), original_matched_rate=old.get("achieved_same_rate"),
                       original_target_feasible=old.get("target_joint_feasible"))
            started = time.perf_counter()
            try:
                if old["status"] != "complete":
                    row.update(status="upstream_" + old["status"], error=old.get("error"))
                else:
                    parsed = parse_scs2(MAIN / old["artifact"])
                    if parsed.sha256 != old["artifact_sha256"] or len(parsed.payloads) != 1:
                        raise RuntimeError("compressed artifact SHA/payload mismatch")
                    if parsed.channels != (s["channel_name"],) or float(parsed.mean[0]) != s["mean"] or float(parsed.std[0]) != s["std"]:
                        raise RuntimeError("SCS2 normalizer mismatch")
                    x = native_bridge.decode_payload(parsed.codec_id, parsed.payloads[0], (32, 1, 128))
                    if x.shape != (32, 1, 128) or x.dtype != np.float32 or not np.isfinite(x).all():
                        raise RuntimeError("invalid persisted decode")
                    key = "r" + str(len(arrays))
                    arrays[key] = x[:, 0, :]
                    row.update(cache_key=key, decoded_sha256=data.array_sha256(x))
            except Exception as exc:
                row.update(status="failed", error=repr(exc))
            row["replay_wall_seconds"] = time.perf_counter() - started
            rows.append(row)
            total[row["status"]] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        write_json(path.with_suffix(".json"), dict(rows=rows, npz_sha256=data.file_sha256(path),
                   seconds=time.perf_counter() - start, freeze_sha256=data.file_sha256(run / "freeze.json")))
        print(json.dumps({"cached": s["series_id"], "counts": dict(total)}), flush=True)
    write_json(run / "CACHE_FINISHED.json", dict(counts=dict(total), finished_utc=now()))
    return dict(total)


def predict(net, values, batch=512):
    import torch
    result = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch):
            x = torch.as_tensor(values[start:start+batch, -96:, None].copy(), device="cuda")
            result.append(net(x).cpu().numpy()[:, :, 0])
    output = np.concatenate(result)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite PatchTST output")
    return output


def train(run, dataset, seed):
    verify(run)
    rt = runtime()
    import torch
    folder = run / "models" / str(seed)
    checkpoint = folder / (dataset + ".npz")
    if checkpoint.with_suffix(".json").exists():
        load_model(run, dataset, seed)
        return {"already_trained": dataset, "seed": seed}
    a = inputs(run, f"inputs/{dataset}/TRAIN_VAL.npz")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = model().to("cuda")
    opt = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])
    pairs = torch.as_tensor(a["pairs"].copy(), device="cuda")
    history, best, best_epoch, stale, state = [], float("inf"), 0, 0, None
    start = time.perf_counter()
    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.perf_counter()
        net.train()
        order = torch.randperm(len(pairs), device="cuda")
        total = torch.zeros((), device="cuda")
        for begin in range(0, len(pairs), CONFIG["batch_size"]):
            chunk = pairs[order[begin:begin+CONFIG["batch_size"]]]
            opt.zero_grad(set_to_none=True)
            loss = ((net(chunk[:, :96, None])[:, :, 0] - chunk[:, 96:]) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.detach() * len(chunk)
        val = predict(net, a["val_x"])
        vl = float(np.mean((val.astype(np.float64) - a["val_y"]) ** 2))
        if not np.isfinite(vl):
            raise RuntimeError("nonfinite validation loss")
        if vl < best:
            best, best_epoch, stale = vl, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        entry = dict(epoch=epoch, train_mse=float((total / len(pairs)).item()), validation_mse=vl,
                     seconds=time.perf_counter() - epoch_start)
        history.append(entry)
        print(json.dumps(dict(dataset=dataset, seed=seed, **entry)), flush=True)
        if stale >= CONFIG["patience"]:
            break
    folder.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as f:
        np.savez_compressed(f, **{k: v.numpy() for k, v in state.items()})
    verify(run)
    meta = dict(dataset=dataset, seed=seed, training_seconds=time.perf_counter()-start,
                training_pairs=len(pairs), selected_epoch=best_epoch, validation_mse=best, epochs=history,
                checkpoint_sha256=data.file_sha256(checkpoint), freeze_sha256=data.file_sha256(run / "freeze.json"),
                runtime=rt, parameters=sum(p.numel() for p in net.parameters()), finished_utc=now())
    write_json(checkpoint.with_suffix(".json"), meta)
    return meta


def load_model(run, dataset, seed):
    import torch
    checkpoint = run / "models" / str(seed) / (dataset + ".npz")
    meta = read(checkpoint.with_suffix(".json"))
    if data.file_sha256(checkpoint) != meta["checkpoint_sha256"] or data.file_sha256(run / "freeze.json") != meta["freeze_sha256"]:
        raise RuntimeError("checkpoint binding mismatch")
    net = model().to("cuda")
    with np.load(checkpoint, allow_pickle=False) as z:
        net.load_state_dict({k: torch.from_numpy(z[k]).to("cuda") for k in z.files})
    net.eval()
    return net, meta


def eval_metrics(future, raw, pred, naive):
    err = np.mean((pred.astype(np.float64) - future) ** 2, axis=1)
    rawerr = np.mean((raw.astype(np.float64) - future) ** 2, axis=1)
    naiveerr = np.mean((naive.astype(np.float64) - future) ** 2, axis=1)
    good = rawerr < naiveerr
    fail = good & (err >= naiveerr)
    catastrophic = (err >= 10 * np.maximum(rawerr, 1e-12)) & (err >= naiveerr)
    return dict(raw_mse=float(rawerr.mean()), mse=float(err.mean()), excess_mse=float((err-rawerr).mean()),
                naive_mse=float(naiveerr.mean()), prediction_mse=float(np.mean((pred.astype(np.float64)-raw)**2)),
                raw_good_count=int(good.sum()), usefulness_failure_count=int(fail.sum()),
                catastrophic_count=int(catastrophic.sum()), origins=len(err),
                per_origin_mse=err.tolist(), per_origin_raw_mse=rawerr.tolist(), per_origin_naive_mse=naiveerr.tolist())


def evaluate(run, dataset, seed):
    frozen = verify(run)
    rt = runtime()
    net, meta = load_model(run, dataset, seed)
    if rt != meta["runtime"]:
        raise RuntimeError("training/evaluation runtime differs")
    total = Counter()
    for s in frozen["series"]:
        if s["dataset"] != dataset:
            continue
        start = time.perf_counter()
        sid = s["series_id"]
        output = run / "evaluation" / str(seed) / dataset / (sid + ".json")
        if output.exists():
            old = read(output)
            if old["checkpoint_sha256"] != meta["checkpoint_sha256"]:
                raise RuntimeError("evaluation model mismatch")
            total.update(r["status"] for r in old["rows"])
            continue
        a = inputs(run, f"inputs/{dataset}/{sid}.npz")
        cp = run / "cache" / dataset / (sid + ".npz")
        cm = read(cp.with_suffix(".json"))
        if data.file_sha256(cp) != cm["npz_sha256"] or cm["freeze_sha256"] != data.file_sha256(run / "freeze.json"):
            raise RuntimeError("reconstruction cache differs from frozen replay")
        raw_start = time.perf_counter()
        raw = predict(net, a["source"][:, 0, :])
        naive = np.repeat(a["source"][:, 0, -1:], 24, axis=1)
        raw_seconds = time.perf_counter() - raw_start
        rows, predictions = [], dict(raw=raw, naive=naive, future=a["future"], starts=a["starts"])
        with np.load(cp, allow_pickle=False) as z:
            valid = [r for r in cm["rows"] if r["status"] == "complete"]
            bulk = np.concatenate([z[r["cache_key"]] for r in valid]) if valid else np.empty((0,128), dtype=np.float32)
            pred_start = time.perf_counter()
            all_predictions = predict(net, bulk) if valid else np.empty((0,24), dtype=np.float32)
            prediction_seconds = time.perf_counter() - pred_start
            pi = 0
            for entry in cm["rows"]:
                row = dict(entry)
                if row["status"] == "complete":
                    pred = all_predictions[pi:pi+32]
                    pi += 32
                    predictions[row["cache_key"]] = pred
                    row["metrics"] = eval_metrics(a["future"], raw, pred, naive)
                rows.append(row)
                total[row["status"]] += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.with_suffix(".npz"), **predictions)
        write_json(output, dict(dataset=dataset, series_id=sid, seed=seed, rows=rows,
                   checkpoint_sha256=meta["checkpoint_sha256"], predictions_sha256=data.file_sha256(output.with_suffix(".npz")),
                   freeze_sha256=data.file_sha256(run / "freeze.json"), raw_inference_seconds=raw_seconds,
                   compressed_inference_seconds=prediction_seconds, evaluation_wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(evaluated=sid, seed=seed, counts=dict(total))), flush=True)
    return dict(total)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run):
    frozen = verify(run)
    flat, coverage, paired = [], Counter(), []
    fields = ("raw_mse", "mse", "excess_mse", "naive_mse", "prediction_mse")
    for path in sorted((run / "evaluation").glob("*/*/*.json")):
        record = read(path)
        rows = record["rows"]
        for r in rows:
            coverage[(record["seed"], r["status"])] += 1
            if r["status"] != "complete":
                continue
            m = r["metrics"]
            flat.append(dict(seed=record["seed"], dataset=record["dataset"], series_id=record["series_id"],
                             method=r["method"], point=r["point"]["label"], comparison=r["comparison"],
                             **{k:m[k] for k in fields}, raw_good_count=m["raw_good_count"],
                             usefulness_failure_count=m["usefulness_failure_count"], catastrophic_count=m["catastrophic_count"],
                             origins=m["origins"], artifact_bytes=r["artifact_bytes"]))
        for point in ("strict", "medium", "loose"):
            candidates = {r["method"]:r for r in rows if r["status"] == "complete" and r["comparison"] == "same_epsilon" and r["point"]["label"] == point}
            for left in ("qpet_pearson", "qpet_point", "qpet_acf"):
                if left not in candidates or "qpet_joint" not in candidates:
                    continue
                dual, triple = candidates[left]["metrics"], candidates["qpet_joint"]["metrics"]
                d, t, raw, naive = [np.asarray(v) for v in (dual["per_origin_mse"], triple["per_origin_mse"], dual["per_origin_raw_mse"], dual["per_origin_naive_mse"])]
                good = raw < naive
                rescue = good & (d >= naive) & (t < naive)
                harm = good & (t >= naive) & (d < naive)
                paired.append(dict(seed=record["seed"], dataset=record["dataset"], series_id=record["series_id"],
                                   point=point, left=left, right="qpet_joint", mse_left=float(d.mean()), mse_right=float(t.mean()),
                                   delta_right_minus_left=float((t-d).mean()), raw_good=int(good.sum()),
                                   rescued=int(rescue.sum()), new_failures=int(harm.sum()),
                                   rescued_indices=np.flatnonzero(rescue).tolist(), new_failure_indices=np.flatnonzero(harm).tolist()))
    group = defaultdict(list)
    for r in flat:
        group[tuple(r[k] for k in ("seed", "dataset", "method", "point", "comparison"))].append(r)
    per_source = []
    for key, rows in sorted(group.items()):
        out = dict(zip(("seed", "dataset", "method", "point", "comparison"), key))
        out.update(series=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in fields})
        for k in ("raw_good_count", "usefulness_failure_count", "catastrophic_count", "origins"):
            out[k] = sum(r[k] for r in rows)
        out["failure_rate_all_origins"] = out["usefulness_failure_count"] / out["origins"]
        out["failure_rate_raw_good"] = out["usefulness_failure_count"] / out["raw_good_count"] if out["raw_good_count"] else None
        per_source.append(out)
    macro_groups = defaultdict(list)
    for r in per_source:
        macro_groups[tuple(r[k] for k in ("seed", "method", "point", "comparison"))].append(r)
    macro = []
    for key, rows in sorted(macro_groups.items()):
        out = dict(zip(("seed", "method", "point", "comparison"), key))
        out.update(sources=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in fields})
        out["source_macro_failure_rate_all_origins"] = float(np.mean([r["failure_rate_all_origins"] for r in rows]))
        out["pooled_raw_good"] = sum(r["raw_good_count"] for r in rows)
        out["pooled_failures"] = sum(r["usefulness_failure_count"] for r in rows)
        macro.append(out)
    seed_groups = defaultdict(list)
    for r in macro:
        seed_groups[tuple(r[k] for k in ("method", "point", "comparison"))].append(r)
    multiseed = []
    for key, rows in sorted(seed_groups.items()):
        out = dict(zip(("method", "point", "comparison"), key))
        out["seeds"] = len(rows)
        for k in fields + ("source_macro_failure_rate_all_origins",):
            out[k + "_mean"] = float(np.mean([r[k] for r in rows]))
            out[k + "_sd"] = float(np.std([r[k] for r in rows], ddof=1)) if len(rows)>1 else None
        multiseed.append(out)
    write_csv(run / "summary" / "series.csv", flat)
    write_csv(run / "summary" / "per_source.csv", per_source)
    write_csv(run / "summary" / "macro_per_seed.csv", macro)
    write_csv(run / "summary" / "macro_three_seeds.csv", multiseed)
    write_json(run / "summary" / "paired_ablation.json", paired)
    counts = {str(seed):{status: n for (s,status),n in coverage.items() if s == seed} for seed in SEEDS}
    complete = all(counts[str(s)].get("complete", 0) == 9514 and sum(counts[str(s)].values()) == 9514 for s in SEEDS)
    status = dict(coverage=counts, expected_test_rows_per_seed=9514, unique_test_windows=4544,
                  calibration_excluded=8236, seeds=list(SEEDS), all_complete=complete,
                  models=len(list((run / "models").glob("*/*.json"))), updated_utc=now(),
                  freeze_sha256=data.file_sha256(run / "freeze.json"))
    write_json(run / "STATUS.json", status)
    if complete:
        write_json(run / "FINISHED.json", status)
    return status


def dispatch(run, command, dataset, seed):
    run.mkdir(parents=True, exist_ok=True)
    log = run / "logs" / f"{command}-{dataset}-{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    args = ["flock", LOCK, "env", "NCLAB_GPU_LOCK_HELD=1", sys.executable, "-u", "-m", __spec__.name,
            command, "--run", str(run), "--dataset", dataset, "--seed", str(seed)]
    start = time.perf_counter()
    with log.open("a", encoding="utf-8") as f:
        result = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    write_json(run / "jobs" / f"{command}-{dataset}-{seed}.json", dict(command=command, dataset=dataset, seed=seed,
               exit_code=result.returncode, queue_and_run_seconds=time.perf_counter()-start, finished_utc=now()))
    if result.returncode:
        print(log.read_text(encoding="utf-8")[-6000:], flush=True)
        raise RuntimeError(f"{command} failed: {dataset}/{seed}")


def smoke(run):
    runtime()
    import torch
    torch.manual_seed(0)
    net = model().to("cuda")
    x = torch.sin(torch.arange(128*96, device="cuda").reshape(128,96,1).float()/19)
    opt = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])
    start = time.perf_counter()
    for _ in range(12):
        loss = (net(x)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    a = predict(net, x.cpu().numpy()[:, :, 0])
    assert a.shape == (128,24)
    state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    clone = model().to("cuda")
    clone.load_state_dict(state)
    np.testing.assert_array_equal(a, predict(clone, x.cpu().numpy()[:, :, 0]))
    return dict(smoke="passed", seconds_12_steps=time.perf_counter()-start, parameters=sum(p.numel() for p in net.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "cache", "train", "evaluate", "summarize", "all", "smoke"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", choices=DATASETS, default="ETTm1")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == "all":
        freeze_run(run)
        cache(run)
        for seed in SEEDS:
            for dataset in DATASETS:
                dispatch(run, "train", dataset, seed)
                dispatch(run, "evaluate", dataset, seed)
                summarize(run)
        result = summarize(run)
    elif args.command in ("train", "evaluate", "smoke") and os.environ.get("NCLAB_GPU_LOCK_HELD") != "1":
        dispatch(run, args.command, args.dataset, args.seed)
        result = {"dispatched": args.command, "dataset": args.dataset, "seed": args.seed}
    elif args.command == "freeze": result = freeze_run(run)
    elif args.command == "cache": result = cache(run)
    elif args.command == "train": result = train(run, args.dataset, args.seed)
    elif args.command == "evaluate": result = evaluate(run, args.dataset, args.seed)
    elif args.command == "summarize": result = summarize(run)
    else: result = smoke(run)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
