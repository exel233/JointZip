"""Frozen NASA anomaly consumers; no fitting and no TEST-dependent selection.

Stages: freeze, smoke, score, metrics, audit.  Old float32 Ridge has no saved
weights and is intentionally excluded; the pre-existing SVD64 coefficients
are restored.  KNN reconstructs its fixed RAW TRAIN dictionary; LSTM restores
the 72 original checkpoints and original cuDNN settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
import torch

from . import common as c
from experiments.vldb.downstream_expansion.anomaly import run as old

OUT = c.RUN / "evaluation/anomaly"
SVD = c.NASA / "supplements/ridge_svd64"
CONSUMERS = [("ridge_svd64", 0), ("knn", 0)] + [("lstm", seed) for seed in c.SEEDS]
ATOL, RTOL = 1e-7, 1e-6
_CHECKED_BINDINGS = set()
PROTOCOL = {
    "schema": "nclab.paper-budget-guided.anomaly-frozen-consumers.v1",
    "strategy": c.STRATEGY,
    "consumers": [[model, seed] for model, seed in CONSUMERS],
    "ridge": "Restore existing float64 SVD coefficients; no fit/alpha selection. Historical float32 Ridge is excluded because its coefficients were not saved.",
    "knn": "Deterministic normalized length150 RAW optimization-TRAIN dictionary; exact nearest-neighbor AB-join; not official transductive Sub-KNN.",
    "lstm": "Official unchanged TSB-AD LSTMModel architecture, original 72 saved checkpoints, batch256 inference, context150/horizon1/hidden20/layers2.",
    "score_alignment": "Unchanged RAW prefix up to train_end; decoded TEST suffix trimmed to true length. Both residual current target and prior history are decoded. Labels unchanged.",
    "threshold": "Existing RAW-normal-VAL 99th percentile, strict score>threshold. Recompute and verify before any new compression score. No oracle TEST threshold/point adjustment.",
    "metrics": "sklearn AP/ROC-AUC/fixed-threshold F1, precision, recall; official TSB generate_curve(opt_mem,thre=250) VUS-PR/VUS-ROC; period fixed on RAW optimization TRAIN.",
    "backend": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                "cudnn_allow_tf32": True, "matmul_allow_tf32": False, "torch_threads": 1},
    "replay_tolerance": {"atol": ATOL, "rtol": RTOL,
                         "cache_reuse": "Requires BITWISE raw and fixed strict-Joint reference replay; numerical closeness alone does not authorize legacy score reuse."},
    "reuse": "Hash-bind old inputs/reconstructions/score and metric files. Reuse old metrics only when labels, threshold, scores and reconstruction identity agree. Exact current-run score duplicates share metrics.",
    "aggregation": "Equal-file macro after averaging LSTM seeds within each file; 24 NASA files, two parent domains, no independence claim for seeds.",
    "failure_policy": "Missing or failed compression yields explicit incomplete/failed rows; inference and metric exceptions retained, never converted to zero.",
    "parent_reuse": "Only unchanged baseline methods/raw may reuse the parent expansion. Bind parent rows, scores, compressed inputs, normalizers, consumer weights, protocol and runtime; require bitwise RAW and strict-Joint anchor replay, identical decoded true suffix, labels, threshold and VUS window. Simple-method identifiers are never mapped to SSBB identifiers for reuse. Metrics retain the original parent row binding.",
}


def bits(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def array_sha(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.dtype.str.encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def _path(path):
    return str(Path(path).relative_to(c.ROOT))


def _binding(path):
    path = Path(path)
    return {"path": _path(path), "sha256": c.sha(path)}


def _checked(binding):
    path = c.ROOT / binding["path"]
    stat = path.stat()
    signature = (str(path.resolve()), binding["sha256"], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature in _CHECKED_BINDINGS:
        return path
    if c.sha(path) != binding["sha256"]:
        raise RuntimeError("Bound anomaly artifact changed: " + str(path))
    _CHECKED_BINDINGS.add(signature)
    return path


def _base(model):
    return SVD if model == "ridge_svd64" else c.NASA


def _score_binding(base, sid, consumer, key):
    sp = base / "scores" / sid / consumer / (key + ".npz")
    mp = base / "metrics" / sid / consumer / (key + ".json")
    score = _binding(sp)
    metric = c.read(mp)
    if metric["score_sha256"] != score["sha256"]:
        raise RuntimeError("Legacy score/metric binding mismatch: " + str(sp))
    return {"scores": score, "metrics": _binding(mp)}


def _runtime_identity():
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}


def _parent_result_binding(entry, consumer, key, parent_protocol_sha):
    """Freeze the complete evidence chain, rather than relabeling a metric."""
    rp = c.PARENT_RUN / "evaluation/anomaly/rows" / entry["sid"] / consumer / (key + ".json")
    row = c.read(rp)
    if row["status"] != "complete" or row["protocol_sha256"] != parent_protocol_sha:
        raise RuntimeError("Parent anomaly result is incomplete or has a different protocol")
    if row["input_sha256"] != entry["input_sha256"] or row["consumer"] != consumer:
        raise RuntimeError("Parent anomaly consumer/source differs")
    sp = c.PARENT_RUN / row["scores"]
    if c.sha(sp) != row["scores_sha256"]:
        raise RuntimeError("Parent anomaly score hash mismatch")
    out = {"row": _binding(rp), "scores": _binding(sp)}
    if row.get("compression_row"):
        cp = c.PARENT_RUN / row["compression_row"]
        compressed = c.read(cp)
        if compressed["status"] != "complete" or compressed["sid"] != entry["sid"]:
            raise RuntimeError("Parent compression record unavailable")
        if compressed["method"] != row["method"] or compressed["point"]["label"] != row["point"]:
            raise RuntimeError("Parent score/compression method mismatch")
        rec = c.PARENT_RUN / compressed["reconstruction"]
        ap = c.PARENT_RUN / compressed["artifact"]
        if c.sha(rec) != compressed["reconstruction_sha256"] or c.sha(ap) != compressed["artifact_sha256"]:
            raise RuntimeError("Parent compression artifact changed")
        out.update(compression=_binding(cp), reconstruction=_binding(rec), artifact=_binding(ap))
    return out


def freeze():
    dest = OUT / "protocol.json"
    if dest.exists():
        return verify()
    source_paths = [Path(__file__), Path(c.__file__), Path(old.__file__),
                    c.ROOT / "experiments/vldb/downstream_expansion/anomaly/linear_stability.py",
                    old.UP / "TSB_AD/models/LSTMAD.py",
                    old.UP / "TSB_AD/evaluation/basic_metrics.py",
                    old.UP / "TSB_AD/utils/slidingWindows.py",
                    c.NASA / "protocol.json", SVD / "protocol.json"]
    parent_path = c.PARENT_RUN / "evaluation/anomaly/protocol.json"
    parent = c.read(parent_path)
    parent_protocol_sha = c.sha(parent_path)
    parent_freeze = c.read(c.PARENT_RUN / "freeze.json")
    if parent["parent_freeze_sha256"] != c.sha(c.PARENT_RUN / "freeze.json"):
        raise RuntimeError("Parent anomaly/input freeze binding changed")
    if parent["runtime"] != _runtime_identity():
        raise RuntimeError("Parent NASA runtime differs; score reuse requires an explicit new runtime audit")
    for field in ("consumers", "ridge", "knn", "lstm", "score_alignment", "threshold", "metrics", "backend"):
        if parent["protocol"][field] != PROTOCOL[field]:
            raise RuntimeError("Parent scientific consumer setting differs: " + field)
    for path, expected in parent["files"].items():
        if c.sha(c.ROOT / path) != expected:
            raise RuntimeError("Parent anomaly code binding changed: " + path)
    parent_by = {e["sid"]: e for e in parent["entries"]}
    parent_inputs = {e["sid"]: e for e in parent_freeze["entries"] if e["task"] == "anomaly"}
    bindings, records = {_path(p): c.sha(p) for p in source_paths}, []
    from TSB_AD.utils.slidingWindows import find_length_rank
    for entry in c.entries("anomaly"):
        data = c.arrays(entry)
        z, y = data["original"], data["labels"]
        tr, cut = int(data["train_end"]), int(data["fit_end"])
        legacy = old.inputs(entry["file"])
        if not (bits(z, legacy[0]) and bits(y, legacy[1]) and tr == legacy[2] and cut == legacy[3]
                and bits(data["source"], legacy[-1])):
            raise RuntimeError("NASA frozen source does not match old consumer data")
        record = {"sid": entry["sid"], "input_sha256": entry["input_sha256"],
                  "period": int(find_length_rank(z[:cut], rank=1)),
                  "labels_sha256": array_sha(y[tr:]), "consumers": {}, "reconstructions": {}}
        pe, pb = parent_inputs[entry["sid"]], parent_by[entry["sid"]]
        pp = c.PARENT_RUN / pe["input"]
        if pe["input_sha256"] != entry["input_sha256"] or c.sha(pp) != pe["input_sha256"]:
            raise RuntimeError("Parent NASA input bytes differ")
        if any(pb[k] != record[k] for k in ("sid", "input_sha256", "period", "labels_sha256")):
            raise RuntimeError("Parent NASA source/labels/VUS period differs")
        record["parent_input"] = _binding(pp)
        for point in old.POINTS:
            for method in old.METHODS:
                key = point["label"] + "__" + method
                row_path = old.row_path(entry["file"], point["label"], method)
                row = c.read(row_path)
                if row["status"] != "ok":
                    raise RuntimeError("Legacy compression incomplete: " + str(row_path))
                rp = row_path.with_suffix(".npy")
                if c.sha(rp) != row["reconstruction_sha256"]:
                    raise RuntimeError("Legacy reconstruction binding mismatch")
                record["reconstructions"][key] = {"row": _binding(row_path), "array": _binding(rp)}
        for model, seed in CONSUMERS:
            consumer = model + "_" + str(seed)
            base = _base(model)
            spec = {"model": model, "seed": seed, "legacy": {}}
            if model == "lstm":
                wp = c.NASA / "models" / entry["sid"] / (consumer + ".pt")
                mp = wp.with_suffix(".json")
                meta = c.read(mp)
                if meta["weights_sha256"] != c.sha(wp) or meta["seed"] != seed or meta["file"] != entry["file"]:
                    raise RuntimeError("LSTM checkpoint metadata mismatch")
                spec.update(weights=_binding(wp), model_metadata=_binding(mp))
            elif model == "ridge_svd64":
                wp = base / "scores" / entry["sid"] / consumer / "weights.npz"
                mp = wp.with_name("model.json")
                if c.read(mp)["weights_sha256"] != c.sha(wp):
                    raise RuntimeError("SVD64 checkpoint metadata mismatch")
                spec.update(weights=_binding(wp), model_metadata=_binding(mp))
            for key in ["raw"] + list(record["reconstructions"]):
                spec["legacy"][key] = _score_binding(base, entry["sid"], consumer, key)
            pspec = pb["consumers"][consumer]
            for field in ("model", "seed", "weights", "model_metadata"):
                if spec.get(field) != pspec.get(field):
                    raise RuntimeError("Parent consumer weights or model identity differs: " + field)
            keys = ["raw__raw"] + [m + "__" + p["label"] for m in c.METHODS
                    if m not in c.SIMPLE_METHODS for p in c.points(entry, m)]
            spec["parent_results"] = {key: _parent_result_binding(entry, consumer, key, parent_protocol_sha) for key in keys}
            spec["parent_anchors"] = {"raw": spec["parent_results"]["raw__raw"],
                "strict_joint_reference": _parent_result_binding(entry, consumer, "qpet_joint__strict", parent_protocol_sha)}
            record["consumers"][consumer] = spec
        records.append(record)
    result = {"protocol": PROTOCOL, "parent_freeze_sha256": c.sha(c.RUN / "freeze.json"),
              "files": bindings, "entries": records,
              "runtime": _runtime_identity(), "parent_protocol": _binding(parent_path),
              "parent_input_freeze": _binding(c.PARENT_RUN / "freeze.json")}
    c.write(dest, result, exclusive=True)
    print(json.dumps({"stage": "anomaly_freeze", "files": len(records),
                      "consumers_per_file": len(CONSUMERS), "checkpoint_count": 96,
                      "protocol_sha256": c.sha(dest)}), flush=True)
    return result


def verify():
    record = c.read(OUT / "protocol.json")
    if record["protocol"] != PROTOCOL or record["parent_freeze_sha256"] != c.sha(c.RUN / "freeze.json"):
        raise RuntimeError("Anomaly protocol differs")
    for path, digest in record["files"].items():
        _checked({"path": path, "sha256": digest})
    _checked(record["parent_protocol"])
    _checked(record["parent_input_freeze"])
    if record["runtime"] != _runtime_identity():
        raise RuntimeError("Frozen NASA runtime changed")
    return record


def _runtime():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def _load(spec, data):
    model = spec["model"]
    z, cut = data["original"], int(data["fit_end"])
    if model == "ridge_svd64":
        with np.load(_checked(spec["weights"]), allow_pickle=False) as weights:
            fitted = Ridge(alpha=float(weights["alpha"]), solver="svd")
            fitted.coef_ = weights["coef"].copy()
            fitted.intercept_ = weights["intercept"].copy()
            fitted.n_features_in_ = len(fitted.coef_)
        return lambda x: old.ridge_scores(np.asarray(x, dtype=np.float64), fitted)
    if model == "knn":
        dictionary = old.znorm(np.lib.stride_tricks.sliding_window_view(z[:cut], old.CONTEXT))
        return lambda x: old.knn_scores(x, dictionary)
    from TSB_AD.models.LSTMAD import LSTMModel
    device = torch.device("cuda")
    fitted = LSTMModel(old.CONTEXT, 1, 20, 1, 2, 128, device).to(device)
    fitted.load_state_dict(torch.load(_checked(spec["weights"]), map_location=device, weights_only=True))
    fitted.eval()
    return lambda x: old.deep_scores(x, fitted)


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _compare(scores, threshold, cached):
    equal = bits(scores, cached["scores"]) and bits(np.asarray(threshold), cached["threshold"])
    close = bool(np.allclose(scores, cached["scores"], atol=ATOL, rtol=RTOL)) and bool(np.isclose(threshold, cached["threshold"], atol=ATOL, rtol=RTOL))
    return {"bitwise_equal": equal, "within_tolerance": close,
            "maximum_absolute_score_difference": float(np.max(np.abs(scores - cached["scores"]))),
            "absolute_threshold_difference": float(abs(threshold - float(cached["threshold"])))}


def _replay(entry, bound, consumer, fn, data):
    z, labels = data["original"], data["labels"]
    tr, cut = int(data["train_end"]), int(data["fit_end"])
    spec = bound["consumers"][consumer]
    raw = fn(z)
    threshold = float(np.quantile(raw[cut:tr], .99))
    cached = _npz(_checked(spec["legacy"]["raw"]["scores"]))
    if not bits(labels[tr:], cached["labels"]):
        raise RuntimeError("Legacy TEST labels changed")
    comparison = {"raw": _compare(raw[tr:], threshold, cached)}
    key = "strict__qpet_joint"
    reference = np.load(_checked(bound["reconstructions"][key]["array"]), allow_pickle=False).ravel()[:len(z) - tr]
    scores = fn(np.concatenate([z[:tr], reference]))[tr:]
    cached_reference = _npz(_checked(spec["legacy"][key]["scores"]))
    if not bits(labels[tr:], cached_reference["labels"]):
        raise RuntimeError("Legacy reference labels changed")
    comparison["strict_joint_reference"] = _compare(scores, threshold, cached_reference)
    comparison["legacy_cache_reuse_allowed"] = all(v["bitwise_equal"] for v in comparison.values())
    comparison["passed"] = all(comparison[k]["within_tolerance"] for k in ("raw", "strict_joint_reference"))
    parent_checks = {}
    for label, values in (("raw", raw[tr:]), ("strict_joint_reference", scores)):
        bound_score = spec["parent_anchors"][label]
        _checked(bound_score["row"])
        cached_parent = _npz(_checked(bound_score["scores"]))
        if not bits(labels[tr:], cached_parent["labels"]):
            raise RuntimeError("Parent anomaly anchor labels changed")
        parent_checks[label] = _compare(values, threshold, cached_parent)
    comparison["parent_anchor_checks"] = parent_checks
    comparison["parent_cache_reuse_allowed"] = all(v["bitwise_equal"] for v in parent_checks.values())
    comparison.update(sid=entry["sid"], consumer=consumer, threshold=threshold,
                      atol=ATOL, rtol=RTOL, protocol_sha256=c.sha(OUT / "protocol.json"))
    c.write(OUT / "replay" / entry["sid"] / (consumer + ".json"), comparison)
    if not comparison["passed"]:
        raise RuntimeError("Fixed consumer replay exceeded frozen tolerance: " + str(comparison))
    return raw[tr:], threshold, comparison


def _parent_candidate(spec, bound, key, rec, labels, threshold, replay):
    """An exact input match authorizes score reuse, never method relabeling."""
    if not replay.get("parent_cache_reuse_allowed") or key not in spec["parent_results"]:
        return None
    binding = spec["parent_results"][key]
    row = c.read(_checked(binding["row"]))
    _checked(bound["parent_input"])
    if row["decoded_true_suffix_sha256"] != array_sha(rec):
        return None
    if row["labels_sha256"] != array_sha(labels) or row["vus_window"] != bound["period"]:
        raise RuntimeError("Parent anomaly label/period metadata mismatch")
    if not bits(np.asarray(row["threshold"]), np.asarray(threshold)):
        raise RuntimeError("Parent anomaly threshold changed")
    if binding.get("compression"):
        _checked(binding["compression"])
        _checked(binding["artifact"])
        parent_rec = np.load(_checked(binding["reconstruction"]), allow_pickle=False).ravel()[:len(rec)]
        if not bits(rec, parent_rec):
            return None
    original = _checked(binding["scores"])
    cached = _npz(original)
    if not bits(cached["labels"], labels) or not bits(cached["threshold"], np.asarray(threshold)):
        raise RuntimeError("Parent anomaly score alignment changed")
    if array_sha(cached["scores"]) != row["score_array_sha256"] or row["scores_sha256"] != binding["scores"]["sha256"]:
        raise RuntimeError("Parent anomaly score array identity mismatch")
    return binding


def _consumer_entry(task):
    sid, consumer, smoke = task
    protocol = verify()
    entry = next(e for e in c.entries("anomaly") if e["sid"] == sid)
    bound = next(e for e in protocol["entries"] if e["sid"] == sid)
    data = c.arrays(entry)
    spec = bound["consumers"][consumer]
    _runtime()
    if spec["model"] == "lstm":
        with c.gpu_lock():
            return _consumer_entry_unlocked(entry, bound, consumer, data, spec, smoke)
    return _consumer_entry_unlocked(entry, bound, consumer, data, spec, smoke)


def _consumer_entry_unlocked(entry, bound, consumer, data, spec, smoke):
    fn = _load(spec, data)
    raw_scores, threshold, replay = _replay(entry, bound, consumer, fn, data)
    if smoke:
        return {"sid": entry["sid"], "consumer": consumer, "status": "smoke_passed", "replay": replay}
    z, labels = data["original"], data["labels"]
    parent_protocol_binding = c.read(OUT / "protocol.json")["parent_protocol"]
    tr, cut = int(data["train_end"]), int(data["fit_end"])
    test_labels = labels[tr:]
    legacy_by_array = {}
    if replay["legacy_cache_reuse_allowed"]:
        for key, bindings in bound["reconstructions"].items():
            rec = np.load(_checked(bindings["array"]), allow_pickle=False).ravel()[:len(z) - tr]
            legacy_by_array.setdefault(array_sha(rec), key)
        legacy_by_array[array_sha(z[tr:])] = "raw"
    current_cache = {}
    folder = OUT / "scores" / entry["sid"] / consumer
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"method": "raw", "point": {"label": "raw"}, "status": "complete"}]
    rows.extend(c.rows(entry, "test", require_complete=False))
    statuses = Counter()
    for row in rows:
        point = row["point"]["label"] if isinstance(row["point"], dict) else row["point"]
        method = row["method"]
        key = method + "__" + point
        rp = OUT / "rows" / entry["sid"] / consumer / (key + ".json")
        if rp.exists() and c.read(rp)["status"] in ("scored", "complete"):
            statuses["existing"] += 1
            continue
        result = {"task": "anomaly", "sid": entry["sid"], "dataset": entry["dataset"],
                  "consumer": consumer, "model": spec["model"], "seed": spec["seed"],
                  "method": method, "point": point,
                  "input_sha256": entry["input_sha256"], "labels_sha256": bound["labels_sha256"],
                  "protocol_sha256": c.sha(OUT / "protocol.json"), "vus_window": bound["period"],
                  "test_points": len(test_labels), "threshold": threshold,
                  "compression_row": row.get("row_path"),
                  "compression_artifact_bytes": row.get("artifact_bytes", row.get("bytes")),
                  "compression_ratio": row.get("cr", row.get("compression_ratio"))}
        try:
            if row["status"] != "complete":
                result.update(status="upstream_failed", upstream_status=row["status"], error=row.get("error"))
            else:
                rec = z[tr:] if method == "raw" else c.decoded(row).ravel()[:len(z) - tr]
                if rec.shape != z[tr:].shape or not np.isfinite(rec).all():
                    raise ValueError("Decoded NASA suffix shape/finiteness mismatch")
                digest = array_sha(rec)
                sp = folder / (key + ".npz")
                legacy_key = legacy_by_array.get(digest)
                parent_match = _parent_candidate(spec, bound, key, rec, test_labels, threshold, replay)
                if parent_match is not None:
                    shutil.copyfile(_checked(parent_match["scores"]), sp)
                    result["parent_result_binding"] = parent_match
                    result["parent_protocol_binding"] = parent_protocol_binding
                    result["parent_input_binding"] = bound["parent_input"]
                    result["parent_replay_binding"] = _binding(OUT / "replay" / entry["sid"] / (consumer + ".json"))
                    result["score_origin"] = "parent_expansion_identical_baseline_input"
                elif legacy_key is not None:
                    old_binding = spec["legacy"][legacy_key]
                    original = _checked(old_binding["scores"])
                    cached = _npz(original)
                    if not bits(cached["labels"], test_labels) or not bits(cached["threshold"], np.asarray(threshold)):
                        raise RuntimeError("Legacy score alignment changed")
                    shutil.copyfile(original, sp)
                    result["legacy_metric_binding"] = old_binding["metrics"]
                    result["score_origin"] = "legacy_bitwise_input_match:" + legacy_key
                elif digest in current_cache:
                    previous = current_cache[digest]
                    shutil.copyfile(previous, sp)
                    result["score_origin"] = "current_run_identical_decoded_input"
                else:
                    scores = raw_scores if method == "raw" else fn(np.concatenate([z[:tr], rec]))[tr:]
                    if scores.shape != test_labels.shape or not np.isfinite(scores).all():
                        raise ValueError("Nonfinite or misaligned anomaly scores")
                    np.savez_compressed(sp, scores=scores, labels=test_labels, threshold=threshold)
                    result["score_origin"] = "fresh_fixed_consumer_inference"
                saved = _npz(sp)
                if method in c.LOSSLESS and not bits(saved["scores"], raw_scores):
                    raise RuntimeError("Lossless input must reproduce RAW anomaly score bit-for-bit")
                current_cache[digest] = sp
                result.update(status="scored", decoded_true_suffix_sha256=digest,
                              scores=str(sp.relative_to(c.RUN)), scores_sha256=c.sha(sp),
                              score_array_sha256=array_sha(saved["scores"]))
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        c.write(rp, result)
        statuses[result["status"]] += 1
    expected = 1 + sum(len(c.points(entry, m)) for m in c.METHODS)
    record = {"sid": entry["sid"], "consumer": consumer, "expected_rows": expected,
              "available_rows": len(rows), "counts_this_pass": dict(statuses),
              "all_upstream_present": len(rows) == expected}
    c.write(OUT / "coverage" / entry["sid"] / (consumer + ".json"), record)
    del fn
    if spec["model"] == "lstm":
        torch.cuda.empty_cache()
    return record


def _metrics_for(data, window):
    from TSB_AD.evaluation.basic_metrics import generate_curve
    scores, labels = data["scores"], data["labels"]
    pred = scores > float(data["threshold"])
    metrics = dict(AP=float(average_precision_score(labels, scores)),
                   ROC_AUC=float(roc_auc_score(labels, scores)),
                   f1_at_val99=float(f1_score(labels, pred, zero_division=0)),
                   precision_at_val99=float(precision_score(labels, pred, zero_division=0)),
                   recall_at_val99=float(recall_score(labels, pred, zero_division=0)))
    curve = generate_curve(labels.astype(int), scores, window, version="opt_mem", thre=250)
    metrics.update(VUS_ROC=float(curve[-2]), VUS_PR=float(curve[-1]))
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("Official anomaly metric returned nonfinite values")
    return metrics


def _parent_metrics(row, data):
    """Check every official metric input before inheriting a parent's metric."""
    binding = row["parent_result_binding"]
    parent = c.read(_checked(binding["row"]))
    _checked(row["parent_protocol_binding"])
    _checked(row["parent_input_binding"])
    replay = c.read(_checked(row["parent_replay_binding"]))
    if not replay["parent_cache_reuse_allowed"]:
        raise RuntimeError("Parent metric reuse lacks a bitwise consumer replay")
    parent_data = _npz(_checked(binding["scores"]))
    for name in ("scores", "labels", "threshold"):
        if not bits(data[name], parent_data[name]):
            raise RuntimeError("Parent metric array differs: " + name)
    for name in ("vus_window", "threshold", "labels_sha256", "score_array_sha256", "decoded_true_suffix_sha256", "model", "seed", "consumer", "sid", "input_sha256", "method", "point"):
        if parent[name] != row[name]:
            raise RuntimeError("Parent metric metadata differs: " + name)
    if parent["scores_sha256"] != row["scores_sha256"] or parent["status"] != "complete":
        raise RuntimeError("Parent metric score file differs or result incomplete")
    if array_sha(data["scores"]) != row["score_array_sha256"] or array_sha(data["labels"]) != row["labels_sha256"]:
        raise RuntimeError("Metric score/label digest mismatch")
    if not bits(data["threshold"], np.asarray(row["threshold"])):
        raise RuntimeError("Metric threshold digest mismatch")
    metrics = parent["metrics"]
    if not metrics or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("Parent metrics are missing or nonfinite")
    return metrics


def metric_one(path):
    path = Path(path)
    row = c.read(path)
    if row["status"] != "scored":
        return row["status"]
    started = time.perf_counter()
    try:
        sp = c.RUN / row["scores"]
        if c.sha(sp) != row["scores_sha256"]:
            raise RuntimeError("Anomaly score changed before metrics")
        data = _npz(sp)
        if row.get("parent_result_binding"):
            metrics, origin = _parent_metrics(row, data), "parent_hash_bound_identical_metric_inputs"
        elif row.get("legacy_metric_binding"):
            metric = c.read(_checked(row["legacy_metric_binding"]))
            if metric["score_sha256"] != row["scores_sha256"] or metric["vus_window"] != row["vus_window"] or metric["frozen_score_threshold"] != row["threshold"]:
                raise RuntimeError("Legacy metric does not bind identical scores/window/threshold")
            metrics, origin = metric["metrics"], "legacy_hash_bound_identical_scores"
        else:
            # Metric cache is keyed by every input to the official metric.
            key = hashlib.sha256(c.canonical(dict(score=row["score_array_sha256"],
                labels=row["labels_sha256"], window=row["vus_window"], threshold=row["threshold"],
                protocol=row["protocol_sha256"]))).hexdigest()
            cache = OUT / "metric_cache" / (key + ".json")
            if cache.exists():
                metrics, origin = c.read(cache)["metrics"], "current_run_identical_metric_inputs"
            else:
                metrics, origin = _metrics_for(data, row["vus_window"]), "fresh_official_TSB_metrics"
                c.write(cache, {"key": key, "metrics": metrics}, exclusive=True)
        row.update(status="complete", metrics=metrics, metric_origin=origin,
                   metric_seconds=time.perf_counter() - started)
    except Exception as exc:
        row.update(status="metric_failed", error=repr(exc), traceback=traceback.format_exc())
    c.write(path, row)
    return row["status"]


def audit_results():
    verify()
    all_rows, missing = [], []
    for entry in c.entries("anomaly"):
        for model, seed in CONSUMERS:
            consumer = model + "_" + str(seed)
            keys = [("raw", "raw")] + [(m, p["label"]) for m in c.METHODS for p in c.points(entry, m)]
            for method, point in keys:
                rp = OUT / "rows" / entry["sid"] / consumer / (method + "__" + point + ".json")
                if not rp.exists():
                    missing.append(str(rp.relative_to(c.RUN)))
                    continue
                row = c.read(rp)
                if row["status"] == "complete" and c.sha(c.RUN / row["scores"]) != row["scores_sha256"]:
                    raise RuntimeError("Final anomaly score hash mismatch")
                if row["status"] == "complete" and row.get("parent_result_binding"):
                    if _parent_metrics(row, _npz(c.RUN / row["scores"])) != row["metrics"]:
                        raise RuntimeError("Parent-derived final metrics changed")
                all_rows.append(row)
    counts = dict(Counter(row["status"] for row in all_rows))
    expected = len(c.entries("anomaly")) * len(CONSUMERS) * (1 + sum(len(c.points(c.entries("anomaly")[0], m)) for m in c.METHODS))
    result = {"expected": expected, "actual": len(all_rows), "missing": missing,
              "counts": counts, "complete": not missing and counts.get("complete", 0) == expected,
              "protocol_sha256": c.sha(OUT / "protocol.json")}
    c.write(OUT / "AUDIT.json", result)
    # One JSON array is convenient for the root report generator; per-file rows remain authoritative.
    c.write(OUT / "results.json", all_rows)
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "smoke", "score", "metrics", "audit"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", choices=["all", "linear", "lstm"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    _runtime()
    if args.stage == "freeze":
        freeze(); return
    verify()
    if args.stage == "audit":
        audit_results(); return
    if args.stage == "metrics":
        jobs = sorted((OUT / "rows").glob("*/*/*.json"))
        jobs = [p for p in jobs if c.read(p)["status"] == "scored"]
        fn = metric_one
    else:
        models = [(m, s) for m, s in CONSUMERS if args.model == "all" or (m == "lstm") == (args.model == "lstm")]
        entries = c.entries("anomaly")[:1] if args.stage == "smoke" else c.entries("anomaly")
        jobs = [(e["sid"], m + "_" + str(s), args.stage == "smoke") for e in entries for m, s in models]
        fn = _consumer_entry
    if args.limit:
        jobs = jobs[:args.limit]
    failures = 0
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fn, job): job for job in jobs}
        for index, future in enumerate(futures.as_completed(pending), 1):
            job = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                result = {"status": "failed", "stage": args.stage, "job": str(job),
                          "error": repr(exc), "traceback": traceback.format_exc()}
                c.write(OUT / "failures" / (hashlib.sha256(str(job).encode()).hexdigest() + ".json"), result)
            if args.stage != "metrics" or index % 50 == 0 or index == len(jobs):
                print(json.dumps({"stage": args.stage, "completed": index, "total": len(jobs), "result": result}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
