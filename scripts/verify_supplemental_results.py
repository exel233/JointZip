"""Verify the public Ford ACF and complete-file lossless result archives.

Uses only Python's standard library and the files in this repository. It checks
recorded values, rosters, byte accounting, and numerical aggregation. It does not
decode streams, recompute signal statistics, run classifiers, regenerate paired
bootstrap samples, or rerun any scientific experiment. Saved independent audits
are identified separately as historical evidence.

By default write generated/supplemental_results.json. --check-only writes no
files and prints the same deterministic verification report to stdout.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
ACF = ROOT / "results/acf_case/vldb_acf_forda_fordb_20261028_v1"
LOSSLESS = ROOT / "results/lossless_comparison/vldb_uniform_lossless_accounting_20261031_v1"
BACKENDS = ("sz3", "zfp", "sperr", "serf_xor", "serf_qt", "sz2016")
ACF_BACKENDS = ("sz3", "zfp")
ARMS = ("ER", "ERA_tau005", "ERA_tau001")
CONSUMERS = ("catch22", "spectrum32", "first_minimum")
DATASETS = {"FordA": 1320, "FordB": 810}
NUMERICAL_SLACK = 64 * sys.float_info.epsilon


class Verifier:
    def __init__(self):
        self.checks = 0
        self.numeric_comparisons = 0
        self.max_absolute_difference = 0.0
        self.inputs = {}

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError(label)

    def close(self, actual, expected, label):
        actual, expected = float(actual), float(expected)
        self.numeric_comparisons += 1
        difference = abs(actual - expected)
        self.max_absolute_difference = max(self.max_absolute_difference, difference)
        self.require(math.isfinite(actual) and math.isfinite(expected)
                     and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                     f"{label}: {actual!r} != {expected!r}")

    def read(self, path):
        data = path.read_bytes()
        self.inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(self.read(path).splitlines()))

    def keyed(self, rows, fields, label):
        keyed = {tuple(row[field] for field in fields): row for row in rows}
        self.require(len(keyed) == len(rows), f"{label}: duplicate keys")
        return keyed

    def boolean(self, value, label):
        self.require(value in ("True", "False"), f"{label}: invalid boolean")
        return value == "True"


def geometric_mean(values):
    values = list(values)
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("Geometric mean requires nonempty positive finite data")
    return math.exp(statistics.mean(math.log(v) for v in values))


def source_balanced_mean(rows, field):
    sources = defaultdict(list)
    for row in rows:
        sources[row["dataset"]].append(float(row[field]))
    return geometric_mean(geometric_mean(values) for values in sources.values())


def check_result_json(v, result, csv_tables):
    key_fields = {
        "downstream": ("dataset", "backend", "arm", "consumer"),
        "mechanism": ("dataset", "backend", "arm"),
        "contrasts": ("dataset", "backend", "consumer", "before", "after"),
    }
    for table, csv_rows in csv_tables.items():
        left = v.keyed(csv_rows, key_fields[table], f"{table} CSV")
        right = v.keyed(result[table], key_fields[table], f"{table} JSON")
        v.require(left.keys() == right.keys(), f"{table}: JSON/CSV roster mismatch")
        for key, csv_row in left.items():
            json_row = right[key]
            v.require(csv_row.keys() == json_row.keys(), f"{table}: fields differ")
            for field, expected in json_row.items():
                actual = csv_row[field]
                if isinstance(expected, (int, float)):
                    v.require(float(actual) == expected,
                              f"{table}/{key}/{field}: JSON/CSV numeric mismatch")
                else:
                    v.require(actual == expected, f"{table}/{key}/{field}: JSON/CSV mismatch")


def verify_acf(v):
    protocol = v.json(ACF / "PROTOCOL.public.json")
    binding = v.json(ACF / "TEST_BINDING.json")
    completion = v.json(ACF / "COMPLETE.json")
    historical = v.json(ACF / "audit/FINAL_RESULT_AUDIT.json")
    tables = {name: v.csv(ACF / (name + ".csv"))
              for name in ("downstream", "contrasts", "mechanism")}
    v.require([len(tables[name]) for name in ("downstream", "contrasts", "mechanism")]
              == [42, 24, 12], "Ford table coverage")
    check_result_json(v, v.json(ACF / "RESULTS.json"), tables)
    v.require(protocol["epsilon"] == 1.0 and protocol["rho"] == 0.99
              and protocol["max_lag"] == 16
              and protocol["arms"] == {"ER": -1.0, "ERA_tau005": 0.05, "ERA_tau001": 0.01},
              "Ford frozen constraints")
    v.require(completion["complete"] is True
              and [completion[k] for k in ("windows", "streams", "consumer_metrics")]
              == [12780, 12, 42], "Ford saved completion counts")
    v.require({k: r["count"] for k, r in binding.items()} == DATASETS, "Ford TEST binding counts")

    downstream = v.keyed(tables["downstream"], ("dataset", "backend", "arm", "consumer"), "Ford downstream")
    mechanism = v.keyed(tables["mechanism"], ("dataset", "backend", "arm"), "Ford mechanism")
    expected_mechanism = set(itertools.product(DATASETS, ACF_BACKENDS, ARMS))
    v.require(set(mechanism) == expected_mechanism, "Ford mechanism roster")
    expected_downstream = {(*key, consumer) for key in expected_mechanism for consumer in CONSUMERS}
    expected_downstream.update((dataset, "Raw", "Raw", consumer)
                               for dataset in DATASETS for consumer in CONSUMERS)
    v.require(set(downstream) == expected_downstream, "Ford downstream roster")

    canonical_pairing = {}
    all_windows = 0
    for (dataset, backend, arm), record in mechanism.items():
        count = DATASETS[dataset]
        window_rows = v.csv(ACF / "artifacts" / f"{dataset}_{backend}_{arm}_windows.csv")
        v.require(len(window_rows) == count and int(record["n"]) == count, "Ford window count")
        all_windows += count
        v.require([int(r["index"]) for r in window_rows] == list(range(count)), "Ford ordered record index")
        pairing = [(int(r["label"]), int(r["first_min_raw"])) for r in window_rows]
        v.require(dict(Counter(str(label) for label, _ in pairing)) == binding[dataset]["labels"],
                  "Ford label class counts")
        if dataset not in canonical_pairing:
            canonical_pairing[dataset] = pairing
        else:
            v.require(pairing == canonical_pairing[dataset], "Ford arms share labels/raw diagnostic")
        errors = [float(r["max_abs_error"]) for r in window_rows]
        correlations = [float(r["pearson"]) for r in window_rows]
        acf_errors = [float(r["max_acf_error"]) for r in window_rows]
        traditional_errors = [float(r["traditional_ACF_MAE"]) for r in window_rows]
        min_errors = [abs(int(r["first_min_decoded"]) - int(r["first_min_raw"])) for r in window_rows]
        tau = protocol["arms"][arm]
        strict_count = numerical_count = 0
        for row, error, correlation, acf_error in zip(window_rows, errors, correlations, acf_errors):
            v.require(all(math.isfinite(x) for x in (error, correlation, acf_error)), "Ford finite diagnostics")
            strict = error <= 1.0 and correlation >= 0.99 and (tau < 0 or acf_error <= tau)
            numerical = (error <= 1.0 + NUMERICAL_SLACK
                         and correlation + NUMERICAL_SLACK >= 0.99
                         and (tau < 0 or acf_error <= tau + NUMERICAL_SLACK))
            v.require(v.boolean(row["strict_pass"], "Ford strict flag") == strict, "Ford strict gate from saved metrics")
            v.require(v.boolean(row["numerical_pass"], "Ford numerical flag") == numerical,
                      "Ford numerical gate from saved metrics")
            v.require(numerical, "Ford enabled constraint violation")
            strict_count += strict
            numerical_count += numerical
        for field, value in {
            "max_E": max(errors), "min_R": min(correlations), "max_slice_ACF": max(acf_errors),
            "traditional_ACF_MAE": statistics.mean(traditional_errors),
            "first_minimum_MAE": statistics.mean(min_errors),
            "first_minimum_changed": sum(x != 0 for x in min_errors),
            "enabled_strict_pass": strict_count, "enabled_numerical_pass": numerical_count,
            "slice_ACF_pass005": sum(x <= 0.05 + NUMERICAL_SLACK for x in acf_errors),
            "slice_ACF_pass001": sum(x <= 0.01 + NUMERICAL_SLACK for x in acf_errors),
            "CR": count * 500 * 4 / int(record["full_bytes"]),
        }.items():
            v.close(record[field], value, f"Ford mechanism/{dataset}/{backend}/{arm}/{field}")
        for field in ("lossless_windows", "repair_values", "correction_payload_bytes"):
            v.require(int(record[field]) >= 0, f"Ford nonnegative {field}")
    v.require(all_windows == 12780, "Ford total windows")

    for (dataset, backend, arm, consumer), row in downstream.items():
        count = DATASETS[dataset]
        v.require(int(row["n"]) == count, "Ford downstream TEST count")
        v.require(all(0 <= float(row[k]) <= 1 for k in ("accuracy", "balanced_accuracy", "AUROC")),
                  "Ford metric ranges")
        v.require(0 <= int(row["flip_vs_raw"]) <= count, "Ford prediction flip range")
        v.close(float(row["accuracy"]) * count, round(float(row["accuracy"]) * count), "Ford accuracy correct-count integrality")
        if backend == "Raw":
            v.require(int(row["full_bytes"]) == count * 500 * 4, "Ford raw bytes")
            v.close(row["CR"], 1, "Ford raw CR")
        else:
            mech = mechanism[dataset, backend, arm]
            v.require(row["full_bytes"] == mech["full_bytes"], "Ford consumers share full bytes")
            v.close(row["CR"], mech["CR"], "Ford consumers share complete CR")

    contrasts = v.keyed(tables["contrasts"], ("dataset", "backend", "consumer", "before", "after"), "Ford contrasts")
    expected_contrasts = {(d, b, c, "ER", a) for d, b, c, a in
                          itertools.product(DATASETS, ACF_BACKENDS, CONSUMERS, ARMS[1:])}
    v.require(set(contrasts) == expected_contrasts, "Ford contrast roster")
    signs = Counter()
    for (dataset, backend, consumer, before, after), row in contrasts.items():
        prior = downstream[dataset, backend, before, consumer]
        later = downstream[dataset, backend, after, consumer]
        for column, metric, scale in (("accuracy_change_pp", "accuracy", 100),
                                      ("balanced_accuracy_change_pp", "balanced_accuracy", 100),
                                      ("AUROC_change", "AUROC", 1)):
            v.close(row[column], scale * (float(later[metric]) - float(prior[metric])), f"Ford contrast {column}")
        outcomes = [int(row[k]) for k in ("label_improved", "label_tied", "label_declined")]
        v.require(min(outcomes) >= 0 and sum(outcomes) == DATASETS[dataset], "Ford paired outcome counts")
        v.close(row["accuracy_change_pp"], 100 * (outcomes[0] - outcomes[2]) / DATASETS[dataset],
                "Ford paired outcomes agree with accuracy delta")
        lo, hi = (float(row[k]) for k in ("descriptive_BA_CI95_low_pp", "descriptive_BA_CI95_high_pp"))
        v.require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, "Ford interval ordering")
        if consumer != "first_minimum":
            change = float(row["balanced_accuracy_change_pp"])
            signs["improved" if change > 0 else "declined" if change < 0 else "tied"] += 1
    v.require(dict(signs) == {"improved": 8, "tied": 8}, "Ford all declared classification contrasts retained")
    return {
        "downstream_rows": 42, "contrast_rows": 24, "mechanism_rows": 12,
        "window_rows": all_windows, "stream_groups": 12,
        "all_enabled_gates_pass_from_recorded_window_metrics": True,
        "classification_contrast_signs": {k: signs[k] for k in ("improved", "tied", "declined")},
        "paired_intervals": "Copied JSON/CSV agreement and endpoint order checked; bootstrap not regenerated.",
        "not_recomputed_from_signals": ["E/R/ACF per-window diagnostics", "traditional_ACF_max",
                                        "correction payload bytes", "lossless-window identity", "classifier predictions"],
    }, {
        "file": (ACF / "audit/FINAL_RESULT_AUDIT.json").relative_to(ROOT).as_posix(),
        "recorded_passed": historical["passed"],
        "recorded_comparison_count": historical["comparison_count"],
        "recorded_max_absolute_difference": historical["max_absolute_difference"],
        "recorded_scope": historical["scope"],
    }


def verify_lossless(v):
    rows = v.csv(LOSSLESS / "conditions.csv")
    condition_keys = ("task", "sid", "backend", "point")
    condition_index = v.keyed(rows, condition_keys, "Lossless conditions")
    v.require(len(rows) == 3024, "Lossless condition count")
    baseline_list = v.json(LOSSLESS / "lossless_files.json")
    baselines = v.keyed(baseline_list, ("task", "sid"), "Lossless inputs")
    v.require(len(baselines) == 172, "Lossless unique input count")
    expected_keys = {(task, sid, backend, point) for task, sid in baselines for backend in BACKENDS
                     for point in (("stress",) if task == "ucr" else ("strict", "medium", "loose"))}
    v.require(set(condition_index) == expected_keys, "Lossless complete backend/setting roster")
    v.require(dict(Counter(task for task, _ in baselines)) == {"forecast": 142, "ucr": 6, "anomaly": 24},
              "Lossless input task coverage")
    for row in rows:
        baseline = baselines[row["task"], row["sid"]]
        raw = int(row["raw_bytes"])
        joint = int(row["jointzip_complete_bytes"])
        loss = int(row["lossless_complete_bytes"])
        common = int(row["common_metadata_bytes"])
        v.require(raw > 0 and raw % 4 == 0 and joint > common > 0 and loss > common,
                  "Lossless positive complete byte accounting")
        v.require(raw == baseline["raw_bytes"] and loss == baseline["complete_bytes"]
                  and common == baseline["common_metadata_bytes"] and row["dataset"] == baseline["dataset"],
                  "Lossless repeated baseline identity")
        v.require(common + int(row["jointzip_payload_bytes"]) == joint, "JointZip full bytes include common metadata")
        v.require(common + baseline["codec_dispatch_bytes"] + baseline["complete_collection_bytes"] == loss,
                  "Lossless full bytes include common metadata and dispatch")
        for field, expected in {
            "jointzip_cr": raw / joint, "lossless_cr": raw / loss,
            "cr_ratio": loss / joint, "byte_savings_percent": 100 * (1 - joint / loss),
        }.items():
            v.close(row[field], expected, f"Lossless condition/{field}")
        outcome = "higher" if joint < loss else "lower" if joint > loss else "tied"
        v.require(row["outcome"] == outcome, "Lossless per-condition outcome")
        v.require(row["lossless_artifact"] == baseline["artifact"], "Lossless artifact reference")
    for baseline in baseline_list:
        v.close(baseline["cr"], baseline["raw_bytes"] / baseline["complete_bytes"], "Lossless file-list CR")

    tables = {}
    for name, keys, expected_count in (
        ("macros", ("backend", "task"), 18),
        ("settings", ("backend", "task", "point"), 42),
        ("sources", ("backend", "task", "dataset"), 84),
    ):
        table_rows = v.csv(LOSSLESS / f"{name}.csv")
        table = v.keyed(table_rows, keys, f"Lossless {name}")
        v.require(len(table) == expected_count, f"Lossless {name} count")
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        v.require(table.keys() == groups.keys(), f"Lossless {name} group roster")
        for key, group in groups.items():
            stored = table[key]
            v.require(int(stored["conditions"]) == len(group), f"Lossless {name} grouped counts")
            outcomes = Counter(row["outcome"] for row in group)
            for outcome in ("higher", "tied", "lower"):
                v.require(int(stored[outcome]) == outcomes[outcome], f"Lossless {name} outcome counts")
            if name == "sources":
                joint = geometric_mean(float(r["jointzip_cr"]) for r in group)
                loss = geometric_mean(float(r["lossless_cr"]) for r in group)
            else:
                joint = source_balanced_mean(group, "jointzip_cr")
                loss = source_balanced_mean(group, "lossless_cr")
            for field, expected in {"jointzip_cr": joint, "lossless_cr": loss, "cr_ratio": joint / loss}.items():
                v.close(stored[field], expected, f"Lossless {name}/{key}/{field}")
            if name == "macros":
                v.require(int(stored["sources"]) == len({r["dataset"] for r in group}), "Lossless macro source counts")
                v.close(stored["geomean_byte_savings_percent"], 100 * (1 - loss / joint), "Lossless macro savings")
        tables[name] = table_rows

    counts = Counter(row["outcome"] for row in rows)
    v.require({k: counts[k] for k in ("higher", "tied", "lower")}
              == {"higher": 2500, "tied": 0, "lower": 524}, "Lossless retained outcomes")
    summary = v.json(LOSSLESS / "SUMMARY.json")
    completion = v.json(LOSSLESS / "COMPLETE.json")
    for field, expected in {"conditions": 3024, "unique_inputs": 172,
                            "generated_complete_files": 3196, "macro_groups": 18,
                            "source_groups": 84, "macro_higher": 18,
                            "source_macro_higher": 83}.items():
        v.require(summary[field] == expected and completion[field] == expected, f"Lossless saved counts/{field}")
    v.require(summary["complete"] is True and completion["complete"] is True, "Lossless saved completion")
    v.require(summary["errors"] == completion["errors"] == [], "Lossless saved errors empty")
    for field in summary:
        v.require(summary[field] == completion[field], f"Lossless summary/completion agreement/{field}")
    for outcome in ("higher", "tied", "lower"):
        v.require(summary["counts"].get(outcome, 0) == counts[outcome], "Lossless summary outcome counts")
    for name, field in (("macro_cr_ratio_range", "cr_ratio"),
                        ("macro_geomean_byte_savings_range", "geomean_byte_savings_percent")):
        values = [float(row[field]) for row in tables["macros"]]
        for stored, expected in zip(summary[name], (min(values), max(values))):
            v.close(stored, expected, f"Lossless summary/{name}")
    v.require(sum(float(r["cr_ratio"]) > 1 for r in tables["macros"]) == 18, "Lossless macro higher count")
    v.require(sum(float(r["cr_ratio"]) > 1 for r in tables["sources"]) == 83, "Lossless source higher count")
    historical = v.json(LOSSLESS / "INDEPENDENT_REVIEW.json")
    roster_audit = v.json(LOSSLESS / "ROSTER_IDENTITY_REVIEW.public.json")
    return {
        "condition_rows": 3024, "macro_rows": 18, "setting_rows": 42, "source_rows": 84,
        "unique_inputs": 172,
        "condition_outcomes": {k: counts[k] for k in ("higher", "tied", "lower")},
        "byte_to_cr_and_ratio_checked": True,
        "complete_accounting_from_recorded_sizes_checked": True,
        "all_aggregate_groups_recomputed_from_public_condition_rows": True,
        "aggregation": "Geometric means within each source, then equal-source geometric means; each source table groups all its declared settings.",
        "not_rechecked": ["Encoded artifact file lengths", "bitwise decode", "payload identities against unbundled streams"],
    }, {
        "file": (LOSSLESS / "INDEPENDENT_REVIEW.json").relative_to(ROOT).as_posix(),
        "recorded_passed": historical["passed"],
        "recorded_scope": historical["review_scope"],
        "roster_file": (LOSSLESS / "ROSTER_IDENTITY_REVIEW.public.json").relative_to(ROOT).as_posix(),
        "recorded_roster_passed": roster_audit["passed"],
    }


def verify_manifest(v):
    # Do not bind unrelated manuscript data: those may be updated independently.
    entries = json.loads((ROOT / "results/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    prefixes = ("results/acf_case/", "results/lossless_comparison/")
    selected = [entry for entry in entries if entry["file"].startswith(prefixes)]
    v.require(len(selected) == 34 and len({e["file"] for e in selected}) == 34,
              "Supplemental source manifest coverage")
    for entry in selected:
        path = ROOT / entry["file"]
        v.read(path)
        v.require(path.stat().st_size == entry["bytes"] and v.inputs[entry["file"]] == entry["sha256"],
                  f"Supplemental manifest size/hash: {entry['file']}")
    return {"files": 34, "byte_exact_source_copies_recorded": 32,
            "documented_path_only_public_copies": 2,
            "scope": "Current public bytes checked against the release manifest. Private originals are not opened."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify without writing any files.")
    args = parser.parse_args()
    v = Verifier()
    manifest = verify_manifest(v)
    acf, acf_history = verify_acf(v)
    lossless, lossless_history = verify_lossless(v)
    report = {
        "schema": "jointzip.public-supplemental-results-verification.v1",
        "passed": True,
        "scope": "Fresh standard-library verification of public recorded results, rosters, recorded-metric gates, and aggregation only.",
        "fresh_encoding": False, "fresh_decoding": False, "fresh_model_inference": False,
        "fresh_bootstrap": False, "private_sources_accessed": False,
        "checks": v.checks, "numeric_comparisons": v.numeric_comparisons,
        "max_absolute_aggregation_difference": v.max_absolute_difference,
        "aggregation_tolerances": {"relative": 1e-12, "absolute": 1e-10},
        "constraint_slack": NUMERICAL_SLACK,
        "source_manifest": manifest,
        "acf_case": acf, "uniform_lossless_accounting": lossless,
        "historical_audit_records_not_rerun": {"acf": acf_history, "lossless": lossless_history},
        "input_sha256": dict(sorted(v.inputs.items())),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if not args.check_only:
        output = ROOT / "generated/supplemental_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
