# Frozen EDBT-source main matrix

This directory evaluates the project's unchanged core codec on the six EDBT
2024 authors' data sources. It is an independent selected-window evaluation,
not a reproduction of every row of the original forecasting paper.

The active run is:

    experiments/results/vldb_edbt_main_20260907_v1

Its immutable freeze SHA-256 is:

    57ffd6211a48aaf56e9238db07850aac9d272552fa3b3914b2347982c1fcb25c

The protocol traverses 142 series, 8 validation and 32 test windows each,
13 method/ablation arms, three target points and two validation-selected
comparisons. It plans 17,750 rows including calibration, failures and
unavailable grid selections. The compressed input is 128 float32 history
values, with an untouched 24-value future. Model context uses the last 96
history values. Each row bills a full SCS2 artifact, including NOR2/PRV1 and
lossless stream closure. Each artifact is read and decoded three times.

## Runtime

Use WSL /usr/bin/python3, the existing three official project binaries, the
CAMEO extension, and the frozen SZ3/ZFP builds in tmp/modern-codecs. The latter
can be built with build_modern_codecs.sh. A dedicated dependency directory
avoids changing the user's main Python installation:

    tmp/vldb_main_20260907_python_deps

From the repository root, use these environment variables in WSL:

    OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1
    PYTHONPATH=/opt/jointzip/tmp/vldb_main_20260907_python_deps:/opt/jointzip/src:/opt/jointzip

Windows Python's Torch import failed on this machine; the evaluation therefore
runs in WSL. native_bridge changes only the process/path transport of the
existing adapters within this evaluation process. Core source was not edited.

## Commands and evidence

    python3 -m experiments.vldb.main_benchmark.self_check
    python3 -m experiments.vldb.main_benchmark.benchmark freeze --output NEW_RUN
    python3 -m experiments.vldb.main_benchmark.benchmark run --output NEW_RUN
    python3 -m experiments.vldb.main_benchmark.benchmark status --output RUN
    python3 -m experiments.vldb.main_results.summary --run RUN --verify-artifacts

The current background driver runs the complete matrix, then calls the
summary command automatically. Do not launch a duplicate against its output.
Check WINDOWS_LAUNCH.json, DRIVER.json and RUNNING.json plus the actual
process state before resuming an interrupted run. The driver lock is exclusive;
the benchmark's run command supports reusing final SHA-checked rows but is
not a concurrent multi-writer API. Existing failures stay visible; changing
code or protocol requires a new freeze/run directory.

Files:

- freeze.json: protocol, code/binary/dependency hashes, complete source identity,
  TRAIN/VAL/TEST arrays, normalizers, split boundaries, origins and hardware.
- source_snapshot/: evaluator and core files captured at freeze time.
- preflight/: synthetic integration and native boundary/cross-decode checks.
- models/: raw-TRAIN Ridge models selected only on raw VAL.
- selections/: baseline epsilon choices made on VAL before TEST compression.
- artifacts/: actual self-contained compressed bytes.
- rows/: every completed, failed or unavailable job; no successful-only export.
- stdout.log / stderr.log: background process logs.
- FINISHED.json: terminal matrix state, including source and row failures.
- summary/: JSON/CSV/Markdown report. Partial snapshots have no final
  six-dataset superiority claim. Dataset pooled CR and series-mean CR are
  separate fields.

The independently frozen DLinear consumer is in ../downstream_models and
experiments/results/vldb_edbt_dlinear_20260907_v1. Its six models have been
trained; a bounded follower evaluates new artifacts until matrix completion.
Its forecast weights are a common raw-trained consumer and are not codec
payload costs.

The scientific interpretation, limitations, sources and current results are
in docs/vldb/CORRELATION_ACF_BENCHMARK_REPORT_2026-09-07_ZH.md.
