# Source snapshot and reproduction scope

This artifact includes the JointZip codec, experiment drivers, downstream
consumers, preparation and training entry points, and analysis code for the
October 15, 16, 17, 28 and 31, 2026 evaluations. It includes the local Python
import dependencies and the dynamic local ResNet snapshot and freeze verifier,
alongside the native adapters and repair implementation. The original directory
layout is retained under `code/`.

`code/SOURCE_MANIFEST.json` records the original relative path, original SHA256,
distributed SHA256 and size of each file. Of the 153 original files, 139 are
byte-identical and 14 replace only machine-specific path prefixes. The one
additional `code/protocols/frozen_parameters.json` file extracts scientific
parameters and the complete 172-input roster from the recorded freezes, with
source hashes. It is a reference, not a runnable replacement for those freezes.
No encoder, selection policy, threshold, model or metric was changed.

The repository does not contain upstream library checkouts, compiled binaries,
raw datasets, model weights, full historical encoded payloads or the original
execution environment. See [EXPERIMENTS.md](EXPERIMENTS.md) for the dependency
order and actual entry points, and [DATASETS.md](DATASETS.md) for pinned data and
model sources. The source release supports inspection and reconstruction of the
workflow, while the public result tables support analysis without model runs.

## Portable offline test

Use Python 3.10 with the dependency versions in `requirements.txt`. From the
repository root:

```sh
python -m venv .venv
# Activate .venv with the command appropriate to your shell.
python -m pip install -r requirements.txt
python -B tests/smoke_offline.py
```

The test constructs a deterministic 32-point sinusoid in memory. It uses a fake
native backend while exercising the actual Python framing, candidate cache,
fixed-trajectory replay and acceptance logic. A successful run prints
`"passed": true`. It checks:

- byte-identical cached and uncached encoding;
- storage of unrepaired native candidates and separation by configuration;
- effective-cap keys for ZFP and float32 rounding for Serf;
- preservation of disabled statistical constraints;
- replay of all recorded logical caps and rejection of infeasible candidates;
- the original numerical tolerance, checkpoint replay and exact fallback.

This is a software invariant check. It does not run the official native
compressors, reproduce task scores, measure compression speed, or establish the
quality of a backend on real datasets.

## Source map

Paths in this table are relative to `code/`.

| Path | Role |
| --- | --- |
| `experiments/vldb/framework_full_20261015/codec.py` | Formula-guided full framework for SZ3, ZFP, SPERR-1D, Serf-XOR and Serf-QT. |
| `experiments/vldb/framework_sz2016_20261016/codec.py` | Same framework policy extended to SZ2016. |
| `experiments/vldb/framework_ablation_20261017/codec.py` | Six-backend E/ER/EA/ERA variants and fixed-trajectory correction ablation. |
| `experiments/vldb/framework_ablation_20261017/native_cache.py` | Persistent cache of unrepaired native candidates, with source and backend identity checks. |
| `experiments/vldb/framework_{full_20261015,sz2016_20261016,ablation_20261017}/` | `prepare.py`, `pipeline.py`, `compress.py`, `consumer_*.py`, `closeout.py` and `report.py`: frozen orchestration and four-task consumers. The notation in this row denotes three actual directories. |
| `experiments/vldb/framework_full_20261015/{density,timing}.py` | VAL-only search-density experiment and separate paired timing. SZ2016 has its own `timing.py`. |
| `experiments/vldb/acf_forda_fordb_20261028/` | TRAIN-only preparation, checks, frozen classifiers, full TEST execution and FordA/FordB analysis. |
| `experiments/vldb/uniform_lossless_accounting_20261031/` | Common JCF1 wrapper, source-independent lossless decode and full-file comparison using saved payloads. |
| `experiments/vldb/framework_*/container.py` | Source-independent JZF1 decode entry point. |
| `experiments/vldb/main_benchmark/{tasks_data,benchmark}.py` | Original six-source data coordinates, splits, origins and Raw-trained Ridge setup. |
| `experiments/vldb/downstream_models/dlinear.py` | Pinned official DLinear model loading and the declared local training protocol. |
| `experiments/vldb/downstream_expansion/` | Raw model preparation, PatchTST and foundation-model adapters, UCR models/retrieval and NASA anomaly consumers. |
| `experiments/vldb/{baseline_expansion_20260922,simple_correction_20260929,paper_budget_guided_20261013_v1}/` | Historical shared input and consumer dependencies imported by the October drivers. Only the required sources and selected preparation entry points are included, not every historical experiment. |
| `src/neural_compression_lab/` | Local historical framing and data helpers imported by the preparation code. These are not an additional evaluated JointZip backend. |
| `protocols/frozen_parameters.json` | Published threshold settings, run coverage, input identities, Ford feature protocol and foundation-model revisions. |
| `docs/vldb/FRAMEWORK_*_PROTOCOL_202610*.md` | Original written protocols for the main experiments and ablations. |
| `experiments/vldb/main_benchmark/modern_codecs.py` | SZ3/ZFP process adapter and build-identity handling. |
| `experiments/vldb/main_benchmark/sz3_api_cli.cpp` | Local SZ3 API wrapper. |
| `experiments/vldb/framework_full_20261015/native/fixed_backend.cpp` | Local SPERR-1D and Serf encode/decode bridge. |
| `experiments/vldb/framework_pilot_20261014/native/anchors.cpp` | Formula-anchor helper reused by the full framework. |
| `tools/qpet_correlation_guided_20261013_v1/` | Shared native constraint, anchor and correction implementation, analytical tests and build files. |
| `tools/sz2016/` | SZ2016 batch wrapper, configuration, upstream safety patch and build recipe. |

The directory names preserve the original source layout and imports. The
`framework_pilot` directory contains only the reused anchor helper, not an
alternative experimental pipeline. The `qpet_correlation_guided` name reflects
the shared implementation's source history and header dependency; it is not a
claim that the full JointZip framework is the QPET baseline.

The shared native executable also contains historical experimental modes. The
released Python framework selects the simple posthoc greedy correction with
the existing legacy representation and formats 0--6. Other modes retained in
the native compilation unit are not substituted for that policy. E/ER/EA retain
the framework search with the corresponding statistical constraints disabled;
E is not a single native encoding at epsilon. The correction ablation replays
the recorded ordered logical-cap/phase trajectory from an existing ERA run.

## Optional native builds

The original Linux/WSL build recipes are included for inspection and rebuilding.
They fetch pinned upstream sources and require network access. They write build
products below `code/tmp/` and the corresponding source directories. They are
not required by the portable offline test and have not been rebuilt as part of
this public snapshot's validation.

Requirements include a C/C++17 toolchain, Git, CMake (at least 3.20 for the shared
repair project), Ninja, OpenMP, pkg-config, Zstd development files, GSL and GMP.
The repair build fetches SymEngine and the pinned QPET, SPERR and Serf sources.
See `THIRD_PARTY_NOTICES.md` for exact revisions.

Run from `code/` in a Linux shell:

```sh
# SZ3 and ZFP; builds native CLIs and a local build manifest.
bash experiments/vldb/main_benchmark/build_modern_codecs.sh

# Shared repair executable, analytical tests and pinned dependencies.
bash tools/qpet_correlation_guided_20261013_v1/build_wsl.sh

# SPERR-1D/Serf bridge, after the preceding dependencies exist.
bash experiments/vldb/framework_full_20261015/native/build.sh

# Formula-anchor helper, after the QPET headers are available.
g++ -O3 -std=c++17 \
  -Itmp/upstreams/QPET-szfamily-revision/include \
  experiments/vldb/framework_pilot_20261014/native/anchors.cpp \
  -o experiments/vldb/framework_pilot_20261014/native/anchors

# Optional SZ2016 wrapper; see the identity limitation below.
bash tools/sz2016/build_wsl.sh
```

`build_isolated_wsl.sh` in the repair directory reuses existing upstream builds;
it does not fetch or provision them. Some retained build-directory names refer
to earlier internal revisions. They are filesystem labels and do not change
the included source directory or pinned dependencies.

For the ablation or SZ2016 Python module, explicitly select the shared bridge,
whose default relative location differs from those modules' directories:

```sh
export JOINTZIP_FULL_BRIDGE="$PWD/experiments/vldb/framework_full_20261015/native/fixed_backend"
export JOINTZIP_FULL_BRIDGE_MANIFEST="$PWD/experiments/vldb/framework_full_20261015/native/manifest.json"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Further overrides are `JOINTZIP_PILOT_REPAIR`, `JOINTZIP_PILOT_ANCHORS`,
`JOINTZIP_PILOT_NATIVE_LIBS`, `MODERN_CODECS_PREFIX`, `JOINTZIP_SZ2016_BINARY`
and `JOINTZIP_SZ2016_CONFIG`; consult the source defaults before using them.
The JZF1 payload includes the information required by its decoder but decoding
still requires the corresponding native backend and correction implementation.

**SZ2016 identity limitation.** The October 16/17 adapters enforce the original
binary SHA256
`34c74e64fc2308bf3bdfefcd84e2f900083bd54cfe679cdb55d58966b601b2b4`
and configuration SHA256
`1d17c31061be76254e86cde1cde2adff09a819fd47097ee3da2a4c8f0a46a5ac`.
A rebuild using another toolchain can produce a different binary and will be
rejected by this frozen source. The original binary is not distributed here.
The check has been retained unchanged; therefore a fresh SZ2016 rebuild is not
promised to run through the frozen adapter without an explicitly documented
port to a new runtime identity.

## Scope of experimental reproduction

The Python scientific drivers and their local import dependencies are included.
They retain their checks on input identities, parent results, native binaries,
model checkpoints and preflight records. Machine-specific Windows launchers,
mount setup, execution logs and stored runtime environments are not included.
The Python drivers therefore are not turnkey reproduction commands for an empty
checkout. They will stop when the required historical artifacts are missing.

Two public path prefixes replace private machine locations in the affected
files: `/opt/jointzip` denotes the `code/` workspace and
`/opt/jointzip-home` denotes the environment containing `.venvs/`, `.cache/`
and `.local/cth_expansion_runs/`. These are placeholders, not installed
locations. The manifest records the exact affected files and line numbers.
Configure an equivalent directory layout or make a documented path-only port
before starting a new run. Do not bypass failed hash or completeness checks.
The adapted source has a different source identity; an existing historical
freeze must not be relabeled as a fresh reproduction of that adapted code.

The four E/R/A arms plus the correction-ablation arm contain 27,900
compression configurations and 220,462 consumer records. The consumer total
includes validation, seeds and reuse; it is not the number of independent
datasets or fresh model inferences.

The fixed-trajectory no-correction comparison additionally requires the ordered
per-window metadata from the matching ERA run. Disabling correction and doing
a new adaptive search is a different experiment. Cache/replay wall time is not
an encoding-speed measurement. Results and paper figures in other repository
directories document the published evaluation separately from this focused
code smoke test.

The portable test was originally validated with Python 3.10, NumPy 1.26.4 and
python-zstandard 0.25.0 on Windows. The expanded snapshot was checked for source
hash correspondence, documented path-only changes, Python syntax and local
import coverage. Native compilation, model downloads and scientific reruns were
not performed when preparing the expanded snapshot. These are separate from
the validation of the recorded experiment results.
