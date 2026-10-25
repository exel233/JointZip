# Source snapshot and focused reproduction

This artifact includes the JointZip codec source used by the October 15--17,
2026 experiments, the local native adapters and repair implementation, and a
portable synthetic invariant test. Original source files are copied without
modification; `code/SOURCE_MANIFEST.json` records their SHA256 hashes and sizes.
The repository does not contain upstream library checkouts, compiled binaries,
datasets, model weights, or the historical experiment execution environment.

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
| `experiments/vldb/framework_*/container.py` | Source-independent JZF1 decode entry point. |
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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
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

The complete historical experiment launcher is not included. Its data loading,
consumer-model setup, prior-result reuse and checkpoint orchestration depend on
the original local environment. This source snapshot is therefore not a
turnkey reproduction of all 27,900 compression configurations and 220,462
consumer records. The consumer total includes validation, seeds and reuse;
it is not the number of independent datasets or fresh model inferences.

The fixed-trajectory no-correction comparison additionally requires the ordered
per-window metadata from the matching ERA run. Disabling correction and doing
a new adaptive search is a different experiment. Cache/replay wall time is not
an encoding-speed measurement. Results and paper figures in other repository
directories document the published evaluation separately from this focused
code smoke test.

The portable test was validated with Python 3.10, NumPy 1.26.4 and
python-zstandard 0.25.0 on Windows. All 44 original files in the source manifest
were checked against their source SHA256 hashes. Native compilation and full
experimental reruns were not part of this validation.
