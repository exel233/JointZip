# Experiment procedures and reproduction boundaries

The paper uses the scientific implementations in `code/experiments/vldb/`.
This document identifies their inputs, frozen choices and outputs. Paths below
are relative to `code/`, unless a repository-level path is explicitly stated.
The package contains source and analysis tables, not the complete historical
execution directories. Running a driver requires the inputs listed below.

## What can be reproduced from this checkout

| Operation | Available material | Additional requirements |
| --- | --- | --- |
| Offline codec/cache invariants | `tests/smoke_offline.py` and released codec modules | Root `requirements.txt`; no dataset or native codec. This uses a synthetic backend. |
| Published table/figure analysis | Repository `results/`, `scripts/` and `paper/` | Follow the root README and analysis script arguments; no new model training. |
| Native codec builds and decoding | Local adapters, C++ repair, build recipes and Python containers | Pinned upstream sources, compilers/libraries and matching binary identities; see `CODE.md`. |
| Main six-backend evaluation and E/ER/EA/ERA ablation | Drivers, consumers, local import dependencies, protocols and parameter reference | Original prepared inputs, Raw-trained models, historical freezes/reuse manifests, native runtimes and per-window payload/metadata. These large artifacts are not in this checkout. |
| FordA/FordB ACF experiment | TRAIN preparation, model fitting, TEST pipeline and analysis | Official dataset archives, native runtime, Python feature/model packages and preflight records. The supplied historical driver also expects its recorded runtime directory layout. |
| Uniform lossless accounting | Common-format implementation and comparison code | Original saved JointZip/JZF1 and Zstd/NCS2 payload files, input arrays and bound row metadata. Summary CSVs alone cannot regenerate or independently decode these files. |

Do not convert missing artifacts into successful checkpoints. New builds,
changed paths or newly trained models need their own recorded source/runtime
identity. The reference JSON in `protocols/frozen_parameters.json` is not a
substitute for a driver's input freeze or completion gate.

## Shared data and scientific settings

The main input roster contains 142 forecasting series from six sources, six
UCR classification datasets and 24 NASA MSL/SMAP evaluation files. The exact
172 identities, prepared-input hashes, normalization parameters and true
lengths are in `protocols/frozen_parameters.json`. Data and model acquisition
are described in [DATASETS.md](DATASETS.md).

| Setting | Pointwise tolerance E | Minimum Pearson R | Maximum protected ACF gap A |
| --- | ---: | ---: | ---: |
| strict | 0.10 | 0.999 | 0.01 |
| medium | 0.25 | 0.995 | 0.02 |
| loose | 0.50 | 0.990 | 0.05 |
| UCR stress | 1.00 | 0.990 | 0.05 |

Main forecasting and NASA use all three settings. UCR uses stress. The
protected ACF is Pearson correlation between separately centered lag slices,
at lags 1 through 16. This differs from a traditional globally centered ACF.
E/R/A are checked on the final decoded values. Numerical acceptance permits
`64 * float64 epsilon`; zero-tolerance results are recorded separately.

Each codec is fixed for a comparison. SZ3, ZFP, SPERR-1D and SZ2016 use up to
22 logical caps, 12 initial proposals and search-floor divisor 512. Serf-XOR
and Serf-QT use 8, 5 and 64. The repair policy is the original posthoc greedy
search with the legacy optimizer, source-value and residual-grid targets
7/15/31/63/127, and correction formats 0--6. The encoder compares actual
base-plus-correction bytes and includes a window-level exact Zstd3 candidate.
Native arms call the same backend once at the user epsilon without R/A repair.

The streams for these runs are JZF1 containing JZP1. Full-file CR includes
normalization, channel/shape/true-length metadata, record overhead, base and
correction payloads. Padding is charged in the denominator and omitted from
the raw-byte numerator. Historical SCS2/NCS2/TSL1 parsing code remains as a
dependency for old artifacts; those names do not describe the main JZF1 stream.

## Recorded stages and dependency order

| Scientific run | Main source directory | Coverage and role |
| --- | --- | --- |
| `vldb_framework_full_20261015_v1` | `framework_full_20261015` | Five fixed backends, native/JointZip, 9,300 compression conditions and 75,142 consumer records; separate VAL density and timing panels. |
| `vldb_framework_sz2016_20261016_v1` | `framework_sz2016_20261016` | SZ2016 under the same native/JointZip policy, 1,860 conditions and 17,014 consumer records. |
| `vldb_framework_ablation_20261017_v1` | `framework_ablation_20261017` | Six backends and five arms: E, ER, EA, ERA and fixed-trajectory ERA without correction; 27,900 conditions and 220,462 consumer records. |
| `vldb_acf_forda_fordb_20261028_v1` | `acf_forda_fordb_20261028` | Two datasets, two backends and ER/two ERA thresholds; 12,780 TEST windows, 12 complete streams and 42 consumer metrics. |
| `vldb_uniform_lossless_accounting_20261031_v1` | `uniform_lossless_accounting_20261031` | Repackage 3,024 saved JointZip TEST streams and 172 saved Zstd streams under common metadata; no codec or downstream rerun. |

The October 15 `prepare.py` copies the frozen inputs from
`vldb_simple_correction_20260929_v1`; the shared consumers also require the
earlier baseline expansion and Raw-trained model manifests. October 16 checks
the completed October 15 run. October 17 checks both previous runs and their
delivery/preflight bindings. Its 5,580 ERA compression conditions, 43,596 ERA
consumer records and 2,482 Raw records reuse matching historical results;
other exact-input/runtime matches may also reuse consumer outputs.

These dependency checks are intentional. The release includes preparation and
training sources for understanding how the input/model lineage was produced,
but does not provide a new bootstrap that fabricates historical result gates.
The original preparation helpers themselves can require earlier freezes,
encoded artifacts and validation records. Rebuilding from raw data is therefore
a documented environment port and new execution, rather than a single command
against the compact public package.

## Main drivers and consumers

Once the complete dependencies and path configuration are present, the actual
Linux entry point for each main run is:

```sh
cd code
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.vldb.framework_full_20261015.pipeline --execute --workers 4
python -m experiments.vldb.framework_sz2016_20261016.pipeline --execute --workers 4
python -m experiments.vldb.framework_ablation_20261017.pipeline --execute --workers 4
```

These are reference entry points, not commands that will complete on a fresh
checkout. The native environment described in `CODE.md` must already be set.
For an interrupted run, the driver requires explicit `--resume`; never start a
second driver while its lock or worker processes remain active.

Each pipeline performs preparation/preflight, compression, every forecasting
VAL consumer, every forecasting TEST consumer, UCR classification/retrieval,
NASA anomaly consumers, exact closeout and report generation. The source
`pipeline.py` is the authoritative stage list. Representative consumer entry
points, after their compression inputs exist, are:

```sh
python -m experiments.vldb.framework_full_20261015.consumers forecast --model ridge --split val
python -m experiments.vldb.framework_full_20261015.consumers forecast --model patchtst --seed 20260915 --split test
python -m experiments.vldb.framework_full_20261015.consumers ucr --consumer resnet_v2 --seed 20260915
python -m experiments.vldb.framework_full_20261015.consumers anomaly --model lstm --seed 20260915
```

| Task | Frozen consumers | Main reported metrics |
| --- | --- | --- |
| Forecasting | Ridge, DLinear, PatchTST, Chronos-2, Chronos-Bolt-small, TimesFM 3.0 | MSE; source/model/setting aggregation retains per-consumer values. |
| Classification | Shape 1NN, logistic regression, ResNet v1 and v2 | Accuracy; both ResNet protocols are retained. |
| Retrieval | Shape distance and frozen ResNet v1/v2 embeddings | Mean average precision and other saved rank metrics. |
| Anomaly detection | `ridge_svd64`, KNN and LSTM | VUS-PR, with other saved detection metrics. |

PatchTST, both ResNet protocols and LSTM use seeds 20260915, 20260916 and
20260917. All consumers are fit or obtained using Raw data and then held fixed
across compressed inputs. Foundation models are inference-only, with 96-point
context and 24-point median forecasts. Training/selection choices and versions
are available in the source and parameter reference. Average seeds first, then
use the paper's source/model/setting hierarchy; do not count the 137 Solar
channels as 137 equally weighted source domains.

## E/R/A and correction ablations

E/ER/EA keep the full framework search while omitting disabled constraints from
anchor generation, correction and acceptance. E is not the native one-cap arm.
All enabled constraints must pass, while audit fields also expose the remaining
R/A behavior.

The no-correction experiment replays each existing ERA window's ordered
logical-cap/phase sequence. It uses unrepaired native bases, checks the logical
cap and E/R/A feasibility, and retains the same exact candidate and tie rules.
It requires the matching ERA metadata and original base identities. Turning
off repair and rerunning adaptive search would be a different experiment.
Native cache entries store source/backend/configuration identity and effective
cap; ZFP's logical cap is checked on every replay.

`framework_ablation_20261017/report.py` produces arm and contrast summaries,
CR and downstream overviews, and correction/mechanism summaries. The correction
results remain available even where they are not a main-paper figure. More
constraints do not imply a gain in every downstream task, and an equal epsilon
does not imply an equal CR or equal realized E/R/A.

## FordA/FordB ACF supplement

The official complete 500-point records are used: FordA TRAIN/TEST = 3601/1320
and FordB = 3636/810. Keep the official normalization. The two backends are SZ3
and ZFP, with E = 1 and R = 0.99. Compare ER against ERA at tau = 0.05 and 0.01.
The common lag limit, L = 16, was chosen from TRAIN first-local-minimum coverage,
before TEST. The exact rule and input archive hashes are in the parameter
reference and `prepare.py`.

The primary consumer uses all 22 pycatch22 features. The complementary consumer
uses 32 equal-width non-DC relative periodogram-energy bands. Both use Raw-TRAIN
median imputation, StandardScaler and fixed logistic regression (`C=1`, balanced
classes, `lbfgs`, `max_iter=5000`, seed 20261028). The first ACF minimum alone is
a mechanism diagnostic, not a third independent validation of task utility.

After provisioning the archives and runtime required by `common.setup_runtime`,
the historical order is:

```sh
python -m experiments.vldb.acf_forda_fordb_20261028.prepare
python -m experiments.vldb.acf_forda_fordb_20261028.smoke
python -m experiments.vldb.acf_forda_fordb_20261028.validate_train
python -m experiments.vldb.acf_forda_fordb_20261028.pipeline --train-only
python -m experiments.vldb.acf_forda_fordb_20261028.pipeline --workers 4
```

These modules read the runtime at import time, so even `--help` needs that setup.
Completed chunk payloads are reused under an exclusive lock. The final assembly
independently decodes 12 complete JZF1 files. Outputs are `downstream.csv` (42
rows), `contrasts.csv` (24), `mechanism.csv` (12), `RESULTS.json`, `REPORT_ZH.md`
and the classification PDF/PNG. The 95% paired intervals use 5,000 within-TEST,
class-stratified bootstrap replicates and are descriptive. Preserve both tau
settings, both backends, unchanged and adverse results. No matched-CR or general
unknown-task guarantee follows from this supplement.

## Uniform complete-file lossless comparison

`uniform_lossless_accounting_20261031/compare.py` reads the saved original
October 15/16 JointZip files and the historical Zstd files from September 29.
It is run as a script after all payloads, rows and input bindings are available:

```sh
python experiments/vldb/uniform_lossless_accounting_20261031/compare.py
python experiments/vldb/uniform_lossless_accounting_20261031/report.py
```

Both sides receive identical per-input JSON fields from JZF1 in a common JCF1
header. JointZip keeps its JZP1 bytes unchanged. Lossless stores a four-byte ZLC1
dispatch tag followed by the unchanged NCS2 collection, retaining record sizes,
closure mode, shapes and ZSL1 window boundaries. The reference uses historical
windowed Zstd19 with its recorded batch closure, not JointZip's window Zstd3
candidate. Required decoding metadata is charged to both sides; external
experiment provenance is excluded from both.

All new lossless files are independently decoded and checked bitwise. JointZip
is checked by byte-identical restoration of the original complete JZF1 file,
its source hash and the existing independent-decode evidence. The comparison
does not claim a fresh rerun of every native JointZip decoder. It emits
`conditions.csv`, `macros.csv`, `settings.csv`, `sources.csv` and associated
bindings. The 18 backend/task macro comparisons and the 3,024 individual
conditions are distinct units; positive macro results do not imply that every
condition beats lossless storage.

## Timing, analysis and completion checks

Timing is a separate standalone paired panel, using the first series from each
forecasting source, its first four VAL windows, strict/loose, one warmup and
three measured repeats, alternating native/JointZip order. October 15 has
480 rows including 120 warmups; October 16 has 96 including 24 warmups. Encoding
includes search, candidate decode/repair and serialization; the final full-file
decode is timed separately, with final audit checks outside the timing. Reuse,
parallel execution and cache/replay wall time are not codec speed measurements.

Main completion requires `execution/COMPLETE.json` and a passing exact
`audit/closeout/CLOSEOUT.json`, including empty missing/error/unexpected sets and
successful enabled numerical constraints. Ford completion requires 12,780
windows, 12 streams and 42 metrics plus complete CSVs. Summary tables and figures
should be regenerated from the released machine-readable evidence, retaining
the defined aggregation and all declared comparisons. Figures are not a
substitute for these coverage checks.
