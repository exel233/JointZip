# JointZip

**A Compression Framework for Preserving Pointwise Accuracy, Correlation, and Autocorrelation**

JointZip compresses time-series archives for reuse when future analytical tasks
are not known at ingestion time. It combines statistics-guided, window-level
error-bound search over a replaceable compressor with decoded-output validation
and correction. The final representation accounts for the compressed base,
correction records, and framing, and supports source-independent decoding.

This repository contains the implementation reference, supplementary experimental
record, and result data accompanying the manuscript. The evaluated backend
configurations are SZ3, ZFP, SPERR-1D, Serf-XOR, Serf-QT, and SZ2016. The four
application families are forecasting, classification, retrieval, and anomaly
detection.

## Contents

| Location | Contents |
| --- | --- |
| [code/](code/) | Selected source snapshots for the current framework and native interfaces |
| [docs/CODE.md](docs/CODE.md) | Code map, dependencies, build requirements, and synthetic checks |
| [paper/JointZip_implementation_results.pdf](paper/JointZip_implementation_results.pdf) | Supplementary implementation details and experimental results |
| [paper/data/](paper/data/) | Current manuscript table and figure inputs |
| [results/](results/) | Final native-comparison pairs and framework-ablation summaries |
| [docs/RESULTS.md](docs/RESULTS.md) | Result provenance, units, aggregation, and interpretation |
| [scripts/reproduce_large_table.py](scripts/reproduce_large_table.py) | Rebuild the detailed downstream table from retained results |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Backend dependencies and their respective terms |

## Reproduce the detailed result table

Python 3.10 or newer is sufficient for this step; it uses the standard library
and does not run compression or model inference.

```sh
python scripts/reproduce_large_table.py
```

The script reads the finalized per-input comparison CSVs, checks exact coverage,
and writes `generated/downstream_large_table.csv`, its LaTeX counterpart, and
`generated/generation_audit.json`. It verifies 16,884 selected downstream pairs,
3,024 TEST compression pairs, and all 1,008 source/model/setting cells before
reproducing the manuscript's 43-row detailed table. It also checks the current
backend-level aggregate values. The retained reference inputs are not modified.

## Run the codec-interface checks

Install the small Python dependency set, then run:

```sh
python -m pip install -r requirements.txt
python -B tests/smoke_offline.py
```

See [docs/CODE.md](docs/CODE.md) for the test scope and native build requirements. The
small synthetic checks exercise framing, independent decoding, and ablation
isolation. Native codec experiments additionally require the pinned third-party
libraries, the specified datasets, and the experiment environment. This release
provides source references and result reproduction; it does not bundle the
original machine environment or promise a one-command rerun of every experiment.

## Reading the results

- `E`, `R`, and `A` denote pointwise accuracy, source-reconstruction Pearson
  correlation, and finite-lag autocorrelation requirements. The complete ablation
  compares E, ER, EA, ERA, and ERA without correction.
- Native and JointZip comparisons use the same requested error tolerance. They
  do not impose the same achieved compression ratio.
- Forecasting and anomaly results include the prescribed strict, medium, and
  loose settings. Classification and retrieval use the UCR stress setting.
- Downstream pair files already average the prescribed seeds. Table reproduction
  then balances inputs within each source/model/setting and averages models
  equally. Backend-level quality summaries balance sources and settings.
- Compression ratios charge complete encoded representations. Classification
  and retrieval share the same UCR compression artifacts; these are not counted
  twice when aggregating storage.
- Correction comparisons use the recorded ERA candidate trajectories. Their
  scope is the contribution of correction within those candidate pools, not
  optimality of a separately tuned algorithm without correction. Improvements,
  ties, and declines are retained in the released results.
- These datasets were also used during development. The results characterize
  the stated experimental setting and do not establish universal downstream
  guarantees.

## Dependencies and data

Third-party codec sources, compiled binaries, model weights, and raw datasets
are not redistributed in this repository. Backend sources and build requirements
are identified in the code documentation and notices. Data acquisition and
full experiment setup require the original providers' terms.

The original project sources did not include a project-level licence; this
snapshot does not introduce a new licence grant. Third-party projects retain
their own terms, including the research/non-commercial restriction noted for
Serf. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
