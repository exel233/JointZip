# JointZip

**A Compression Framework for Preserving Pointwise Accuracy, Correlation, and Autocorrelation**

JointZip compresses time-series archives whose future analytical tasks are not known at ingestion time. It wraps a compatible error-bounded compressor with window-level parameter search, validation of decoded values, and selective correction. Pointwise error (E), source–reconstruction Pearson correlation (R), and finite-lag autocorrelation (A) are checked together. Stored corrections and metadata are included in the encoded size, and decoding needs no original data.

This is the supporting artifact for the manuscript by **Xingjian Jiang and Yinan Yu**, Chalmers University of Technology. It covers six backend configurations (SZ3, ZFP, SPERR-1D, Serf-XOR, Serf-QT, and SZ2016), four downstream task families, the E/ER/EA/ERA ablation, the FordA/FordB ACF study, and the common complete-file accounting comparison with lossless Zstd.

## Start here

| Goal | Entry point |
| --- | --- |
| Understand what is supplied and how it maps to the paper | [Artifact guide](docs/ARTIFACT.md) |
| Rebuild the detailed table, including task means | [Table script](scripts/reproduce_large_table.py) |
| Rebuild the four-task E/ER/EA/ERA bar figure | [Figure script](scripts/reproduce_ablation_figure.py) |
| Check the two later studies from retained numeric results | [Supplemental checks](scripts/verify_supplemental_results.py) |
| Read implementation and run a small codec-interface test | [Code guide](docs/CODE.md) |
| Obtain benchmarks and model sources | [Data guide](docs/DATASETS.md) |
| Configure and run the scientific workflows | [Experiment guide](docs/EXPERIMENTS.md) |
| Inspect every result, unit, aggregation rule, and mixed outcome | [Result guide](docs/RESULTS.md) |
| Check source provenance and dependency terms | [Source manifest](code/SOURCE_MANIFEST.json), [notices](THIRD_PARTY_NOTICES.md) |

## Rebuild tables and check results

Python 3.10+ and its standard library are sufficient. These commands read the released result CSVs and do not run compression, training, or inference.

```sh
python scripts/reproduce_large_table.py
python scripts/verify_supplemental_results.py
python scripts/verify_artifact.py
```

Outputs go to `generated/`. The large table includes 51 display rows: 36 source/setting quality rows, eight source-equal Mean rows, and seven compression rows. Its 306 paired cells are checked against the manuscript's unrounded data. Use `--check-only` with either of the first two commands for a read-only check.

To regenerate Figure 5:

```sh
python -m pip install -r requirements-figures.txt
python scripts/reproduce_ablation_figure.py
```

This produces the four-task, six-backend, four-arm figure (96 bars), including the labeled axis breaks and signed ERA–ER changes. The distributed PDF and PNG are in [paper/figures](paper/figures/).

## Check the codec interface

```sh
python -m pip install -r requirements.txt
python -B tests/smoke_offline.py
```

These synthetic checks exercise framing, source-independent decoding, and constraint-ablation isolation using test backends. They do **not** certify a native codec or reproduce the reported task scores. The native build and experiment instructions are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Scope of the evidence

The result archive preserves all prescribed settings and unfavorable outcomes. Comparisons use the same requested tolerance, not a matched compression ratio. The correction ablation is conditional on the recorded candidate pools. The Ford study includes both A thresholds and both classification consumers, while the lossless study charges both methods' complete files under common metadata. These are exploratory results on the specified workloads, not a guarantee of improvement on every unknown task.

The original native runs are not rerun when this package is assembled. Source snapshots, recorded scientific results, and the checks executed for this release are identified separately. See [validation](validation/) and the artifact guide.

## Version, citation, and terms

The `vldb2027-submission-20261101-r2` tag fixes this supporting-material snapshot. Use that tag or its commit hash when citing results, rather than a moving branch. Citation metadata is supplied in [CITATION.cff](CITATION.cff). This repository does not claim a PVLDB reproducibility badge or acceptance of the manuscript.

Raw benchmarks, pretrained weights, and native binaries are obtained separately from their providers. Dataset/model identities and preparation entry points are documented. The project has not granted a new project-wide software licence; upstream dependencies retain their own terms, including Serf's non-commercial restriction. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Private logs, credentials, local caches, and unrelated working files are not part of the artifact. See the brief [tool-use statement](docs/TOOL_USE.md).
