# Supporting artifact guide

This snapshot accompanies the November 1, 2026 manuscript. It is organized for
inspection and result reproduction without access to the authors' computer.
An ordinary Git clone or the tagged source archive provides the same files.
No reviewer account or identifying request to the authors is needed to read it.

## Paper-to-artifact map

| Paper item | Evidence or editable source | Reproduction route |
| --- | --- | --- |
| Figure 1: constructed motivation examples | `paper/figures/motivation.pdf`, `.drawio`, `.svg` | Editable constructed illustration; not a benchmark measurement |
| Figure 2: framework | `paper/figures/architecture.pdf`, `.drawio` | Editable framework diagram |
| Figure 3: parameter search and stored records | `paper/figures/budget.pdf`, `.drawio` | Editable constructed illustration |
| Figure 4: correction statistics update | `paper/figures/mechanism.pdf` | Method illustration; equations and source implementation in `code/` |
| Figure 5: four-task constraint ablation | `paper/figures/ablation_tasks.pdf`, `.png`; `paper/data/ablation_downstream_overview.csv` | `python scripts/reproduce_ablation_figure.py` |
| Joint protection and storage table | `paper/data/backend_storage_rows.tex`, `backend_overview.csv`; lossless comparison results | Public per-condition and aggregate result files, with all common-wrapper bytes billed |
| Detailed downstream table | `paper/data/downstream_large_table.csv`, `.tex` | `python scripts/reproduce_large_table.py` |
| FordA/FordB ACF table | `results/acf_case/vldb_acf_forda_fordb_20261028_v1/downstream.csv`, `contrasts.csv`, `mechanism.csv` | `python scripts/verify_supplemental_results.py` checks all arms, not only the main-paper rows |
| Runtime table | `paper/data/backend_runtime_rows.tex`, `backend_timing_table.csv`; native-comparison `timing_pairs.csv` | Recorded independent timing pairs; no timing claim is made for packaging or cache replay |
| Correction ablation and additional details | `results/ablation/`; `paper/JointZip_implementation_results.pdf` | Full recorded outcomes retained, including declines |

The companion PDF is a dated October 21 implementation/ablation supplement.
The two later studies are supplied separately in `results/` and documented in
[RESULTS.md](RESULTS.md). The current main-paper source is not required to run
the result scripts. This package does not replace the conference's paper page
limit or promise that reviewers will consult external material.

## Three levels of reproduction

1. **Inspect and recompute released results.** Python 3.10+ is sufficient for
   table reconstruction, supplemental numeric checks, and SHA-256 verification.
   The ablation figure additionally needs NumPy, pandas, and Matplotlib.
2. **Exercise the interface.** The small offline smoke test uses NumPy and
   python-zstandard. It checks synthetic data and test backends, not native
   scientific performance.
3. **Run scientific experiments.** Follow [EXPERIMENTS.md](EXPERIMENTS.md) and
   [DATASETS.md](DATASETS.md). Linux, the pinned native libraries, benchmark
   inputs, model dependencies, and prepared models are additional requirements.
   Historical reuse requires the matching preceding studies and identities.
   This package is not a prebuilt container or a one-command full rerun.

No new end-to-end native experiment was performed for this artifact update.
Release checks are recorded under `validation/`, distinct from copied historical
completion/audit evidence under `results/`. Recomputed CSV arithmetic cannot
substitute for independently decoding omitted streams or rerunning inference.

## File integrity and provenance

`MANIFEST.sha256` covers the public files, apart from itself and generated outputs.
`python scripts/verify_artifact.py` checks those hashes and the public source/result
manifests. `code/SOURCE_MANIFEST.json` maps sources to the original experiment
files, recording both original and public hashes wherever machine paths are
normalized. `results/SOURCE_MANIFEST.json` identifies byte-exact result copies
and the few path-normalized protocol copies. `paper/SOURCE_MANIFEST.json` records
the current manuscript display inputs and figures. None of these transformations
changes scientific results, formulas, model parameters, or acceptance policies.

The manifests are integrity records, not signatures or an independent
reproducibility certification. Historical completion records can refer to original
hashes or paths not packaged here; those references are explained in RESULTS.md.

## Durable access and policy

The public repository is hosted on GitHub and this snapshot has a dated tag.
Keep that tag available during review and after publication. An updated version
should receive a new tag rather than replacing this snapshot. There is no DOI
or archival-service deposit associated with this version.

The manuscript uses the official PVLDB Artifact Availability field. Public
author-identifying GitHub hosting is compatible with VLDB 2027's single-blind
review. This submission artifact is distinct from the later reproducibility
evaluation and badge process. See the [submission guidelines](https://vldb.org/2027/submission-guidelines.html),
[publication policy](https://www.vldb.org/pvldb/general-information/publication-policies),
and [reproducibility information](https://www.vldb.org/pvldb/reproducibility).
