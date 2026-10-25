# Experimental result archive

This release contains machine-readable results used by the October 21, 2026
JointZip manuscript. CSV files are copied byte for byte from finalized analyses;
no compression, training, inference, threshold selection, or scientific rerun was
performed to package this archive. `results/SOURCE_MANIFEST.json` records each
copied file's original project-relative path, byte size, and SHA-256 hash.

## Evidence map

| Release location | Final source | Purpose |
| --- | --- | --- |
| `results/native_comparison/vldb_framework_full_20261015_v1/` | Five-backend completed comparison | SZ3, ZFP, SPERR-1D, Serf-XOR, and Serf-QT native versus JointZip |
| `results/native_comparison/vldb_framework_sz2016_20261016_v1/` | Completed SZ2016 comparison | The same comparison and independent timing for SZ2016 |
| `results/ablation/vldb_framework_ablation_20261017_v1/` | Completed six-backend ablation | E, ER, EA, ERA, and ERA without correction; all prescribed settings |
| `paper/data/` | October 21 manuscript data | Exact displayed aggregate inputs, large table, and complete companion tables |

Each native-comparison directory contains `downstream_pairs.csv`,
`downstream_summary.csv`, `cr_pairs.csv`, `cr_summary.csv`, `timing_pairs.csv`, and
`timing_summary.csv`. A downstream pair is an input/model/setting/metric record
whose prescribed seeds have already been averaged. A compression pair represents
stored artifacts, rather than repeating storage once per downstream consumer.
The independent timing tables are distinct from diagnostic execution timings.

The ablation directory contains source/model aggregates, arm summaries, paired
contrast summaries, mechanism summaries, and the correction CR overview. These
are numeric results and public benchmark identifiers, not redistributed time
series or trained model weights. In particular:

- `cr_overview.csv` and `downstream_overview.csv` contain the final coordinates
  needed for the four-task E/ER/EA/ERA figure.
- `downstream_source_model.csv` retains the source/model/setting level below the
  four-task overview.
- `downstream_contrast_summary.csv` and
  `downstream_contrast_overview.csv` retain improvements, ties, and declines.
- `mechanism_overview.csv` and `mechanism_summary.csv` report complete ERA
  checks, exact fallback, correction counts, and bytes.
- `correction_cr_overview.csv` reports the fixed-candidate-pool comparison.

The large native/framework table uses `downstream_pairs.csv` and `cr_pairs.csv`
from both native-comparison runs. `paper/data/downstream_gain_cells.csv` is its
1008-cell source/model/setting cross-check input, while
`paper/data/downstream_large_table.csv` retains the complete numeric table.
`paper/data/backend_overview.csv` and `backend_timing_table.csv` supply the
overview and independent timing values. The ablation plot uses
`paper/data/ablation_cr_overview.csv` and
`paper/data/ablation_downstream_overview.csv`.

## Coverage and units

The study includes 172 inputs: 142 forecasting series (137 Solar variables and
one from each of five other sources), six UCR datasets, and 24 NASA MSL/SMAP
files. The six backend configurations are SZ3, ZFP, SPERR-1D, Serf-XOR, Serf-QT,
and SZ2016. The two Serf configurations share a compressor family.

The complete five-arm ablation covers 930 compression configurations per backend
and arm, or 27,900 in total. Its 220,462 consumer records comprise 206,752
forecasting, 1,488 classification, 1,302 retrieval, and 10,920 anomaly records.
These counts include validation, prescribed seeds, Raw, and verified reuse; they
are not independent statistical samples or all newly executed inferences. ERA
reuses 5,580 compression records and 43,596 consumer records from the completed
native/framework studies; 2,482 Raw records are also reused. The four E/ER/EA/ERA
arms shown in the main paper contain 22,320 compression configurations. All
27,900 five-arm configurations passed their enabled reconstruction constraints
under the prescribed numerical convention.

The native/framework comparison has 3,024 TEST artifact pairs and 103,410 TEST
window conditions per side. JointZip satisfies the joint contract in all 103,410
TEST window conditions; native compression does so in 40,722. This is a pooled
window-condition count, not a count of independent datasets.

The contracts for forecasting and NASA are strict `(0.10, 0.999, 0.01)`, medium
`(0.25, 0.995, 0.02)`, and loose `(0.50, 0.990, 0.05)` for pointwise tolerance,
minimum source-reconstruction Pearson correlation, and maximum ACF distortion,
respectively. UCR classification and retrieval use stress `(1.00, 0.990, 0.05)`.
The ACF contract checks lags 1 through 16. Inputs use Float32 values standardized
with TRAIN statistics. NASA contracts use edge-padded blocks, while downstream
scores use the trimmed TEST suffix.

Quality first averages seeds, then inputs within a source, then sources equally.
Overview values also average models equally. Forecasting has six equally weighted
sources, UCR six datasets, and NASA two sources (MSL and SMAP). Forecasting and
NASA overviews weight strict, medium, and loose equally; UCR uses stress only.
The large table retains source and setting separately. Forecasting percentage
changes are ratios of resulting macro-average MSEs, not averages of per-record
percentage changes. Classification accuracy and retrieval AP are stored as
fractions; figures/tables may display them as percentages. Anomaly VUS-PR is a
fraction and is displayed in either fraction or percent units as labeled.

Complete-stream CR is unpadded Float32 input bytes divided by all stored JZF1
bytes, including normalization metadata, lengths, framing, backend bytes, and
corrections. CR overviews use source- and setting-balanced geometric means.
Comparisons at the same user tolerance do not imply equal CR.

## Interpretation boundaries and retained mixed results

E retains the complete framework search, unlike the single-call native baseline.
Disabling R or A disables the corresponding anchor, repair, and acceptance
components; these are whole-framework constraint ablations. ERA improves the E
macro average in each of the four task families for every backend, but does not
uniformly outperform ER or EA. R's effect is generally stronger for normalized
shape-sensitive tasks; A's additional downstream effect varies by task/backend.

The correction comparison replays the ordered logical-cap/phase trajectories
recorded by ERA, with the same native-cap gates, joint acceptance, exact candidate,
and tie rules. It is conditional on that candidate pool and does not establish
the optimum of an independently adapted no-correction algorithm. Both sides meet
ERA. All complete-stream accounting is retained.

Correction raises aggregate SZ3 CR by 12.7802% for forecasting, 17.1986% for UCR,
and 1.8274% for NASA. Other effects are often small; SPERR-1D forecasting CR drops
by 0.0650% and Serf-XOR NASA CR by 0.3463%. Across 3,024 TEST correction artifact
pairs, complete CR improves in 489, ties in 2,295, and declines in 240. Forecasting
MSE macro averages are slightly higher with correction for all six backends.
No significance claim is made for these small differences. These outcomes remain
in the released tables even when not emphasized in the main-paper narrative.

Native/framework comparisons also retain unfavorable cases. For example,
SPERR-1D anomaly VUS-PR falls from approximately 0.3321 to 0.2803, despite its
jointly validated reconstruction. Constraint satisfaction is not a universal
guarantee of better task scores, identical-rate superiority, or monotone CR.

No new independent timing experiment was conducted for the ablation. Do not use
parallel execution, cache hits, fixed-trajectory replay durations, or reused
consumer timings as codec speed measurements. The separate native-comparison
timing files contain 12 paired conditions per backend spanning six sources.

The input data informed framework development, so the results are exploratory
evidence on the specified workloads. Frozen models and short contexts assess
reconstruction sensitivity; pretrained-model overlap is not excluded. The
release contains no claim about unseen tasks or benchmark-wide optimality.

## Companion document

`paper/JointZip_implementation_results.pdf` is the 11-page companion rebuilt
from the October 21 manuscript's implementation/ablation sources. It contains
correction-format details, numerical conventions, and all 24 aggregate tables,
including unfavorable results. It is separate experimental documentation, not
extra pages permitted for the research paper.

The release changes presentation only: the title page names Xingjian Jiang and
Yinan Yu, both at Chalmers University of Technology, Sweden; unassigned
publication DOI/pages and proceedings frontmatter are omitted. The scientific
body and included table sources remain byte-identical to the manuscript source
snapshot. No numerical values, experimental policies, or result interpretations
were changed. The source manifest distinguishes this rebuilt PDF from the
byte-identical CSV and table-source copies.

The older grid-version `supplement.pdf`, historical quality-policy CSVs, local
machine logs, browser/session data, raw benchmark files, and model checkpoints
are deliberately not part of this result archive. Dataset/model acquisition and
runtime requirements must be followed separately to rerun the scientific study;
reproducing plots from these recorded results is not a fresh scientific run.
