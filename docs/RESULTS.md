# Experimental result archive

This release contains machine-readable results from the October 15--31, 2026
JointZip studies. CSV files are copied byte for byte from finalized analyses;
no compression, training, inference, threshold selection, or scientific rerun was
performed to package this archive. `results/SOURCE_MANIFEST.json` records each
copied file's original project-relative path, byte size, and SHA-256 hash.
For the two files named `*.public.json`, it also records the original file's
size and hash and the exact text-only path transformation. Numeric values,
scientific settings, and source/dependency hashes are unchanged.

## Evidence map

| Release location | Final source | Purpose |
| --- | --- | --- |
| `results/native_comparison/vldb_framework_full_20261015_v1/` | Five-backend completed comparison | SZ3, ZFP, SPERR-1D, Serf-XOR, and Serf-QT native versus JointZip |
| `results/native_comparison/vldb_framework_sz2016_20261016_v1/` | Completed SZ2016 comparison | The same comparison and independent timing for SZ2016 |
| `results/ablation/vldb_framework_ablation_20261017_v1/` | Completed six-backend ablation | E, ER, EA, ERA, and ERA without correction; all prescribed settings |
| `results/acf_case/vldb_acf_forda_fordb_20261028_v1/` | Completed FordA/FordB case study | ER versus both declared ERA thresholds, two backends, both classification consumers, and mechanism diagnostics |
| `results/lossless_comparison/vldb_uniform_lossless_accounting_20261031_v1/` | Complete-file comparison with lossless Zstd | All 3,024 original ERA TEST conditions under a common metadata wrapper |
| `paper/data/` | Manuscript display data | Exact displayed aggregate inputs, large table, and companion tables; each source version is identified in the manifest |

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
The large table retains source and setting separately; its added Mean rows
average sources equally within each task/backend/setting before rounding.
These comprise eight displayed rows and 48 backend cells. Forecasting percentage
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

## FordA/FordB ACF case study

`results/acf_case/vldb_acf_forda_fordb_20261028_v1/` contains all three arms
(`ER`, `ERA_tau005`, and `ERA_tau001`) for SZ3 and ZFP. FordA has 3,601 TRAIN
and 1,320 TEST records; FordB has 3,636 TRAIN and 810 TEST records. Each record
is a complete 500-point window. Every official TEST record is included. The
datasets were selected before TEST evaluation using the low-frequency structure
discussed in Section 3.4 of [Lubba et al., catch22](https://doi.org/10.1007/s10618-019-00647-x).

The pointwise tolerance is 1.0 and minimum source-reconstruction correlation is
0.99. ER disables the ACF gate; ERA uses maximum ACF distortion 0.05 or 0.01.
The common maximum lag, 16, was selected using TRAIN-only coverage of the first
traditional ACF minimum and its right neighbor. The original backend search
budgets and correction policy are retained. The primary consumer uses all 22
pycatch22 0.4.5 features. The complementary consumer uses 32 non-DC frequency
bands. Both use median imputation, standardization, and the same fixed
LogisticRegression configuration fitted on raw TRAIN only. A classifier using
only the first ACF minimum is a mechanism diagnostic. Full parameters, versions,
input/model hashes, and native binary identities appear in `PROTOCOL.public.json`.

| File | Coverage and interpretation |
| --- | --- |
| `downstream.csv` | All 42 dataset/backend/arm/consumer records, including Raw; accuracy, balanced accuracy, AUROC, full bytes, complete CR, and prediction flips |
| `contrasts.csv` | All 24 paired ER-to-ERA contrasts, including the mechanism-only consumer; changes in percentage points, descriptive intervals, and improved/tied/declined counts |
| `mechanism.csv` | All 12 stored streams; actual E/R/ACF, enabled-constraint pass counts, complete CR, exact fallback, correction points/bytes, and traditional-ACF/first-minimum errors |
| `artifacts/*_windows.csv` | All 12,780 per-window mechanism and constraint records, in 12 files; these are numeric diagnostics, not reconstructed time series |
| `RESULTS.json` | The complete downstream, contrast, and mechanism records together |
| `TEST_BINDING.json`, `TRAIN_PREPARATION.json` | Frozen TEST identity and TRAIN-only lag-selection evidence |
| `COMPLETE.json`, `audit/FINAL_RESULT_AUDIT.json` | Saved completion and independent verification evidence |

Accuracy and balanced accuracy in `downstream.csv` are fractions, AUROC is on
0--1, and contrast changes and confidence limits are in percentage points.
The intervals are class-stratified paired bootstrap intervals with 5,000
replicates on the fixed TEST sets. They are descriptive and have no
multiple-comparison correction. All enabled numerical constraints pass for all
12,780 windows. The saved independent audit compares 39,018 quantities, with a
maximum absolute difference of approximately 1.33e-14.

For the primary catch22 consumer, tightening A from disabled to 0.01 improves
balanced accuracy by 3.354 and 20.500 percentage points on FordA for SZ3 and
ZFP, and by 3.218 and 8.111 points on FordB. The 0.05 setting ties ER in balanced
accuracy. Across both classification consumers and both A thresholds there are
eight improved, eight tied, and zero declined aggregate comparisons. All
mechanism-only outcomes are also retained, including the small negative
FordA/SZ3 first-minimum result. The FordB/SZ3 frequency-band interval includes
zero. Stronger A constraints reduce CR in the 0.01 cases; equal requested E/R
thresholds do not imply equal actual E/R or matched CR.

The two feature representations are related, so this is not statistically
independent spectral validation. The traditional global-mean ACF used for
feature diagnostics differs from the slice-Pearson ACF contract. The case study
does not establish a guarantee for arbitrary unknown downstream tasks, and
cache/parallel timings are not codec speed measurements.

`PROTOCOL.public.json` replaces private host path prefixes with
`{experiment-storage}/` and `{project-root}/` placeholders only. Its recorded
model, binary, source, and runtime-file hashes identify the originals; those
files are not all redistributed. The protocol hash in `COMPLETE.json` and the
saved audit is the **original frozen protocol hash**, listed as `source_sha256`
in `results/SOURCE_MANIFEST.json`, not the hash of the public path-normalized
copy. Completion also refers to historical local recovery records that are
not part of this archive. Raw inputs, predictions, reconstructed signals,
trained models, execution logs, and caches are omitted.

## Complete-file comparison with lossless Zstd

`results/lossless_comparison/vldb_uniform_lossless_accounting_20261031_v1/`
compares all 172 original TEST inputs and all 3,024 ERA JointZip configurations
against saved lossless Zstd output. It does not include the later Ford case
study. Classification and retrieval share the UCR compressed files, so this
comparison has three compression task groups: forecasting, UCR, and anomaly.

Both sides use a JCF1 wrapper with byte-identical per-input metadata, including
dtype, shape, true count, normalization mean/std, channel, and version. Each
method's own dispatch, window/record boundaries, encoded payload, and correction
records remain fully billed. Provenance hashes are external for both methods.
The lossless reference is the saved windowed Zstd level-19 collection with
its existing batch closure, rather than JointZip's internal Zstd level-3 exact
candidate. The original payloads are preserved; no scientific encoding,
consumer evaluation, or speed experiment was rerun.

| File | Coverage and interpretation |
| --- | --- |
| `conditions.csv` | All 3,024 paired conditions, including full bytes, CR, ratio, byte savings, outcome, payload identities, and artifact hashes |
| `macros.csv` | All 18 backend/task macro groups, with source-balanced geometric means |
| `settings.csv` | Every backend/task/setting group, retaining strict, medium, loose, and UCR stress |
| `sources.csv` | All 84 backend/source groups |
| `SUMMARY.json`, `COMPLETE.json` | Complete coverage, aggregate ranges, verification counts, and an unfavorable example |
| `PROTOCOL.json` | Exact scope, common accounting, lossless reference, and aggregation policy |
| `INDEPENDENT_REVIEW.json`, `ROSTER_IDENTITY_REVIEW.public.json` | Saved complete-file, metadata, roster, backend identity, and aggregation checks |
| `source_bindings.json`, `lossless_files.json`, `channel_mapping.json` | Original artifact/input identities, the 172 lossless-file records, and historical/current channel mapping |
| `REPORT_MANIFEST.json` | Original producer/result hashes; references are to the original study tree |

Complete CR uses unpadded Float32 bytes divided by complete JCF1 file bytes.
Means are geometric within each source and then equally weighted across
sources. Forecasting and anomaly retain all three settings; UCR uses stress.
The 18 backend/task CR ratios are all above one, ranging from 1.521 to 20.136.
This range is a range of macro ratios, not a per-file range or a ratio of total
storage bytes. Across individual configurations, JointZip CR is higher in
2,500, tied in zero, and lower in 524. The lower cases comprise 343 SZ3
forecasting conditions and 181 anomaly conditions. Of 84 backend/source groups,
83 favor JointZip; SZ3/Solar is the exception. These related configurations are
not independent statistical samples.

All 172 lossless files were independently decoded bit-exact. All 3,024 JointZip
payloads and original-file reconstruction hashes were verified unchanged, using
the existing independently decoded scientific evidence; the original native
decoders were not all rerun for this accounting study. The public archive
contains the recorded numeric and verification evidence, not the 3,196 encoded
files. Artifact paths in the CSV/JSON identify files in the original study tree,
not additional downloads in this repository. `ROSTER_IDENTITY_REVIEW.public.json`
changes only a private absolute-path description. `REPORT_MANIFEST.json` retains
the hash of the original review, which is also recorded as `source_sha256` in
the release manifest.

Window-level selection against an internal exact candidate does not guarantee
that the final JointZip stream is smaller than a separately encoded lossless
file. This study changes common framing and accounting only; it does not add a
whole-file fallback. The 524 lower-CR conditions must not be confused with the
581 conditions from an earlier payload-only accounting comparison.

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
