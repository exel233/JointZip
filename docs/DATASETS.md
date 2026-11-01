# Data, model sources and preprocessing

The release does not redistribute raw datasets, upstream model repositories or
model weights. Obtain them from the cited sources under their original terms.
The paths below are relative to `code/`. They match the retained source layout;
the original private machine paths have been replaced as described in
[CODE.md](CODE.md).

`protocols/frozen_parameters.json` records the complete main input roster,
prepared-input identities, thresholds, Ford archive hashes and foundation-model
revisions. Its `origins` field identifies the recorded freezes from which the
reference was extracted. It does not contain raw values or fitted weights.

## Main forecasting panel

The data are the authors' files in [EvalImpLSTS at revision
61392420bb39f42eaa3a53f308e1e454ae8e39d6](https://github.com/cmcuza/EvalImpLSTS/tree/61392420bb39f42eaa3a53f308e1e454ae8e39d6).
Place that checkout at `tmp/upstreams/EvalImpLSTS-audit`; the loader reads its
`data/raw/` directory. Use the artifact's actual Parquet files, not a similarly
named dataset from another distribution.

| Source | File under `data/raw/` | Retained value columns | File rows |
| --- | --- | --- | ---: |
| ETTm1 | `ETTm1/ETTm1.parquet` | `OT` | 69,680 |
| ETTm2 | `ETTm2/ETTm2.parquet` | `OT` | 69,680 |
| Solar | `Solar/solar.parquet` | All 137 non-time columns, in file order | 52,560 |
| Weather | `Weather/weather.parquet` | `OT` | 52,704 |
| ElecDem | `AUSElecDem/aus_electrical_demand.parquet` | `y` | 230,736 |
| Wind | `Wind/wind.parquet` | `active power` | 432,000 |

The loader is `experiments/vldb/main_benchmark/tasks_data.py`. It verifies file
structure, expected row counts, timestamps and finite values, and records file
and array hashes. There are 142 series: five single-channel sources plus Solar's
137 channels. Aggregation averages series within a source before giving the six
sources equal weight.

For ETT, the TRAIN/VAL/TEST boundaries are 0/34,560/46,080/57,600. The remaining
file rows are outside this selected benchmark split. Other sources use the
loader's 70/10/20 chronological split, with integer rounding as implemented.
Each series uses a mean and scale computed from Raw TRAIN only. Arithmetic
produces Float32 normalized inputs; nonfinite values are rejected rather than
silently imputed. Eight VAL and 32 TEST origins per series are chosen by length
alone. Selected 128-point input histories and their following 24-point futures
have disjoint supports within each split. The consumer sees the final 96
history points and predicts the unchanged 24-point future.

`main_benchmark/benchmark.py` and `run_driver.py` are the historical setup and
execution entry points. Later drivers copy their prepared inputs through the
recorded parent freezes. Raw data alone do not replace those historical input,
model and codec-result bindings.

## UCR classification and retrieval

Use the [UCR/UEA archive](https://www.timeseriesclassification.com/) distribution
at `https://timeseriesclassification.com/aeon-toolkit/<Name>.zip` for these six
datasets: **ECG200, ArrowHead, GunPoint, Earthquakes, ShapeletSim and Meat**.
Keep the official complete TRAIN and TEST files and class labels. The download
entry point is:

```sh
python -m experiments.vldb.external_utility.ucr_classification download
```

It stores archives under `tmp/external_utility/ucr/`. The original panel's
metadata selection and file-hash checks are in `external_utility/protocol.py`
and `ucr_classification.py`. The downstream preparation in
`downstream_expansion/classification/prepare.py` checks the previously frozen
panel and artifacts before constructing its arrays; it is not a standalone
replacement for those historical freezes.

Codec normalization is fitted on Raw TRAIN. Classification and retrieval use
the same held-fixed Raw TRAIN reference. The logistic and ResNet v1 validation
split is stratified within official TRAIN, with seed 20260915 and
`max(1, ceil(0.2 * class_count))` validation examples per class. ResNet v2 uses
all official TRAIN, per-query normalization and training-loss checkpoint
selection. These are separate, retained protocols. Test labels are used for
evaluation, not hyperparameter selection.

Shape retrieval uses centered unit-L2 series distance. Deep retrieval uses
embeddings from the corresponding fixed ResNet checkpoints. The reference
dictionary, labels and model weights are downstream resources, not extra bytes
in an individual compressed query's file.

## NASA anomaly panel

The source is [TSB-AD at revision
6beac72e11d1155ade40870492c00d0d1cfdcaaf](https://github.com/TheDatumOrg/TSB-AD/tree/6beac72e11d1155ade40870492c00d0d1cfdcaaf).
Obtain the univariate archive from the official [TSB-AD-U dataset
download](https://www.thedatum.org/datasets/TSB-AD-U.zip).
The expected code checkout is `tmp/upstreams/TSB-AD-20260915`, with CSV data
under `tmp/external_utility/tsb_ad/TSB-AD-U`.

Membership is **all 24 MSL/SMAP files** in the pinned
`Datasets/File_List/TSB-AD-U-Eva.csv`, selected by `_MSL_` or `_SMAP_` in the
filename. This is the NASA portion of the official Eval list, not all of TSB-AD
or an arbitrary selection from it. Exact file identities are also in the
published 172-input roster. TSB-AD's curated files are not interchangeable with
an unmodified Telemanom download.

`downstream_expansion/anomaly/run.py` parses the labeled TEST start from each
filename's `_tr_<length>_` field. The first 80% of that normal prefix is the
optimization TRAIN portion and the remaining 20% is chronological validation.
Scaling is fit only on optimization TRAIN. The TEST suffix is protected in
128-point blocks, with edge padding for the last block. Task scoring trims that
padding. Main JointZip JZF1 files store the true count; earlier TSL1/SCS2 files
remain historical dependencies and are not the current main format.

The detector context is 150. Ridge uses the retained float64 SVD implementation
in `anomaly/linear_stability.py`; KNN compares with a frozen Raw TRAIN
subsequence dictionary. LSTM uses the official pinned TSB-AD `LSTMModel`
architecture and raw-VAL checkpoint selection. The feature/score consumer is
fixed across reconstructed TEST inputs. Official VUS metrics come from the
pinned TSB-AD checkout. The period selector for metric tolerance uses Raw TRAIN
and a traditional ACF, which is distinct from the codec's slice-Pearson ACF.

## FordA/FordB ACF supplement

Download the official [FordA archive](https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip)
and [FordB archive](https://www.timeseriesclassification.com/aeon-toolkit/FordB.zip)
to `experiments/results/vldb_acf_forda_fordb_20261028_v1/downloads/`, preserving
the names `FordA.zip` and `FordB.zip`.

| Dataset | TRAIN | TEST | Points per record | Recorded archive SHA256 |
| --- | ---: | ---: | ---: | --- |
| FordA | 3,601 | 1,320 | 500 | `4d78a72e5165e8da40c51115a8219c9d01978aafac6334ed8dfabd29da359ffb` |
| FordB | 3,636 | 810 | 500 | `a73810d108de24bae741de3b67a5f73286f9913da47d3dfbf73ce55b9386fff8` |

Keep all records, labels and the official normalization; the preparation casts
values to Float32 without fitting another dataset-wide scaler. On TRAIN only,
choose the smallest common L in {16, 32, 64, 128} covering the traditional ACF
first local minimum and its right neighbor in at least 99% of each dataset.
The recorded choice is L = 16. The classification pipelines' feature imputer
and StandardScaler are subsequently fitted on Raw TRAIN features, not on TEST
series or reconstructed values.

Dataset selection was motivated by Section 3.4 of [Lubba et al.,
*catch22*](https://doi.org/10.1007/s10618-019-00647-x). The actual primary consumer
uses all 22 features from **pycatch22 0.4.5**, not only the feature discussed in
that motivation. The complementary spectrum consumer and first-minimum
diagnostic are separately identified in `EXPERIMENTS.md`.

## Model sources and training entry points

| Consumer | Pinned source or acquisition entry | Actual use |
| --- | --- | --- |
| Ridge forecast | `main_benchmark/tasks_data.py` | Raw TRAIN fitting, Raw VAL order/regularization selection; original weights reused. |
| DLinear | [EvalImpLSTS model](https://github.com/cmcuza/EvalImpLSTS/blob/61392420bb39f42eaa3a53f308e1e454ae8e39d6/forecasting/DLinear/components/model.py) | `downstream_models/dlinear.py`; official architecture with declared short-context training protocol. Source SHA256 `06659056434eedf125b0740073c2c35a80830db1a9a58281d95d6d161f8abb64`. |
| PatchTST | [Official repository, revision 204c21efe0b39603ad6e2ca640ef5896646ab1a9](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | Checkout `tmp/upstreams/PatchTST-20260915`; `downstream_expansion/forecasting/runner.py` loads `PatchTST_supervised/models/PatchTST.py`. |
| ResNet v1/v2 | [hfawaz reference, revision e0233efd886df8c6ca18e6f1b545d15aaf423627](https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/resnet.py) | Explicit local PyTorch architecture port, not execution of the reference TensorFlow runtime; `classification/train.py` and `classification/standard_protocol_v2/train.py`. |
| NASA LSTM and metrics | Pinned TSB-AD checkout above | `downstream_expansion/anomaly/run.py`; local frozen-consumer protocol, not a TSB-AD leaderboard rerun. |
| Ford feature classifiers | `acf_forda_fordb_20261028/pipeline.py` | pycatch22/spectrum features and fixed logistic regression fit on full Raw TRAIN. |

DLinear uses context/horizon 96/24, kernel 25, shared weights, up to 10 epochs,
batch 32, Adam at 0.001, patience 3 and seed 20260907. PatchTST uses three
transformer layers, four heads, model width 16, patch length 16 and stride 8,
up to 20 epochs, batch 128, learning rate 0.0001 and patience 5. Its TRAIN pair
cap is 32,768 per source, allocated across every channel. Solar remains fully
represented; it is not reduced to one convenient channel.

Both ResNet versions use 64/128/128 channels and 8/5/3 kernels. V1 selects on
Raw TRAIN validation, then refits. V2 uses 1,500 epochs, training-loss selection
and its declared per-query normalization. The common three seeds are 20260915,
20260916 and 20260917. Full settings, normalization rules and explicit deviations
from reference training are preserved in the corresponding source dictionaries.
These training entry points explain the weights used by the later consumers;
retraining can require the earlier prepared inputs and will create new model
hashes rather than reproduce a missing checkpoint by declaration.

## Pretrained forecasting models

The downloader `downstream_expansion/foundation/prepare_models.py` pins these
Hugging Face revisions and records each downloaded file hash:

| Model | Revision | Recorded weight license |
| --- | --- | --- |
| [amazon/chronos-2](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c) | `29ec3766d36d6f73f0696f85560a422f50e8498c` | Apache-2.0 |
| [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small/tree/772f3d25d38aec6d914c8949dab4462e2d46f5d8) | `772f3d25d38aec6d914c8949dab4462e2d46f5d8` | Apache-2.0 |
| [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2) | `43046b85ec22d584a13f8098c2ed39c889e129c2` | timesfm-non-commercial-license-v1.0 |

Use the license and model card at the pinned revision. These weights are not
redistributed here. The downloader requires network access and the separate
foundation environment. The models are inference-only, Float32, median forecasts
with no fine-tuning or cross-series learning. The wrapper uses context 96,
horizon 24 and fixed internal batch size 128. Historical input/runtime manifests
are still required for the paper's hash-bound consumer reuse.

## Recorded Python environments

The compact root `requirements.txt` supports the offline check and analysis. It
is not a lockfile for all native/model workloads. Historical science used Linux
Python 3.10.12 and NumPy 1.26.4. Ford's recorded feature environment adds SciPy
1.14.1, scikit-learn 1.5.1 and pycatch22 0.4.5.

The recorded foundation manifest lists torch 2.4.0, NumPy 1.26.4,
chronos-forecasting 2.3.2, timesfm 3.0.2, transformers 4.44.1,
huggingface-hub 0.36.2, safetensors 0.8.0, accelerate 1.15.0 and einops 0.8.2.
GPU runtime identity, CUDA determinism, batch settings and checkpoint hashes are
part of the original consumers' checks. Installing these Python versions alone
does not establish equivalence to that complete runtime. No model download or
scientific rerun was performed to validate the expanded public source snapshot.

Third-party source, datasets and weights retain their upstream terms. The local
ports preserve their source attribution. See the repository
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for codec dependencies and
additional notices.
