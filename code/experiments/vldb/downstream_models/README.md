# Independent DLinear task evaluation

This directory runs the unmodified DLinear architecture from the [EDBT 2024 official artifact](https://github.com/cmcuza/EvalImpLSTS/blob/61392420bb39f42eaa3a53f308e1e454ae8e39d6/forecasting/DLinear/components/model.py), commit `61392420bb39f42eaa3a53f308e1e454ae8e39d6`. The exact source SHA-256 is `06659056434eedf125b0740073c2c35a80830db1a9a58281d95d6d161f8abb64`. The script loads that local file directly and rejects a changed hash.

It evaluates a new protocol on the six authors' data sources. This is not a strict replication of the entire paper. The data/normalization/split/window decisions are shared with `main_benchmark/tasks_data.py`: ETT uses the artifact's fixed 12/4/4 month split, other sources 70/10/20; TRAIN-only float32 standardization; 128-point compression windows; the predictor sees the final 96 points and predicts the untouched following 24. All 137 Solar variates remain in the evaluation, using one shared-parameter DLinear model trained jointly, rather than 137 separately fitted models. The paper's longer Solar context and complete model panel are not reproduced here.

Training uses every contiguous TRAIN-only 96+24 pair, Adam, batch size 32, up to 10 epochs, patience 3, and the official type1 learning-rate schedule. We retain a final incomplete batch (`drop_last=False`) and select the earliest minimum raw-VAL MSE epoch over the eight fixed validation origins. These two details are declared changes from upstream behavior. The same fixed seed is used in each source group. Neither compression output nor TEST score participates in training or model selection. The model training function has no TEST argument. Whole batches may include history at later origins, so this selected-window archival experiment is not a claim of online causal streaming.

`freeze` binds the model source, data helper, this evaluator, runtime, configuration, source hashes, split identities, normalization and selected origins. `train` saves NPZ numeric parameter arrays (no pickle) and JSON provenance, with complete checkpoint SHA-256. `evaluate` binds to a single compression-run freeze, verifies the same source/label arrays, checks persisted SCS2 bytes and NOR2, decodes its payload, and runs the fixed model. A failed compression row stays present as `upstream_failed` or `upstream_unavailable`. Task results preserve per-origin errors and both raw/reconstructed predictions. Existing output rows are only reused if their source compression-row hash matches.

Run in the dedicated Linux/WSL environment from the repository root:

```bash
python3 -m experiments.vldb.downstream_models.dlinear smoke
python3 -m experiments.vldb.downstream_models.dlinear freeze --device cuda
python3 -m experiments.vldb.downstream_models.dlinear train
python3 -m experiments.vldb.downstream_models.dlinear evaluate --main-run experiments/results/vldb_edbt_main_20260907_v1
python3 -m experiments.vldb.downstream_models.dlinear train-watch --main-run experiments/results/vldb_edbt_main_20260907_v1 --max-hours 48
```

`--run` chooses the independent DLinear output directory; `--datasets ETTm1 Solar` selects work from the already frozen complete six-source protocol. Selecting execution order does not change the declared dataset universe. Rerun `evaluate` as the compression job produces additional final rows. All model/data source hashes must remain unchanged once frozen. An interrupted training attempt that left a checkpoint without its matching JSON requires a new output run rather than silently replacing its bytes.

`train-watch` resumes or trains the full six-model set, then polls completed compression rows every 30 seconds. It performs a final drain and exits after observing the main `FINISHED.json`, or exits with `WATCH_TIMEOUT.json` after 48 hours. Missing models and unexpected failures produce `FAILED.json`; the evaluator cannot silently run forever. `RUNNING.json` records its Linux PID; `TRAIN_STATUS.json` and `WATCH_STATUS.json` distinguish actual progress from future work. The standalone `watch` command starts the bounded follower after models already exist.

The predictor cost and checkpoint bytes describe downstream evaluation and are not codec payload bytes: this is a shared raw-trained consumer for every method. The paper must retain raw absolute model quality alongside degradation, and aggregate Solar variates within Solar before the six-source macro result. ACF/Pearson constraints do not imply an arbitrary DLinear forecast guarantee, especially because the predictor consumes a 96-point suffix of the 128-point protected window.
