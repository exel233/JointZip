"""Reproduce the paper source/setting table from finalized results, without experiments.

Portable copy of the 2026-10-21 table builder; only input/output paths are changed.
The original numerical aggregation and coverage assertions are retained.

Quality CSVs already average prescribed seeds. Average inputs within each
source/model/setting, then models equally. Preserve every source and setting.
CR is read once per compression artifact, geometrically averaged within source,
and then geometrically averaged over sources for the seven displayed CR rows.
"""
from collections import defaultdict, Counter
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'paper/data'
OUT = ROOT / 'generated'
FINAL_DIRS = [ROOT / 'results/native_comparison/vldb_framework_full_20261015_v1',
              ROOT / 'results/native_comparison/vldb_framework_sz2016_20261016_v1']
BACKENDS = ['sz3', 'zfp', 'sperr', 'serf_xor', 'serf_qt', 'sz2016']
BACKEND_LABELS = ['SZ3', 'ZFP', 'SPERR-1D', 'Serf-XOR', 'Serf-QT', 'SZ2016']
POINTS = ['strict', 'medium', 'loose']
SPECS = [
    dict(task='forecast', metric='mse', label=r'Forecasting: MSE $\downarrow$',
         sources=['ETTm1', 'ETTm2', 'Solar', 'Weather', 'ElecDem', 'Wind'], points=POINTS,
         models=['ridge', 'dlinear', 'patchtst', 'chronos2', 'chronos_bolt_small', 'timesfm3'],
         scale=1, decimals=4, unit='MSE', cr_task='forecast'),
    dict(task='classification', metric='accuracy', label=r'Classification: accuracy (\%) $\uparrow$',
         sources=['ECG200', 'ArrowHead', 'GunPoint', 'Earthquakes', 'ShapeletSim', 'Meat'], points=['stress'],
         models=['one_nn', 'logistic', 'resnet_v1', 'resnet_v2'],
         scale=100, decimals=2, unit='percent', cr_task='ucr'),
    dict(task='retrieval', metric='AP', label=r'Retrieval: mAP (\%) $\uparrow$; same UCR storage as above',
         sources=['ECG200', 'ArrowHead', 'GunPoint', 'Earthquakes', 'ShapeletSim', 'Meat'], points=['stress'],
         models=['shape', 'resnet_v1', 'resnet_v2'],
         scale=100, decimals=2, unit='percent', cr_task='ucr'),
    dict(task='anomaly', metric='VUS_PR', label=r'Anomaly detection: VUS-PR $\uparrow$',
         sources=['MSL', 'SMAP'], points=POINTS,
         models=['ridge_svd64', 'knn', 'lstm'],
         scale=1, decimals=4, unit='VUS-PR', cr_task='anomaly'),
]
KEYS = ['task', 'backend', 'model', 'dataset', 'point']


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values)


def geomean(values):
    values = list(values)
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(mean(math.log(v) for v in values))


def value_text(value, spec):
    return f"{value * spec['scale']:.{spec['decimals']}f}"


def winner_at_display_precision(n, j, spec):
    n, j = float(value_text(n, spec)), float(value_text(j, spec))
    if n == j:
        return 'tie'
    native_better = n < j if spec['task'] == 'forecast' else n > j
    return 'native' if native_better else 'jointzip'


metrics = {s['task']: s['metric'] for s in SPECS}
source_info = []
grouped = defaultdict(list)
all_pair_keys = []
raw_checks = defaultdict(list)
for directory in FINAL_DIRS:
    path = directory / 'downstream_pairs.csv'
    rows = read_csv(path)
    selected = [r for r in rows if r['split'] == 'test' and r['metric'] == metrics.get(r['task'])]
    source_info.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            rows=len(rows), selected_test_pairs=len(selected)))
    for row in selected:
        all_pair_keys.append(tuple(row[k] for k in ['task', 'backend', 'model', 'sid', 'dataset', 'point', 'metric']))
        grouped[tuple(row[k] for k in KEYS)].append(row)
        raw_checks[tuple(row[k] for k in ['task', 'model', 'dataset', 'sid'])].append(float(row['raw']))
assert len(all_pair_keys) == len(set(all_pair_keys)) == 16884
expected = set()
for spec in SPECS:
    expected.update((spec['task'], b, m, d, p) for b, m, d, p in
                    itertools.product(BACKENDS, spec['models'], spec['sources'], spec['points']))
assert set(grouped) == expected and len(expected) == 1008

raw_max_spread = max(max(values) - min(values) for values in raw_checks.values())
assert raw_max_spread <= 1e-12, ('Raw changes across backend or setting', raw_max_spread)
cells = {}
for key, rows in grouped.items():
    ids = [r['sid'] for r in rows]
    assert len(ids) == len(set(ids))
    cells[key] = {name: mean(float(r[name]) for r in rows) for name in ['raw', 'native', 'jointzip']}
    cells[key]['inputs'] = len(rows)
    assert all(math.isfinite(cells[key][name]) for name in ['raw', 'native', 'jointzip'])

# Reconcile every source/model/setting cell against the retained atlas data.
atlas_path = DATA / 'downstream_gain_cells.csv'
atlas_rows = read_csv(atlas_path)
atlas = {tuple(r[k] for k in KEYS): r for r in atlas_rows}
assert len(atlas_rows) == len(atlas) == 1008 and set(atlas) == expected
atlas_max_error = 0.0
for key, cell in cells.items():
    n, j = cell['native'], cell['jointzip']
    gain = 100 * (1 - j / n) if key[0] == 'forecast' else 100 * (j - n)
    assert cell['inputs'] == int(atlas[key]['inputs'])
    for field, value in [('native', n), ('jointzip', j), ('gain', gain)]:
        error = abs(float(atlas[key][field]) - value)
        atlas_max_error = max(error, atlas_max_error)
        assert error < 1e-12, (key, field, error)

# Compression artifacts, rather than repeated consumer rows, determine CR.
cr_groups = defaultdict(list)
cr_ids = []
for directory in FINAL_DIRS:
    path = directory / 'cr_pairs.csv'
    rows = read_csv(path)
    selected = [r for r in rows if r['split'] == 'test']
    source_info.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                            rows=len(rows), selected_test_pairs=len(selected)))
    for row in selected:
        cr_ids.append(tuple(row[k] for k in ['task', 'backend', 'dataset', 'sid', 'point']))
        cr_groups[tuple(row[k] for k in ['task', 'backend', 'dataset', 'point'])].append(row)
assert len(cr_ids) == len(set(cr_ids)) == 3024
source_cr = {}
for key, rows in cr_groups.items():
    source_cr[key] = dict(native=geomean(float(r['native_cr']) for r in rows),
                          jointzip=geomean(float(r['jointzip_cr']) for r in rows), inputs=len(rows))
assert len(source_cr) == 180

quality = {}
quality_rows = []
for spec in SPECS:
    for dataset, point, backend in itertools.product(spec['sources'], spec['points'], BACKENDS):
        model_cells = [cells[(spec['task'], backend, model, dataset, point)] for model in spec['models']]
        assert len({cell['inputs'] for cell in model_cells}) == 1
        values = {name: mean(cell[name] for cell in model_cells) for name in ['raw', 'native', 'jointzip']}
        cr = source_cr[(spec['cr_task'], backend, dataset, point)]
        assert cr['inputs'] == model_cells[0]['inputs']
        values.update(inputs=model_cells[0]['inputs'], models=len(model_cells),
                      native_cr=cr['native'], jointzip_cr=cr['jointzip'])
        quality[(spec['task'], dataset, point, backend)] = values
        quality_rows.append(dict(row_kind='quality', task=spec['task'], dataset=dataset,
                                 point=point, backend=backend, metric=spec['metric'], unit=spec['unit'],
                                 **values, raw_display=value_text(values['raw'], spec),
                                 native_display=value_text(values['native'], spec),
                                 jointzip_display=value_text(values['jointzip'], spec),
                                 display_winner=winner_at_display_precision(values['native'], values['jointzip'], spec)))
assert len(quality_rows) == 216

cr_rows = []
cr_aggregate = {}
for spec in [SPECS[0], SPECS[1], SPECS[3]]:
    for point, backend in itertools.product(spec['points'], BACKENDS):
        parts = [source_cr[(spec['cr_task'], backend, dataset, point)] for dataset in spec['sources']]
        values = {name: geomean(p[name] for p in parts) for name in ['native', 'jointzip']}
        cr_aggregate[(spec['cr_task'], point, backend)] = values
        cr_rows.append(dict(row_kind='source_balanced_cr', task=spec['cr_task'], dataset='all_sources_equal',
                            point=point, backend=backend, metric='complete_stream_CR', unit='ratio',
                            raw='', **values, native_cr=values['native'], jointzip_cr=values['jointzip'],
                            inputs=sum(p['inputs'] for p in parts), models=0, raw_display='--',
                            native_display=f"{values['native']:.2f}", jointzip_display=f"{values['jointzip']:.2f}",
                            display_winner='not_emphasized'))
assert len(cr_rows) == 42

lines = [r'% Generated by analysis/build_large_table.py; finalized TEST data only.',
         r'\begingroup', r'\setlength{\tabcolsep}{1.5pt}', r'\renewcommand{\arraystretch}{1.02}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llr*{6}{rr}@{}}', r'\toprule',
         ' & & & ' + ' & '.join(r'\multicolumn{2}{c}{' + label + '}' for label in BACKEND_LABELS) + r' \\',
         ''.join(r'\cmidrule(lr){' + f'{4 + i*2}-{5+i*2}' + '}' for i in range(6)),
         'Source & Setting & Raw & ' + ' & '.join(['N & J']*6) + r' \\', r'\midrule']
displayed_quality_rows = 0
displayed_cr_rows = 0
for spec_index, spec in enumerate(SPECS):
    if spec_index:
        lines.append(r'\midrule')
    lines.append(r'\multicolumn{15}{@{}l}{\textbf{' + spec['label'] + r'}} \\')
    for dataset in spec['sources']:
        for pi, point in enumerate(spec['points']):
            rows = [quality[(spec['task'], dataset, point, b)] for b in BACKENDS]
            raw_values = [r['raw'] for r in rows]
            assert max(raw_values)-min(raw_values) <= 1e-12
            columns = [dataset if pi == 0 else '', point.capitalize(), value_text(raw_values[0], spec)]
            for row in rows:
                winner = winner_at_display_precision(row['native'], row['jointzip'], spec)
                for arm in ['native', 'jointzip']:
                    txt = value_text(row[arm], spec)
                    columns.append(r'\textbf{' + txt + '}' if winner == arm else txt)
            lines.append(' & '.join(columns) + r' \\')
            displayed_quality_rows += 1
    if spec['task'] != 'retrieval':
        lines.append(r'\addlinespace[2pt]')
        for pi, point in enumerate(spec['points']):
            columns = [r'\textit{CR (geo.)}' if pi == 0 else '', point.capitalize(), '--']
            for backend in BACKENDS:
                row = cr_aggregate[(spec['cr_task'], point, backend)]
                columns.extend([f"{row['native']:.2f}", f"{row['jointzip']:.2f}"])
            lines.append(' & '.join(columns) + r' \\')
            displayed_cr_rows += 1
lines.extend([r'\bottomrule', r'\end{tabular*}', r'\endgroup', ''])
assert displayed_quality_rows == 36 and displayed_cr_rows == 7
OUT.mkdir(parents=True, exist_ok=True)
tex_path = OUT / 'downstream_large_table.tex'
tex_path.write_text('\n'.join(lines), encoding='utf-8')
csv_path = OUT / 'downstream_large_table.csv'
fields = ['row_kind', 'task', 'dataset', 'point', 'backend', 'metric', 'unit', 'raw', 'native', 'jointzip',
          'native_cr', 'jointzip_cr', 'inputs', 'models', 'raw_display', 'native_display', 'jointzip_display', 'display_winner']
with csv_path.open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(quality_rows + cr_rows)

# Setting examples for manuscript interpretation, all computed after the stated
# model/source aggregation; they are not means of percentage-gain cells.
examples = []
for task, dataset, point, backend in [
    ('forecast', 'ETTm1', 'strict', 'serf_qt'), ('forecast', 'ETTm1', 'loose', 'serf_qt'),
    ('forecast', 'ETTm1', 'strict', 'serf_xor'), ('forecast', 'ETTm1', 'loose', 'serf_xor'),
    ('classification', 'ArrowHead', 'stress', 'sz2016'),
    ('anomaly', 'MSL', 'medium', 'serf_qt'), ('anomaly', 'SMAP', 'medium', 'serf_qt'),
    ('anomaly', 'MSL', 'loose', 'sperr'), ('anomaly', 'SMAP', 'loose', 'sperr')]:
    row = quality[(task, dataset, point, backend)]
    examples.append(dict(task=task, dataset=dataset, point=point, backend=backend, **row,
                         gain=(100*(1-row['jointzip']/row['native']) if task == 'forecast'
                               else 100*(row['jointzip']-row['native']))))

# The detailed table must reproduce the existing manuscript's aggregate claims.
overview_path = DATA / 'backend_overview.csv'
overview = {r['backend']: r for r in read_csv(overview_path)}
macro_max_error = 0.0
macro_values_checked = 0
for backend in BACKENDS:
    for spec in SPECS:
        rows = [quality[(spec['task'], d, p, backend)] for d, p in
                itertools.product(spec['sources'], spec['points'])]
        native = mean(r['native'] for r in rows)
        jointzip = mean(r['jointzip'] for r in rows)
        if spec['task'] == 'forecast':
            column, expected_value = 'forecast_mse_relative_percent', 100 * (jointzip / native - 1)
        else:
            column, expected_value = spec['task'] + '_pp', 100 * (jointzip - native)
        error = abs(float(overview[backend][column]) - expected_value)
        macro_max_error = max(macro_max_error, error)
        macro_values_checked += 1
        assert error < 1e-10, (backend, column, error)
    for spec in [SPECS[0], SPECS[1], SPECS[3]]:
        prefix = spec['cr_task']
        for arm in ['native', 'jointzip']:
            value = geomean(cr_aggregate[(prefix, p, backend)][arm] for p in spec['points'])
            error = abs(float(overview[backend][prefix + '_' + arm]) - value)
            macro_max_error = max(macro_max_error, error)
            macro_values_checked += 1
            assert error < 1e-10, (backend, prefix, arm, error)

audit = dict(scientific_run=False, operation='Read finalized results and author a numeric table only',
             sources=source_info, atlas_csv_sha256=sha(atlas_path), atlas_cells_checked=1008,
             atlas_max_absolute_error=atlas_max_error, selected_quality_input_pairs=16884,
             selected_compression_artifact_pairs=3024, raw_invariant_checks=len(raw_checks),
             raw_max_spread_across_backend_and_setting=raw_max_spread,
             quality_source_setting_rows=36, quality_backend_cells=216,
             displayed_cr_rows=7, cr_backend_cells=42, displayed_data_rows=43,
             displayed_paired_cells=258, displayed_native_jointzip_values=516,
             source_cr_cells_in_csv=216, missing_or_unexpected_cells=0,
             existing_macro_values_checked=macro_values_checked,
             existing_macro_max_absolute_error=macro_max_error,
             existing_macro_overview_sha256=sha(overview_path),
             quality_aggregation='Already seed-averaged input pairs -> equal inputs within source/model/setting -> equal models; preserve source and setting.',
             cr_aggregation='One TEST compression artifact pair per sid/backend/setting -> geometric mean within source -> geometric mean over sources equally.',
             raw_invariance='Raw checked independently per task/model/source/input across all backends/settings before aggregation.',
             shared_ucr='Classification and retrieval use exactly the same CR source cells; the single UCR CR row serves both.',
             units={'forecast':'absolute MSE, four decimals', 'classification':'accuracy percent, two decimals',
                    'retrieval':'mAP percent, two decimals', 'anomaly':'absolute VUS-PR, four decimals',
                    'CR':'complete-stream ratio, two decimals'},
             emphasis='Better Native or JointZip within each quality pair only; ties at display precision unbolded; CR and Raw unbolded.',
             display_quality_outcomes=dict(Counter(r['display_winner'] for r in quality_rows)),
             specifications=SPECS, examples=examples,
             outputs={p.name:dict(sha256=sha(p), bytes=p.stat().st_size) for p in [tex_path, csv_path]})
(OUT / 'generation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps({key:audit[key] for key in ['displayed_data_rows', 'displayed_paired_cells',
       'atlas_cells_checked', 'atlas_max_absolute_error', 'raw_max_spread_across_backend_and_setting',
       'display_quality_outcomes', 'examples', 'outputs']}, indent=2))
