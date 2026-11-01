"""Render finalized four-task E/ER/EA/ERA macro averages; no experiments.

The input is the delivered downstream overview CSV. Its seed, source, model,
and setting aggregation is preserved exactly. Only units and (for forecasting)
the per-backend E denominator change for display. All six backends and all four
arms appear. The small signed annotation row exposes near-tied ERA/ER results.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'results/ablation/vldb_framework_ablation_20261017_v1/downstream_overview.csv'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check-only', action='store_true',
                    help='Validate all bars and render in memory without writing files.')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated',
                    help='Destination for PDF, PNG, and ablation_tasks_validation.json.')
parser.add_argument('--reference-csv', type=Path,
                    default=ROOT / 'paper/data/ablation_downstream_overview.csv',
                    help='Paper aggregate CSV used for an exact numerical cross-check.')
args = parser.parse_args()
OUT = args.output_dir
ORDER = ['sz3', 'zfp', 'sperr', 'serf_xor', 'serf_qt', 'sz2016']
LABELS = ['SZ3', 'ZFP', 'SPERR-\n1D', 'Serf-\nXOR', 'Serf-\nQT', 'SZ2016']
ARMS = ['E', 'ER', 'EA', 'ERA']
COLORS = ['#0072B2', '#009E73', '#D55E00', '#5B4BB7']
PANELS = [
    dict(task='forecast', metric='mse', title='(a) Forecasting',
         ylabel='MSE / E baseline (%)  ↓', limits=(0, 110), ticks=[0,25,50,75,100],
         baseline=0, unit='100 × MSE / E MSE within each backend', direction='lower'),
    dict(task='classification', metric='accuracy', title='(b) Classification',
         ylabel='Accuracy (%)  ↑', limits=(50, 80), ticks=[50,60,70,80],
         baseline=50, unit='accuracy percent', direction='higher'),
    dict(task='retrieval', metric='AP', title='(c) Retrieval',
         ylabel='Mean average precision (%)  ↑', limits=(50, 80), ticks=[50,60,70,80],
         baseline=50, unit='average precision percent', direction='higher'),
    dict(task='anomaly', metric='VUS_PR', title='(d) Anomaly detection',
         ylabel='VUS-PR (%)  ↑', limits=(25, 32), ticks=[25,27,29,31],
         baseline=25, unit='VUS-PR percent', direction='higher'),
]

frame = pd.read_csv(SOURCE)
reference = pd.read_csv(args.reference_csv)
pd.testing.assert_frame_equal(frame, reference, check_exact=True)
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':7.3,
    'axes.titlesize':8, 'axes.labelsize':7.4,
    'xtick.labelsize':6.6, 'ytick.labelsize':6.7,
    'axes.linewidth':.6, 'pdf.fonttype':42, 'ps.fonttype':42,
})
fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.85))
fig.subplots_adjust(left=.077, right=.99, bottom=.138, top=.855,
                    wspace=.28, hspace=.81)
x = np.arange(6)
width = .185
validation_panels = []

for ax, spec in zip(axes.flat, PANELS):
    data = frame[(frame.task == spec['task']) & (frame.metric == spec['metric'])]
    assert len(data) == 6 and not data.backend.duplicated().any()
    assert set(data.backend) == set(ORDER)
    data = data.set_index('backend').loc[ORDER]
    assert data.raw.nunique() == 1
    source_values = data[ARMS].to_numpy(dtype=float)
    assert source_values.shape == (6,4) and np.isfinite(source_values).all()
    if spec['task'] == 'forecast':
        assert (source_values[:,0] > 0).all()
        values = 100 * source_values / source_values[:,[0]]
        assert np.allclose(values[:,0], 100, rtol=0, atol=1e-12)
        reconstructed = values * source_values[:,[0]] / 100
        raw = None  # A common absolute Raw MSE is not a common normalized line.
    else:
        values = source_values * 100
        reconstructed = values / 100
        raw = float(data.raw.iloc[0] * 100)
    error = float(np.max(np.abs(reconstructed - source_values)))
    assert error < 1e-15
    assert values.min() > spec['baseline']
    assert values.max() < spec['limits'][1]
    delta = values[:,3] - values[:,1]
    for j, (arm, color) in enumerate(zip(ARMS, COLORS)):
        ax.bar(x+(j-1.5)*width, values[:,j]-spec['baseline'],
               bottom=spec['baseline'], width=width, color=color,
               edgecolor='white', linewidth=.35, zorder=3)
    ax.set_xlim(-.60, 5.60)
    ax.set_ylim(*spec['limits'])
    ax.set_yticks(spec['ticks'])
    ax.set_xticks(x, LABELS)
    ax.set_ylabel(spec['ylabel'], labelpad=3)
    ax.set_title(spec['title'], loc='left', pad=4, fontweight='medium')
    ax.tick_params(axis='x', length=0, pad=3)
    ax.tick_params(axis='y', length=2, width=.5, pad=2)
    ax.grid(axis='y', color='.87', linewidth=.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    if raw is not None:
        assert spec['limits'][0] < raw < spec['limits'][1]
        ax.axhline(raw, color='.25', linewidth=.7,
                   linestyle=(0,(4,3)), zorder=4)
        # Top-right labeling avoids covering bars in the two UCR panels.
        if spec['task'] != 'anomaly':
            ax.text(.99, raw+.40, f'Raw {raw:.2f}',
                    transform=ax.get_yaxis_transform(), ha='right',
                    va='bottom', fontsize=6.1, color='.25')
    if spec['baseline']:
        ax.text(.015,.96,f"Axis starts at {spec['baseline']}%",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=6.0, color='.25')
        for y in [-.010,.022]:
            ax.plot([-.010,.010],[y-.013,y+.013], transform=ax.transAxes,
                    color='white', linewidth=2.7, clip_on=False, zorder=6)
            ax.plot([-.010,.010],[y-.013,y+.013], transform=ax.transAxes,
                    color='.15', linewidth=.7, clip_on=False, zorder=7)

    # All values are retained at full precision in validation. Annotations are
    # signed differences in the plotted percentage units (percentage points).
    ax.text(-.59,-.34,'Δ',transform=ax.get_xaxis_transform(),
            ha='center',va='center',fontsize=6.1,color='.25')
    for position, value in zip(x,delta):
        assert round(float(value),3) != 0, 'Annotation would hide a nonzero sign'
        ax.text(position,-.34,f'{value:+.3f}',
                transform=ax.get_xaxis_transform(),ha='center',va='center',
                fontsize=6.1,color='.20')

    favorable = delta < 0 if spec['direction']=='lower' else delta > 0
    validation_panels.append({
        'task':spec['task'], 'source_metric':spec['metric'],
        'display_unit':spec['unit'], 'better':spec['direction'],
        'axis_limits':list(spec['limits']), 'bar_baseline':spec['baseline'],
        'truncated_axis_marked':bool(spec['baseline']), 'raw_line':raw,
        'raw_original':float(data.raw.iloc[0]),
        'source_values':dict(zip(ORDER, source_values.tolist())),
        'display_values':dict(zip(ORDER, values.tolist())),
        'ERA_minus_ER_display_pp':dict(zip(ORDER,delta.tolist())),
        'ERA_improves_ER':[b for b,v in zip(ORDER,favorable) if v],
        'ERA_worse_than_ER':[b for b,v in zip(ORDER,favorable) if not v],
        'ERA_improves_E_all_backends':bool(np.all(values[:,3]<values[:,0])
            if spec['direction']=='lower' else np.all(values[:,3]>values[:,0])),
        'roundtrip_max_abs_error':error, 'bars_below_axis':int((values<spec['baseline']).sum()),
        'bars_above_axis':int((values>spec['limits'][1]).sum()),
    })

handles = [Patch(facecolor=c,label=a) for a,c in zip(ARMS,COLORS)]
handles.append(Line2D([0],[0],color='.25',linewidth=.8,
                      linestyle=(0,(4,3)),label='Raw (b–d)'))
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.995),
           ncol=5,frameon=False,columnspacing=1.6,handlelength=1.4,
           handletextpad=.45,fontsize=7.2)
fig.text(.53,.015,'Δ: ERA − ER in plotted percentage points',
         ha='center',va='bottom',fontsize=6.8,color='.25')

# Draw even in validation mode so invalid plot configuration fails immediately.
fig.canvas.draw()
if not args.check_only:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'ablation_tasks.pdf',metadata={
        'Title':'Complete E/ER/EA/ERA ablation across four downstream tasks',
        'Creator':'reproduce_ablation_figure.py',
        'Subject':'All six backends and finalized source-balanced macro averages.'})
    fig.savefig(OUT/'ablation_tasks.png',dpi=250)
plt.close(fig)

validation = {
    'source':str(SOURCE.relative_to(ROOT)),
    'paper_reference_sha256':hashlib.sha256(args.reference_csv.read_bytes()).hexdigest(),
    'paper_aggregate_values_exact':True,
    'generated_files_written':not args.check_only,
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'scientific_run':'vldb_framework_ablation_20261017_v1',
    'operation':'Plot finalized aggregates only; no experiments or reaggregation.',
    'backends':ORDER, 'arms':ARMS, 'panel_count':4,
    'displayed_bars':96, 'omitted_bars':0,
    'aggregation':'Input CSV unchanged: seeds first, then equal source, model, and prescribed-setting macro aggregation.',
    'forecast_normalization':'Each backend uses its own E MSE denominator; E=100%. This compares arm effects, not absolute MSE across backends.',
    'raw_reference':'Absolute Raw lines in classification, retrieval, anomaly panels; omitted in normalized forecasting because each backend has a different E denominator.',
    'figure_inches':[7.05,3.85], 'font':'DejaVu Sans, 6.0–8 points',
    'panels':validation_panels,
}
if not args.check_only:
    (OUT/'ablation_tasks_validation.json').write_text(
        json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:validation[k] for k in [
    'displayed_bars','omitted_bars','source_sha256','figure_inches',
    'paper_aggregate_values_exact','generated_files_written']},indent=2))
