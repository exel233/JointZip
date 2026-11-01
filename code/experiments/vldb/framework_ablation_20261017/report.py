"""Full-roster E/R/A and fixed-candidate correction ablation reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
import statistics

from . import common as c

PRIMARY={'forecast':('mse',-1),'classification':('accuracy',1),
         'retrieval':('AP',1),'anomaly':('AP',1)}
CONTRASTS=(('add_R_to_E','E','ER'),('add_A_to_E','E','EA'),
           ('joint_vs_E','E','ERA'),('add_A_to_ER','ER','ERA'),
           ('add_R_to_EA','EA','ERA'),('correction','ERA_no_correction','ERA'))
TIE=1e-10


def method_parts(method):
    for arm in sorted(c.ARMS,key=len,reverse=True):
        suffix='_'+arm
        if method.endswith(suffix) and method[:-len(suffix)] in c.BACKENDS:
            return method[:-len(suffix)],arm
    return None,None


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _geo(values):
    return math.exp(statistics.fmean(math.log(x) for x in values)) if values else None


def _source(rows, field, geometric=False):
    grouped=defaultdict(list)
    for row in rows:
        value=row.get(field)
        if value is not None:
            grouped[row['dataset']].append(value)
    reducer=_geo if geometric else _mean
    return reducer([reducer(values) for values in grouped.values()])


def _macro_model_source(rows,field):
    """Each source within each model, then each model, receives equal weight."""
    groups=defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row['model']].append(row)
    return _mean([_source(values,field) for values in groups.values()])


def _result(gain):
    return 'improved' if gain>TIE else 'declined' if gain<-TIE else 'tied'


def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _fmt(value):
    return '—' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)


def _table(lines,headers,rows):
    lines.append('| '+' | '.join(headers)+' |')
    lines.append('| '+' | '.join('---' for _ in headers)+' |')
    lines.extend('| '+' | '.join(_fmt(value) for value in row)+' |' for row in rows)


def consumer_paths():
    return sorted([*(c.RUN/'evaluation/forecast/rows').glob('*/*/*/*/*.json'),
        *(c.RUN/'evaluation/ucr/classification').glob('*/*/*/*.json'),
        *(c.RUN/'evaluation/ucr/retrieval').glob('*/*/*/*.json'),
        *(c.RUN/'evaluation/anomaly/rows').glob('*/*/*.json')])


def compression_tables():
    records=[]; by_key={}; mechanisms=[]
    for path in sorted((c.RUN/'compression').glob('*/*/*/*.json')):
        row=c.read(path); backend,arm=method_parts(row.get('method',''))
        if backend is None:
            raise RuntimeError('Unknown method in compression report: '+str(path))
        key=(row['task'],row['sid'],row['split'],backend,row['point']['label'],arm)
        if key in by_key:
            raise RuntimeError('Duplicate compression report key: '+repr(key))
        strict=row['strict_audit']; numerical=row['numerical_audit']
        value=dict(task=row['task'],dataset=row['dataset'],sid=row['sid'],split=row['split'],
            backend=backend,arm=arm,point=row['point']['label'],status=row['status'],
            raw_bytes=row['raw_bytes'],artifact_bytes=row['artifact_bytes'],cr=row['cr'],
            enabled_contract=row['enabled_contract'],enabled_strict=row['enabled_pass_strict'],
            enabled_numerical=row['enabled_pass_numerical'],windows=strict['series_windows'],
            all_ERA_strict_pass=strict['joint_pass_count'],all_ERA_numerical_pass=numerical['joint_pass_count'],
            E_numerical_pass=numerical['point_pass_count'],R_numerical_pass=numerical['pearson_pass_count'],
            A_numerical_pass=numerical['acf_pass_count'],max_abs_error=numerical['max_abs_error'],
            pearson_min=numerical['pearson_min'],acf_max_abs=numerical['acf_max_abs'],
            reconstruction_mse=numerical['mse'],reuse_kind=row.get('reuse',{}).get('kind','new'),
            source_row=path.relative_to(c.RUN).as_posix())
        records.append(value); by_key[key]=value
        if row['status']!='complete' or row['split']!='test':
            continue
        metadata=c.read(c.RUN/row['metadata'])
        selected=[w['selected'] for w in metadata['windows']]
        if len(selected)!=value['windows'] or any(w['kind'] not in (0,1) for w in selected):
            raise RuntimeError('Selected-window mechanism coverage differs: '+str(path))
        mechanisms.append(dict(task=value['task'],dataset=value['dataset'],sid=value['sid'],
            backend=backend,arm=arm,point=value['point'],windows=len(selected),
            backend_windows=sum(s['kind']==1 for s in selected),
            exact_fallback_windows=sum(s['kind']==0 for s in selected),
            corrected_windows=sum(s['count']>0 for s in selected),correction_values=sum(s['count'] for s in selected),
            base_bytes=sum(s['base_bytes'] for s in selected),correction_bytes=sum(s['correction_bytes'] for s in selected)))
    pairs=[]
    for key,before in by_key.items():
        if before['status']!='complete':
            continue
        for contrast,base_arm,target_arm in CONTRASTS:
            if key[-1]!=base_arm:
                continue
            after=by_key.get(key[:-1]+(target_arm,))
            if after is None or after['status']!='complete':
                continue
            if (before['raw_bytes'],before['windows'])!=(after['raw_bytes'],after['windows']):
                raise RuntimeError('Compression pairing coverage/numerator differs')
            ratio=after['cr']/before['cr']
            pairs.append(dict(task=before['task'],dataset=before['dataset'],sid=before['sid'],split=before['split'],
                backend=before['backend'],point=before['point'],contrast=contrast,before_arm=base_arm,after_arm=target_arm,
                before_cr=before['cr'],after_cr=after['cr'],cr_ratio=ratio,cr_change_percent=100*(ratio-1),
                before_bytes=before['artifact_bytes'],after_bytes=after['artifact_bytes'],result=_result(ratio-1)))
    arm_groups=defaultdict(list); pair_groups=defaultdict(list); mechanism_groups=defaultdict(list)
    for row in records:
        if row['status']=='complete':
            arm_groups[tuple(row[k] for k in ('task','split','backend','arm','point'))].append(row)
    for row in pairs:
        pair_groups[tuple(row[k] for k in ('task','split','backend','point','contrast','before_arm','after_arm'))].append(row)
    for row in mechanisms:
        mechanism_groups[tuple(row[k] for k in ('task','backend','arm','point'))].append(row)
    arm_summary=[]
    for key,values in sorted(arm_groups.items()):
        value=dict(zip(('task','split','backend','arm','point'),key))
        windows=sum(r['windows'] for r in values)
        value.update(conditions=len(values),sources=len({r['dataset'] for r in values}),
            cr_source_geomean=_source(values,'cr',True),artifact_bytes_sum=sum(r['artifact_bytes'] for r in values),
            enabled_strict_conditions=sum(r['enabled_strict'] for r in values),
            enabled_numerical_conditions=sum(r['enabled_numerical'] for r in values),windows=windows,
            **{field:sum(r[field] for r in values) for field in ('all_ERA_strict_pass','all_ERA_numerical_pass',
                                                                  'E_numerical_pass','R_numerical_pass','A_numerical_pass')})
        arm_summary.append(value)
    pair_summary=[]
    for key,values in sorted(pair_groups.items()):
        counts=Counter(r['result'] for r in values)
        value=dict(zip(('task','split','backend','point','contrast','before_arm','after_arm'),key))
        value.update(pairs=len(values),sources=len({r['dataset'] for r in values}),
            before_cr_source_geomean=_source(values,'before_cr',True),after_cr_source_geomean=_source(values,'after_cr',True),
            cr_ratio_source_geomean=_source(values,'cr_ratio',True),improved=counts['improved'],tied=counts['tied'],declined=counts['declined'])
        pair_summary.append(value)
    mechanism_summary=[]
    for key,values in sorted(mechanism_groups.items()):
        value=dict(zip(('task','backend','arm','point'),key))
        value.update(conditions=len(values),**{field:sum(r[field] for r in values) for field in
            ('windows','backend_windows','exact_fallback_windows','corrected_windows','correction_values','base_bytes','correction_bytes')})
        mechanism_summary.append(value)
    return records,pairs,arm_summary,pair_summary,mechanisms,mechanism_summary


def downstream_tables():
    """Keep slim values, not 220k full persisted lineage objects, in memory."""
    by_key={}; coverage=Counter()
    for path in consumer_paths():
        row=c.read(path); coverage[(row['task'],row['status'])]+=1
        if row['status']!='complete':
            continue
        point=row['point']['label'] if isinstance(row['point'],dict) else row['point']
        key=(row['task'],row['model'],row['seed'],row['sid'],row.get('split','test'),row['method'],point)
        if key in by_key:
            raise RuntimeError('Duplicate consumer report key: '+repr(key))
        fields=[PRIMARY[row['task']][0]]+(['VUS_PR'] if row['task']=='anomaly' else [])
        values={field:float(row['metrics'][field]) for field in fields}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError('Nonfinite consumer metric: '+str(path))
        by_key[key]=(row['dataset'],values,row.get('prediction_origin'))
    grouped=defaultdict(list); seed_rows=[]
    for key,(dataset,metrics,origin) in by_key.items():
        task,model,seed,sid,split,method,point=key
        if method=='raw':
            continue
        backend,arm=method_parts(method)
        if backend is None:
            raise RuntimeError('Unknown downstream method '+method)
        raw=by_key.get((task,model,seed,sid,split,'raw','raw'))
        if raw is None:
            continue
        for metric,value in metrics.items():
            record=dict(task=task,model=model,seed=seed,sid=sid,dataset=dataset,split=split,backend=backend,
                arm=arm,point=point,metric=metric,value=value,raw=raw[1][metric],prediction_origin=origin)
            seed_rows.append(record)
            gkey=tuple(record[field] for field in ('task','model','sid','dataset','split','backend','arm','point','metric'))
            grouped[gkey].append(record)
    cells=[]; cell_index={}
    for key,values in sorted(grouped.items()):
        model=key[1]; expected=set(c.SEEDS) if model in ('patchtst','resnet_v1','resnet_v2','lstm') else {0}
        if {r['seed'] for r in values}!=expected:
            continue
        record=dict(zip(('task','model','sid','dataset','split','backend','arm','point','metric'),key))
        record.update(seeds=len(expected),value=_mean([r['value'] for r in values]),raw=_mean([r['raw'] for r in values]))
        cells.append(record); cell_index[key]=record
    pairs=[]
    for key,before in cell_index.items():
        for contrast,base_arm,target_arm in CONTRASTS:
            if key[6]!=base_arm:
                continue
            after=cell_index.get(key[:6]+(target_arm,)+key[7:])
            if after is None:
                continue
            if not math.isclose(before['raw'],after['raw'],rel_tol=0,abs_tol=TIE):
                raise RuntimeError('Paired downstream Raw differs')
            direction=PRIMARY[before['task']][1]
            gain=direction*(after['value']-before['value'])
            record={field:before[field] for field in ('task','model','sid','dataset','split','backend','point','metric','seeds')}
            record.update(contrast=contrast,before_arm=base_arm,after_arm=target_arm,raw=before['raw'],
                before=before['value'],after=after['value'],gain=gain,result=_result(gain))
            pairs.append(record)
    source_groups=defaultdict(list); arm_groups=defaultdict(list); pair_groups=defaultdict(list)
    for row in cells:
        source_groups[tuple(row[k] for k in ('task','model','split','backend','arm','point','metric','dataset'))].append(row)
        arm_groups[tuple(row[k] for k in ('task','model','split','backend','arm','point','metric'))].append(row)
    for row in pairs:
        pair_groups[tuple(row[k] for k in ('task','model','split','backend','point','metric','contrast','before_arm','after_arm'))].append(row)
    source_summary=[]; arm_summary=[]; pair_summary=[]
    for key,values in sorted(source_groups.items()):
        record=dict(zip(('task','model','split','backend','arm','point','metric','dataset'),key))
        record.update(inputs=len(values),value=_mean([r['value'] for r in values]),raw=_mean([r['raw'] for r in values]))
        source_summary.append(record)
    for key,values in sorted(arm_groups.items()):
        record=dict(zip(('task','model','split','backend','arm','point','metric'),key))
        record.update(inputs=len(values),sources=len({r['dataset'] for r in values}),
                      value_source_macro=_source(values,'value'),raw_source_macro=_source(values,'raw'))
        arm_summary.append(record)
    for key,values in sorted(pair_groups.items()):
        record=dict(zip(('task','model','split','backend','point','metric','contrast','before_arm','after_arm'),key))
        counts=Counter(r['result'] for r in values)
        record.update(pairs=len(values),sources=len({r['dataset'] for r in values}),
            before_source_macro=_source(values,'before'),after_source_macro=_source(values,'after'),
            raw_source_macro=_source(values,'raw'),gain_source_macro=_source(values,'gain'),
            improved=counts['improved'],tied=counts['tied'],declined=counts['declined'])
        pair_summary.append(record)
    return seed_rows,cells,pairs,source_summary,arm_summary,pair_summary,[dict(task=k[0],status=k[1],rows=v) for k,v in sorted(coverage.items())]


def write_report():
    out=c.RUN/'analysis'; out.mkdir(parents=True,exist_ok=True)
    compression,cr_pairs,cr_arms,cr_summary,mechanisms,mechanism_summary=compression_tables()
    seed_rows,cells,pairs,sources,task_arms,task_summary,coverage=downstream_tables()
    for name,rows in (('compression',compression),('cr_pairs',cr_pairs),('cr_arm_summary',cr_arms),
        ('cr_contrast_summary',cr_summary),('mechanism_conditions',mechanisms),('mechanism_summary',mechanism_summary),
        ('downstream_seed_values',seed_rows),('downstream_values',cells),('downstream_pairs',pairs),
        ('downstream_source_model',sources),('downstream_arm_summary',task_arms),('downstream_contrast_summary',task_summary),
        ('consumer_coverage',coverage)):
        _csv(out/(name+'.csv'),rows)
    closeout_path=c.RUN/'audit/closeout/CLOSEOUT.json'
    closeout=c.read(closeout_path) if closeout_path.exists() else {}
    complete=closeout.get('passed') is True
    status='完整科学覆盖及 closeout 已通过；文件导出另行记录' if complete else '部分覆盖，不能视为完整消融结论'
    lines=['# JointZip：E／ER／EA／ERA 与 correction 完整消融','',f'**状态：{status}。**','',
        '本轮保持六个固定后端配置、公式候选政策、原始简单 correction、模型权重和数据划分。'
        '已有 ERA 码流与可严格匹配的预测直接复用；只补缺失的消融条件。'
        '六配置是 SZ3、ZFP、SPERR-1D、Serf-XOR、Serf-QT、SZ2016；Serf 的两种模式不算两个独立算法家族。','',
        '## 1. 各组回答什么问题','']
    _table(lines,['组','启用约束','候选及修复','对应问题'],[
        ['E','仅逐点误差界','同一框架的公式候选流程与简单修复，关闭 R/A','只保护数值误差是否足够'],
        ['ER','误差界与原始/重建 Pearson','关闭 A，其余不变','增加 R 的实际作用'],
        ['EA','误差界与 ACF 差','关闭 R，其余不变','增加 A 的实际作用'],
        ['ERA','三项同时保护','已完成的当前框架，原样复用','联合保护的代价和收益'],
        ['ERA_no_correction','三项同时保护','固定既有 ERA 候选轨迹，关闭修复，保留验收与精确回退','给定相同候选池时修复的作用']])
    lines += ['', 'E 组不是原生后端的一次 epsilon 调用：它仍运行框架的候选选择与完整封装。'
        'E/ER/EA 在公式生成、修复及验收中关闭相应约束，因此其比较衡量启用约束后的整体效果；'
        '不同组的自适应候选轨迹可能不同，不把它误称为候选完全固定的单一数学项试验。',
        'correction 对照则明确固定已有 ERA 的候选参数轨迹，使用相同后端、精确回退和最终验收。'
        '该候选池来自有修复的策略，结论条件于这一池；它不等同于独立运行一个重新搜索的无修复算法。'
        '无修复可能更多地保存精确数据，因此其下游分数可能更好但 CR 更低；修复收益须同时结合存储与任务指标。','',
        '## 2. 覆盖与统计口径','']
    _table(lines,['项目','已完成','计划'],[
        ['压缩配置',sum(r['status']=='complete' for r in compression),27900],
        ['预测记录',sum(r['rows'] for r in coverage if r['task']=='forecast' and r['status']=='complete'),206752],
        ['分类记录',sum(r['rows'] for r in coverage if r['task']=='classification' and r['status']=='complete'),1488],
        ['检索记录',sum(r['rows'] for r in coverage if r['task']=='retrieval' and r['status']=='complete'),1302],
        ['异常记录',sum(r['rows'] for r in coverage if r['task']=='anomaly' and r['status']=='complete'),10920]])
    reuse=Counter(r['reuse_kind'] for r in compression)
    lines += ['', f'压缩记录来源：{dict(reuse)}。',
        '计划覆盖 172 个输入：142 条预测序列、6 个 UCR 数据集、24 个 NASA 异常文件。'
        '阈值、VAL/TEST、后端和种子重复均计入记录数；这些数不是独立样本数，也不是全部重新调用模型的次数。',
        '所有 CR 使用完整 JZF1 文件字节，包含后端载荷、修复、归一化和形状等元数据；'
        '分子为真实未填充 float32 数据字节。预测和异常使用既定 strict/medium/loose，UCR 使用既定 stress。',
        'CR 先在来源内取几何平均，再对来源等权；下游先平均同一条件的种子，再在来源内平均、来源间等权。'
        '跨模型概览再对模型等权；137 条 Solar 序列不会在宏平均中获得 137 倍权重。'
        '细表保留每个来源、模型、阈值和输入。全部改善、持平和下降均保留，未按本轮 TEST 分数筛选条件。','',
        '## 3. CR、启用约束与完整 ERA','',
        '下表汇总 TEST，同一任务的既定阈值均纳入。启用通过是配置级通过数；完整 ERA 是窗口级通过数，'
        '两者分母不同。E/ER/EA 不必通过被关闭的约束，这种失败是研究结果，不是实验执行失败。','']
    groups=defaultdict(list)
    for row in compression:
        if row['split']=='test' and row['status']=='complete': groups[(row['task'],row['backend'],row['arm'])].append(row)
    cr_overview=[]
    for (task,backend,arm),values in sorted(groups.items()):
        n=len(values); windows=sum(r['windows'] for r in values)
        cr_overview.append(dict(task=task,backend=backend,arm=arm,conditions=n,
            cr_source_geomean=_source(values,'cr',True),enabled_pass=sum(r['enabled_numerical'] for r in values),
            windows=windows,all_ERA_pass=sum(r['all_ERA_numerical_pass'] for r in values)))
    _csv(out/'cr_overview.csv',cr_overview)
    _table(lines,['任务','后端','组','CR','启用通过配置','完整ERA通过窗口'],[
        [r['task'],r['backend'],r['arm'],r['cr_source_geomean'],f"{r['enabled_pass']}/{r['conditions']}",f"{r['all_ERA_pass']}/{r['windows']}"]
        for r in cr_overview])
    lines += ['', '严格不加容差的审计另见 `analysis/cr_arm_summary.csv`；本表通过率使用既定数值容差。'
        '增加约束可能改变候选生成及压缩结果，因此不预设各消融组 CR 必须单调。','',
        '## 4. 四类下游任务','',
        '预测 MSE 越低越好；分类 accuracy、检索 mAP（保存字段 AP）、异常 AP/VUS-PR 越高越好。'
        '下表是 TEST 来源及模型等权概览。原始数据 Raw 为冻结模型的参照，不能把不同任务的数字直接比较。','']
    groups=defaultdict(list)
    for row in cells:
        if row['split']=='test': groups[(row['task'],row['backend'],row['metric'])].append(row)
    task_overview=[]
    for (task,backend,metric),values in sorted(groups.items()):
        record=dict(task=task,backend=backend,metric=metric)
        for arm in c.ARMS:
            selected=[r for r in values if r['arm']==arm]
            record[arm]=_macro_model_source(selected,'value')
        raw_rows=[r for r in values if r['arm']=='ERA']
        record['raw']=_macro_model_source(raw_rows,'raw'); task_overview.append(record)
    _csv(out/'downstream_overview.csv',task_overview)
    _table(lines,['任务','后端','指标','Raw',*c.ARMS],[
        [r['task'],r['backend'],r['metric'],r['raw'],*[r[a] for a in c.ARMS]] for r in task_overview])
    lines += ['', '### 每个模型的结果','',
        '下表将阈值汇总，种子已先平均。逐阈值值见 `downstream_arm_summary.csv`；'
        '逐来源模型值见 `downstream_source_model.csv`，逐输入值见 `downstream_values.csv`。','']
    model_groups=defaultdict(list)
    for row in cells:
        if row['split']=='test': model_groups[(row['task'],row['model'],row['backend'],row['metric'])].append(row)
    model_rows=[]
    for (task,model,backend,metric),values in sorted(model_groups.items()):
        model_rows.append([task,model,backend,metric,_source([r for r in values if r['arm']=='ERA'],'raw'),
                          *[_source([r for r in values if r['arm']==a],'value') for a in c.ARMS]])
    _table(lines,['任务','模型','后端','指标','Raw',*c.ARMS],model_rows)
    lines += ['', '### 增加 R、A 的配对变化','',
        '“改善/持平/下降”统计同一来源输入、模型、阈值的种子平均配对；这些条件相关，不能当独立样本作显著性推断。'
        '正的平均收益表示目标组更好：预测为 MSE 减少，其余为分数增加。','']
    pair_groups=defaultdict(list)
    for row in pairs:
        if row['split']=='test': pair_groups[(row['task'],row['backend'],row['metric'],row['contrast'])].append(row)
    contrast_overview=[]
    for (task,backend,metric,contrast),values in sorted(pair_groups.items()):
        counts=Counter(r['result'] for r in values)
        contrast_overview.append(dict(task=task,backend=backend,metric=metric,contrast=contrast,
            before=_macro_model_source(values,'before'),after=_macro_model_source(values,'after'),
            gain=_macro_model_source(values,'gain'),pairs=len(values),
            improved=counts['improved'],tied=counts['tied'],declined=counts['declined']))
    _csv(out/'downstream_contrast_overview.csv',contrast_overview)
    _table(lines,['任务','后端','指标','比较','平均收益','改善/平/降'],[
        [r['task'],r['backend'],r['metric'],r['contrast'],r['gain'],f"{r['improved']}/{r['tied']}/{r['declined']}"]
        for r in contrast_overview if r['contrast']!='correction'])
    lines += ['', '## 5. correction 固定候选池消融','',
        '下表比较同一个 ERA 候选池下关闭和保留简单修复。候选不满足 ERA 时，无修复组只能选其他合格候选或精确回退；'
        '它不会输出违反 ERA 的文件。完整文件 CR 会受到外层封装影响，不能只按内部载荷推断其排序。','']
    correction_cr=[]
    cr_groups=defaultdict(list)
    for row in cr_pairs:
        if row['split']=='test' and row['contrast']=='correction': cr_groups[(row['task'],row['backend'])].append(row)
    for (task,backend),values in sorted(cr_groups.items()):
        counts=Counter(r['result'] for r in values)
        correction_cr.append(dict(task=task,backend=backend,pairs=len(values),
            no_correction_cr=_source(values,'before_cr',True),correction_cr=_source(values,'after_cr',True),
            ratio=_source(values,'cr_ratio',True),improved=counts['improved'],tied=counts['tied'],declined=counts['declined']))
    _csv(out/'correction_cr_overview.csv',correction_cr)
    _table(lines,['任务','后端','无修复CR','有修复CR','CR比值','改善/平/降'],[
        [r['task'],r['backend'],r['no_correction_cr'],r['correction_cr'],r['ratio'],f"{r['improved']}/{r['tied']}/{r['declined']}"] for r in correction_cr])
    lines += ['']
    _table(lines,['任务','后端','指标','无修复','有修复','平均收益','改善/平/降'],[
        [r['task'],r['backend'],r['metric'],r['before'],r['after'],r['gain'],f"{r['improved']}/{r['tied']}/{r['declined']}"]
        for r in contrast_overview if r['contrast']=='correction'])
    lines += ['', '## 6. 精确回退与修复记录','',
        '只计最终写入码流的 TEST 窗口，未把搜索中淘汰的候选重复计入。后端路径不必每次都产生非零误差；'
        '精确回退是保存原 float32 的无损候选。不同任务和阈值重复的窗口不能当独立样本。','']
    mechanism_groups=defaultdict(list)
    for row in mechanisms: mechanism_groups[(row['task'],row['backend'],row['arm'])].append(row)
    mechanism_overview=[]
    for (task,backend,arm),values in sorted(mechanism_groups.items()):
        windows=sum(r['windows'] for r in values); fallback=sum(r['exact_fallback_windows'] for r in values)
        corrected=sum(r['corrected_windows'] for r in values)
        mechanism_overview.append(dict(task=task,backend=backend,arm=arm,windows=windows,
            fallback_windows=fallback,fallback_percent=100*fallback/windows,corrected_windows=corrected,
            corrected_percent=100*corrected/windows,correction_values=sum(r['correction_values'] for r in values),
            base_bytes=sum(r['base_bytes'] for r in values),correction_bytes=sum(r['correction_bytes'] for r in values)))
    _csv(out/'mechanism_overview.csv',mechanism_overview)
    _table(lines,['任务','后端','组','窗口','精确回退%','有修复窗口%','修复值','修复载荷字节'],[
        [r['task'],r['backend'],r['arm'],r['windows'],r['fallback_percent'],r['corrected_percent'],r['correction_values'],r['correction_bytes']]
        for r in mechanism_overview])
    lines += ['', '修复载荷字节不含完整容器头及外层编码，不能直接用于完整 CR 的分母。'
        '修复发生频率只描述机制使用情况，其净收益应以第 5 节固定候选池比较为准。','',
        '## 7. 适用范围与耗时','',
        '- 本轮使用相同输入、用户阈值和后端，不声称各组占用相同存储。质量与 CR 应结合阅读。',
        '- E/R/A 是重建合同，不保证所有下游任务都改善；保留下降和持平有助于判断边界。',
        '- 公式给出窗口标量候选锚点，并允许后续细化；最终保护来自解码、原简单修复和验收，不是压缩前非线性 ACF 的确定性保证。',
        '- 已有 ERA 和预测缓存的复用、并行运行及候选缓存会改变流水线耗时。本轮未新增独立计时，不以运行日志给消融组作速度排名，也不把旧独立计时移植成新组件耗时。',
        '- 无修复对照可能需要重新生成旧流程未保存的候选基础码流；这是补足未保存候选，不是重复已完成 ERA 配置，报告来源与复用记录可追溯。',
        '- 本轮不扩展公式/无公式、22/44/88 搜索密度或等 CR 实验；此前相关结果仍是独立研究。',
        '- 当前数据曾用于方法开发，这是一轮完整但探索性的消融，不冒充未见测试数据的预注册确认实验。',
        '- SZ2016 保留此前披露的官方旧实现及 zlib 容量修补；完整压缩存在已知上游 ASan best-fit 构造器问题，不声称所有内存检查通过。','',
        '完整配对数据、来源/模型/阈值明细见 `analysis/`。严格覆盖、缺失和异常见 `audit/closeout/CLOSEOUT.json`。','']
    (c.RUN/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    summary=dict(complete=complete,compression_rows=len(compression),compression_pairs=len(cr_pairs),
        consumer_coverage=coverage,downstream_seed_values=len(seed_rows),downstream_seed_averaged_values=len(cells),
        downstream_pairs=len(pairs),mechanism_test_conditions=len(mechanisms),
        timing='No new independent timing; diagnostic or reused times excluded from comparison.')
    c.write(out/'REPORT_SUMMARY.json',summary)
    return dict(complete=complete,report=str(c.RUN/'REPORT_ZH.md'),compression_rows=len(compression),downstream_pairs=len(pairs))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.parse_args()
    print(write_report(),flush=True)
