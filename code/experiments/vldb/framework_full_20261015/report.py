"""Complete-byte and fixed-consumer paired reports, including negative results."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics

from . import common as c

PRIMARY = {'forecast': ('mse', -1), 'classification': ('accuracy', 1),
           'retrieval': ('AP', 1), 'anomaly': ('AP', 1)}
TIE = 1e-10


def method_parts(method):
    for arm in ('native', 'jointzip'):
        suffix = '_' + arm
        if method.endswith(suffix):
            return method[:-len(suffix)], arm
    return None, None


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _geo(values):
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def _source_macro(rows, key):
    by_source = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            by_source[row['dataset']].append(row[key])
    return _mean([_mean(values) for values in by_source.values()])


def _source_geo(rows, key):
    by_source = defaultdict(list)
    for row in rows:
        if row.get(key) is not None and row[key] > 0:
            by_source[row['dataset']].append(row[key])
    return _geo([_geo(values) for values in by_source.values()])


def _audit(row, name='strict_audit'):
    audit = row.get(name, {})
    windows = audit.get('series_windows', audit.get('windows', row.get('shape', [0])[0]))
    passed = audit.get('joint_pass_count', audit.get('strict_pass_windows' if name == 'strict_audit' else 'numerical_pass_windows'))
    return int(windows), None if passed is None else int(passed)


def compression_tables():
    records, by_key = [], {}
    for path in sorted((c.RUN/'compression').glob('*/*/*/*.json')):
        row = c.read(path)
        backend, arm = method_parts(row.get('method', ''))
        if backend is None:
            continue
        point = row['point']['label']
        key = (row['task'],row['sid'],row['split'],backend,point,arm)
        if key in by_key:
            raise RuntimeError('Duplicate compression report key: '+repr(key))
        by_key[key] = row
        windows, passed = _audit(row)
        _, numerical = _audit(row, 'numerical_audit')
        records.append(dict(task=row['task'],dataset=row['dataset'],sid=row['sid'],split=row['split'],
            backend=backend,arm=arm,point=point,status=row['status'],
            raw_bytes=row.get('raw_bytes'),artifact_bytes=row.get('artifact_bytes'),cr=row.get('cr'),
            windows=windows,joint_strict_pass=passed,joint_numerical_pass=numerical,
            encode_seconds=row.get('encode_seconds'),decode_seconds=row.get('decode_seconds'),
            reuse_kind=row.get('reuse',{}).get('kind'),source_row=str(path.relative_to(c.RUN))))
    pairs = []
    for key, native in by_key.items():
        if key[-1] != 'native':
            continue
        framework = by_key.get(key[:-1]+('jointzip',))
        if framework is None or native['status'] != 'complete' or framework['status'] != 'complete':
            continue
        if native['raw_bytes'] != framework['raw_bytes']:
            raise RuntimeError('Paired complete-file raw numerator differs')
        nw,np = _audit(native); fw,fp = _audit(framework)
        if nw != fw:
            raise RuntimeError('Paired constraint window coverage differs')
        ratio = framework['cr']/native['cr']
        pairs.append(dict(task=native['task'],dataset=native['dataset'],sid=native['sid'],split=native['split'],
            backend=key[3],point=key[4],native_cr=native['cr'],jointzip_cr=framework['cr'],cr_ratio=ratio,
            cr_change_percent=100*(ratio-1),native_bytes=native['artifact_bytes'],jointzip_bytes=framework['artifact_bytes'],
            windows=nw,native_joint_pass=np,jointzip_joint_pass=fp,
            result='improved' if ratio>1+TIE else 'declined' if ratio<1-TIE else 'tied'))
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair['task'],pair['split'],pair['backend'],pair['point'])].append(pair)
    summary=[]
    for (task,split,backend,point), values in sorted(groups.items()):
        counts=Counter(v['result'] for v in values)
        summary.append(dict(task=task,split=split,backend=backend,point=point,paired_conditions=len(values),
            sources=len({v['dataset'] for v in values}),native_cr_source_geomean=_source_geo(values,'native_cr'),
            jointzip_cr_source_geomean=_source_geo(values,'jointzip_cr'),cr_ratio_source_geomean=_source_geo(values,'cr_ratio'),
            native_bytes_sum=sum(v['native_bytes'] for v in values),jointzip_bytes_sum=sum(v['jointzip_bytes'] for v in values),
            improved=counts['improved'],tied=counts['tied'],declined=counts['declined'],
            windows=sum(v['windows'] for v in values),
            native_joint_pass=sum(v['native_joint_pass'] or 0 for v in values),
            jointzip_joint_pass=sum(v['jointzip_joint_pass'] or 0 for v in values)))
    return records,pairs,summary


def consumer_paths():
    return sorted([*(c.RUN/'evaluation/forecast/rows').glob('*/*/*/*/*.json'),
        *(c.RUN/'evaluation/ucr/classification').glob('*/*/*/*.json'),
        *(c.RUN/'evaluation/ucr/retrieval').glob('*/*/*/*.json'),
        *(c.RUN/'evaluation/anomaly/rows').glob('*/*/*.json')])


def downstream_tables():
    by_key, counts = {}, Counter()
    for path in consumer_paths():
        row=c.read(path)
        counts[(row['task'],row['status'])]+=1
        if row['status'] != 'complete':
            continue
        point = row['point']['label'] if isinstance(row['point'],dict) else row['point']
        key=(row['task'],row['model'],row['seed'],row['sid'],row.get('split','test'),row['method'],point)
        if key in by_key:
            raise RuntimeError('Duplicate consumer report key: '+repr(key))
        by_key[key]=row
    seed_pairs=[]
    for key,native in by_key.items():
        task,model,seed,sid,split,method,point=key
        backend,arm=method_parts(method)
        if arm!='native':
            continue
        framework=by_key.get((task,model,seed,sid,split,backend+'_jointzip',point))
        raw=by_key.get((task,model,seed,sid,split,'raw','raw'))
        if framework is None or raw is None:
            continue
        metric,direction=PRIMARY[task]
        metric_names=[metric]
        if task=='anomaly':
            metric_names+=['VUS_PR']
        for field in metric_names:
            if any(field not in r['metrics'] for r in (native,framework,raw)):
                raise RuntimeError('Missing complete consumer primary metric: '+field)
            n,f,r=(float(v['metrics'][field]) for v in (native,framework,raw))
            if not all(math.isfinite(v) for v in (n,f,r)):
                raise RuntimeError('Nonfinite complete consumer primary metric')
            delta=f-n;gain=direction*delta
            seed_pairs.append(dict(task=task,model=model,seed=seed,sid=sid,dataset=native['dataset'],
                split=split,backend=backend,point=point,metric=field,direction=direction,
                raw=r,native=n,jointzip=f,delta=delta,gain=gain,
                result='improved' if gain>TIE else 'declined' if gain<-TIE else 'tied',
                native_prediction_origin=native.get('prediction_origin'),jointzip_prediction_origin=framework.get('prediction_origin')))
    # Seeds are repeated fits of the same source, not separate data sets.
    grouped=defaultdict(list)
    for row in seed_pairs:
        key=tuple(row[k] for k in ('task','model','sid','dataset','split','backend','point','metric','direction'))
        grouped[key].append(row)
    pairs=[]
    for key,values in sorted(grouped.items()):
        expected=3 if key[1] in ('patchtst','resnet_v1','resnet_v2','lstm') else 1
        if len({v['seed'] for v in values})!=expected:
            continue  # Partial reports never compare a favorable subset of seeds.
        value=dict(zip(('task','model','sid','dataset','split','backend','point','metric','direction'),key))
        value.update(seeds=expected,**{f:_mean([r[f] for r in values]) for f in ('raw','native','jointzip','delta','gain')})
        value['result']='improved' if value['gain']>TIE else 'declined' if value['gain']<-TIE else 'tied'
        pairs.append(value)
    groups=defaultdict(list)
    for row in pairs:
        key=tuple(row[k] for k in ('task','model','split','backend','point','metric'))
        groups[key].append(row)
    summary=[]
    for key,values in sorted(groups.items()):
        value=dict(zip(('task','model','split','backend','point','metric'),key))
        counts2=Counter(r['result'] for r in values)
        value.update(paired_inputs=len(values),sources=len({r['dataset'] for r in values}),
            raw_source_macro=_source_macro(values,'raw'),native_source_macro=_source_macro(values,'native'),
            jointzip_source_macro=_source_macro(values,'jointzip'),gain_source_macro=_source_macro(values,'gain'),
            improved=counts2['improved'],tied=counts2['tied'],declined=counts2['declined'])
        summary.append(value)
    return seed_pairs,pairs,summary,[dict(task=k[0],status=k[1],rows=v) for k,v in sorted(counts.items())]


def _fmt(value):
    return '—' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)


def timing_tables():
    """One warmup excluded; three independent measurements per paired cell."""
    records=[];groups=defaultdict(dict)
    for path in sorted((c.RUN/'timing/rows').glob('*.json')):
        row=c.read(path)
        if row.get('status')!='complete':
            continue
        records.append(row)
        if row.get('warmup') or row['repeat']==0:
            continue
        key=(row['dataset'],row['backend'],row['point']['label'])
        arm='jointzip' if row['framework'] else 'native'
        cell=groups[key].setdefault(arm,{})
        if row['repeat'] in cell:
            raise RuntimeError('Duplicate independent timing repetition')
        cell[row['repeat']]=row
    pairs=[]
    for (dataset,backend,point),arms in sorted(groups.items()):
        if set(arms)!= {'native','jointzip'} or any(set(v)!={1,2,3} for v in arms.values()):
            continue
        value=dict(dataset=dataset,backend=backend,point=point,windows=4,measured_repeats=3)
        for arm,rows in arms.items():
            for field in ('encode_seconds','decode_seconds','cr'):
                value[arm+'_'+field+'_median']=statistics.median(r[field] for r in rows.values())
        for field in ('encode_seconds','decode_seconds','cr'):
            value[field+'_ratio']=value['jointzip_'+field+'_median']/value['native_'+field+'_median']
        pairs.append(value)
    by_backend=defaultdict(list)
    for pair in pairs:
        by_backend[pair['backend']].append(pair)
    summary=[]
    for backend,values in sorted(by_backend.items()):
        summary.append(dict(backend=backend,paired_conditions=len(values),source_count=len({v['dataset'] for v in values}),
            native_encode_seconds_mean=_source_macro(values,'native_encode_seconds_median'),
            jointzip_encode_seconds_mean=_source_macro(values,'jointzip_encode_seconds_median'),
            encode_ratio_source_geomean=_source_geo(values,'encode_seconds_ratio'),
            native_decode_seconds_mean=_source_macro(values,'native_decode_seconds_median'),
            jointzip_decode_seconds_mean=_source_macro(values,'jointzip_decode_seconds_median'),
            decode_ratio_source_geomean=_source_geo(values,'decode_seconds_ratio'),
            cr_ratio_source_geomean=_source_geo(values,'cr_ratio')))
    return records,pairs,summary


def write_report():
    out=c.RUN/'analysis';out.mkdir(parents=True,exist_ok=True)
    records,cr_pairs,cr_summary=compression_tables()
    seed_pairs,task_pairs,task_summary,coverage=downstream_tables()
    timing_records,timing_pairs,timing_summary=timing_tables()
    for name,rows in [('compression',records),('cr_pairs',cr_pairs),('cr_summary',cr_summary),
                      ('downstream_seed_pairs',seed_pairs),('downstream_pairs',task_pairs),
                      ('downstream_summary',task_summary),('consumer_coverage',coverage),
                      ('timing_pairs',timing_pairs),('timing_summary',timing_summary)]:
        _csv(out/(name+'.csv'),rows)
    closeout_path=c.RUN/'audit/closeout/CLOSEOUT.json'
    closeout=c.read(closeout_path) if closeout_path.exists() else {}
    complete=closeout.get('passed') is True
    status='完整主实验已完成' if complete else '部分结果；不能作为完整主实验结论'
    lines=['# JointZip 固定后端完整实验报告','',f'**状态：{status}。**','',
      '对同一后端比较原生压缩与 JointZip：同输入、同用户 epsilon、同训练好的下游模型；'
      'JointZip 使用 Pearson/ACF 公式指导的窗口参数、原简单 correction，保留无损回退。',
      '本轮固定后端为 SZ3、ZFP、SPERR 1D、Serf-XOR、Serf-QT；SPERR 1D 与历史组合版的 2D n×1 模式分别记账。',
      '旧数据能复用的原生窗口码流、重建及严格匹配的模型预测均注明来历；旧 SCS2/pilot CR 不直接搬入新表。'
      '所有 CR 统一按本轮完整 JZF1 文件字节计算，包含归一化、真实长度及内部 JZP1。','',
      '|覆盖|已提交 complete|预期|','|---|---:|---:|',
      f"|压缩条件|{sum(r['status']=='complete' for r in records)}|9300|",
      f"|压缩配对|{len(cr_pairs)}|4650|",'']
    for item in coverage:
        lines.append(f"- {item['task']}：{item['status']} {item['rows']} 条消费者记录。")
    lines += ['', '记录包含同一数据的阈值、后端、种子、VAL/TEST及缓存复用，不能当作独立样本数。',
      '', '## 压缩率及联合约束', '',
      '下表展示 TEST；VAL 明细保留在 CSV。CR 先在每个数据来源内取几何平均，再对来源等权，'
      '避免 137 条 Solar 序列主导总体。改善/持平/下降是条件配对数；严格通过统计是含重复条件的窗口记录。', '',
      '|任务|后端|阈值|配对|原生CR|框架CR|CR比值|CR改善/平/降|原生联合通过|框架联合通过|',
      '|---|---|---|---:|---:|---:|---:|---|---:|---:|']
    for row in cr_summary:
        if row['split']!='test':continue
        fields=[row['task'],row['backend'],row['point'],row['paired_conditions'],
                row['native_cr_source_geomean'],row['jointzip_cr_source_geomean'],row['cr_ratio_source_geomean'],
                f"{row['improved']}/{row['tied']}/{row['declined']}",
                f"{row['native_joint_pass']}/{row['windows']}",f"{row['jointzip_joint_pass']}/{row['windows']}"]
        lines.append('| '+' | '.join(map(_fmt,fields))+' |')
    lines += ['', '## 下游效果', '',
      '神经模型先平均同一输入的三个种子；再在来源内平均、来源间等权。预测 MSE 越低越好；'
      '分类准确率、检索 mAP（字段 AP）、异常 AP/VUS-PR 越高越好。全部改善、持平、下降均保留；'
      '没有以 TEST 分数选择压缩参数或后端。','',
      '|任务|模型|后端|阈值|指标|配对输入|Raw|原生|框架|改善/平/降|',
      '|---|---|---|---|---|---:|---:|---:|---:|---|']
    for row in task_summary:
        if row['split']!='test':continue
        fields=[row['task'],row['model'],row['backend'],row['point'],row['metric'],row['paired_inputs'],
                row['raw_source_macro'],row['native_source_macro'],row['jointzip_source_macro'],
                f"{row['improved']}/{row['tied']}/{row['declined']}"]
        lines.append('| '+' | '.join(map(_fmt,fields))+' |')
    lines += ['', '## 参数搜索与耗时', '',
      '本轮主实验沿用已验证的公式候选政策。增大候选上限的密度试验单列，不能用其结果替换主实验或删去下降结果。'
      '复用码流不计为零成本的新压缩；共享缓存及流水线墙钟不用于官方内核吞吐量排名。']
    timing_complete=c.RUN/'timing/COMPLETE.json'
    if timing_complete.exists() and c.read(timing_complete).get('complete') and len(timing_records)==480 and len(timing_pairs)==60:
        lines += ['', '独立计时已完成 480 次调用：120 次预热不计入结果，360 次正式测量。']
    else:
        lines += ['', f'独立计时仍为部分覆盖：{len(timing_records)}/480 条调用记录，{len(timing_pairs)}/60 个完整三重复配对。']
    lines += ['', '每个条件取三次正式调用的中位数；下表秒数再对来源及两个阈值平均，比值取来源等权几何平均。'
      '面板为六预测来源的首条序列、前四个 VAL 窗口、strict/loose；不能当作 UCR/NASA 或全数据吞吐量。'
      '编码计时含公式、候选调用、修复、完整封装、候选解码及约束验证，排除编码末尾额外的独立解码审计；解码列单独测量完整流解码。','',
      '|后端|完整配对|原生编码秒/4窗|框架编码秒/4窗|编码耗时比|原生解码秒/4窗|框架解码秒/4窗|解码耗时比|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in timing_summary:
        fields=[row['backend'],row['paired_conditions'],row['native_encode_seconds_mean'],row['jointzip_encode_seconds_mean'],
                row['encode_ratio_source_geomean'],row['native_decode_seconds_mean'],row['jointzip_decode_seconds_mean'],row['decode_ratio_source_geomean']]
        lines.append('| '+' | '.join(map(_fmt,fields))+' |')
    density=c.RUN/'density/REPORT_ZH.md'
    if density.exists():
        lines += ['', '搜索密度试验报告：`density/REPORT_ZH.md`。', '', density.read_text(encoding='utf-8')]
    else:
        lines += ['', '搜索密度的完整比较尚未写入本报告；当前不宣称增加搜索次数已经改善。']
    lines += ['', '## 解释范围', '',
      '- 这是相同用户 epsilon 下的比较，不能直接推导相同 CR 的任务优势。',
      '- 统计约束达标与任务改善分别报告；约束不保证任何下游模型必然改善。',
      '- 无损回退、基础后端和 correction 的贡献需按实际选中记录分别解释。',
      '- UCR 使用完整官方 TEST 与完整原始 TRAIN 参照库；检索是类别相关性派生实验。',
      '- NASA 只还原真实 TEST 后缀，训练前缀、阈值和标签不参与压缩参数生成。',
      '- 当前版本是在已见数据上的方法开发结果，不冒充未见测试集上的预注册确认实验。', '',
      '完整逐配置表见 `analysis/`。覆盖与缺失名单见 `audit/closeout/CLOSEOUT.json`。','']
    (c.RUN/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    c.write(out/'REPORT_SUMMARY.json',dict(complete=complete,compression_rows=len(records),cr_pairs=len(cr_pairs),
        downstream_seed_pairs=len(seed_pairs),downstream_seed_averaged_pairs=len(task_pairs),consumer_coverage=coverage,
        independent_timing_records=len(timing_records),complete_timing_pairs=len(timing_pairs)))
    return dict(complete=complete,report=str(c.RUN/'REPORT_ZH.md'),compression_pairs=len(cr_pairs),downstream_pairs=len(task_pairs))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args()
    print(write_report(),flush=True)
