"""Render the verified complete-file accounting results without rerunning codecs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/results/vldb_uniform_lossless_accounting_20261031_v1'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def rows(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    summary = read('SUMMARY.json')
    assert read('COMPLETE.json')['complete']
    assert read('INDEPENDENT_REVIEW.json')['passed']
    assert read('ROSTER_IDENTITY_REVIEW.json')['passed']
    data = rows('macros.csv')
    names = {'sz3': 'SZ3', 'zfp': 'ZFP', 'sperr': 'SPERR-1D',
             'serf_xor': 'Serf-XOR', 'serf_qt': 'Serf-QT', 'sz2016': 'SZ2016'}
    tasks = {'forecast': '预测', 'ucr': '分类与检索', 'anomaly': '异常检测'}
    indexed = {(r['backend'], r['task']): r for r in data}
    lines = [
        '# JointZip 与无损 Zstd 的统一完整文件计费比较',
        '',
        '2026 年 10 月 31 日',
        '',
        '在相同输入、相同公共元数据及完整文件计费下，18 个“后端 × 任务组”的来源等权几何平均压缩率均高于无损 Zstd。逐配置比较为 **2,500 个更高、0 个持平、524 个更低**。因此，汇总压缩优势仍然成立，但当前实现没有逐文件不劣于这一无损基线的保证。',
        '',
        '此次只重新封装并核对现有编码结果，没有改变重建数据，没有重跑压缩搜索、correction 或下游任务，也没有据此修改论文或 Overleaf。',
        '',
        '## 比较范围与计费规则',
        '',
        '覆盖原主实验全部 172 个 TEST 输入和六个后端的 3,024 个完整 ERA 配置。每后端 504 项，包括 142 个预测输入 × 3 档设置、6 个 UCR 输入 × stress 设置、24 个 NASA 输入 × 3 档设置。分类与检索共用 UCR 压缩文件，故压缩率分为三个任务组。此次范围不包含后续 FordA/FordB 补充实验。',
        '',
        '两边均实际生成 JCF1 完整文件。对同一个输入，公共文件头和 JSON 元数据逐字节相同，计入数据类型、形状、有效点数、归一化均值与标准差、通道名称以及版本。各方法自己的编码标识、窗口边界、记录长度、压缩数据和修复数据也全部计费。实验哈希等追溯记录在两边均放入外部审计清单，不影响独立解码。',
        '',
        '| 文件组成 | JointZip | 无损 Zstd |',
        '|---|---|---|',
        '| 公共文件头与元数据 | 同一份公共元数据，全计 | 同一份公共元数据，全计 |',
        '| 编码类型标识 | 4 字节 JZP1，全计 | 4 字节 ZLC1，全计 |',
        '| 原有编码结构 | 原 JZP1 内部全部数据保留 | 原 NCS2 集合全部数据保留 |',
        '| 窗口与批次信息 | 窗口记录、长度、cap 等全计 | 批次模式与长度、窗口形状与长度全计 |',
        '| 修复记录 | 全计 | 不需要修复记录 |',
        '| 追溯信息 | 文件外保存 | 文件外保存 |',
        '',
        '统一的是公共封装与计费原则，不是强行让无损文件承担 JointZip 专有的修复记录。各自解码需要的内部结构全部保留。这里的无损基线沿用已有逐窗 Zstd level 19，以及已保存的批次外层编码选择（原样或 Zstd level 3/9/15/19），没有重新挑选或压缩。它与 JointZip 内部的逐窗 Zstd level 3 候选不同。',
        '',
        '压缩率 CR = 原始有效 Float32 字节数 / 完整文件实际字节数，越大越好。分子为 4 × 有效点数，不把补齐点当作原始数据。表中先在各来源内部计算几何平均，再对来源等权汇总。预测有 6 个来源，UCR 有 6 个来源，NASA 有 2 个来源。预测与 NASA 同时覆盖 loose、medium、strict，UCR 使用原实验 stress 设置。',
        '',
        '## 完整文件压缩率',
        '',
        '| 方法 | 预测 | 分类与检索 | 异常检测 |',
        '|---|---:|---:|---:|',
    ]
    base = [float(indexed['sz3', t]['lossless_cr']) for t in tasks]
    lines.append('| 无损 Zstd | ' + ' | '.join(f'{v:.3f}' for v in base) + ' |')
    for backend, name in names.items():
        values = [float(indexed[backend, t]['jointzip_cr']) for t in tasks]
        lines.append(f'| JointZip + {name} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += [
        '',
        '18 组的压缩率比值 JointZip/Zstd 为 1.521–20.136。此范围是上述分组几何平均之比，不是逐配置范围，也不是总存储字节之比。',
        '',
        '## 逐配置更高与更低的情况',
        '',
        '下表每格为“JointZip CR 更高 / 持平 / 更低”的配置数。总计 2,500 / 0 / 524，即 82.67% / 0% / 17.33%。这些配置共享输入与设置，不应当作为 3,024 个独立统计样本。',
        '',
        '| 后端 | 预测 426 项 | 分类与检索 6 项 | 异常检测 72 项 |',
        '|---|---:|---:|---:|',
    ]
    for backend, name in names.items():
        values = [' / '.join(indexed[backend, t][k] for k in ('higher', 'tied', 'lower')) for t in tasks]
        lines.append(f'| {name} | ' + ' | '.join(values) + ' |')
    lines += [
        '',
        '524 个更低的配置中，343 个来自 SZ3 预测，181 个来自 NASA 异常检测。按“后端 × 来源”进一步分组，84 组中 83 组更高。唯一更低的一组是 SZ3/Solar，JointZip CR 为 3.348，无损为 3.616，411 个配置中 70 个更高、341 个更低。Solar 包含 137 个输入，其他五个预测来源各 1 个输入，因此来源等权宏平均与逐配置多数票并不等价。',
        '',
        '## 为什么统一元数据后仍有更大的文件',
        '',
        '当前实现对每个窗口比较“原生码流字节 + correction 字节”与该窗口的 Zstd level 3 无损候选，选取较小者，然后组装所有窗口记录，并对完整码流进行外层压缩。它没有在最后把完整 JointZip 文件与完整独立无损文件再次比较并取较小者。',
        '',
        '因此，“窗口候选不会比其 Zstd3 候选更大”不能推出“最终文件不会比独立的 Zstd19 无损基线更大”。编码等级、内部记录与最终外层压缩都可能影响完整大小。若要求最后这一保证，需要另行增加完整文件级选择机制。此次遵照统一计费的范围，未增加该机制。',
        '',
        '一个真实负例是 SMAP 544 / ZFP / strict：',
        '',
        '| 完整文件组成 | JointZip | 无损 Zstd |',
        '|---|---:|---:|',
        '| 相同公共文件头与元数据 | 143 B | 143 B |',
        '| 各自编码数据及全部内部结构 | 5,115 B | 3,548 B |',
        '| 完整文件大小 | **5,258 B** | **3,691 B** |',
        '| 原始有效数据大小 | 23,744 B | 23,744 B |',
        '| 完整文件 CR | **4.516** | **6.433** |',
        '',
        '这次两边公共元数据都是 143 B，负例已不能归因于只给无损免除必要元数据。对这个输入，JointZip 完整文件比无损大 42.45%。',
        '',
        '## 核验与证据',
        '',
        '- 实际保存 3,196 个完整文件，包括 172 个无损文件和 3,024 个 JointZip 文件，覆盖全部条件且无重复或缺失。',
        '- 172 个新无损文件均从文件本身独立解码，与冻结 Float32 输入和旧无损重建逐位一致。归一化参数与有效长度已计入文件。',
        '- 3,024 个 JointZip 新文件均能还原为原 JZF1 完整文件，逐字节一致。原码流、修复数据和元数据保持不变，沿用其已绑定文件哈希的独立解码与约束验收证据。本次没有重新运行全部 JointZip 原生解码器。',
        '- 独立复核核对了全部实文件、公共元数据、冻结输入列表和汇总结果。18 组宏平均与胜平负重算差异为零。另行核对了条件唯一性、设置和内部后端身份。',
        '',
        '此次结果替代先前 payload-only 比较用于回答完整文件大小问题。原先 581 个更低属于另一计费口径，不应继续与这里的 524 个混用。没有新增下游质量或编码速度实验。',
        '',
        '## 明细文件',
        '',
        '- [逐配置完整大小与 CR](conditions.csv)',
        '- [主汇总表](macros.csv)',
        '- [loose、medium、strict 与 stress 分表](settings.csv)',
        '- [各来源分表](sources.csv)',
        '- [计费协议](PROTOCOL.json)',
        '- [独立复核](INDEPENDENT_REVIEW.json)',
        '- [条件与后端身份复核](ROSTER_IDENTITY_REVIEW.json)',
        '- [无损文件清单](lossless_files.json)',
        '- [完成记录](COMPLETE.json)',
        '',
    ]
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'code_sha256': {}, 'result_sha256': {}}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        manifest['code_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['PROTOCOL.json', 'conditions.csv', 'macros.csv', 'settings.csv', 'sources.csv',
                 'SUMMARY.json', 'COMPLETE.json', 'INDEPENDENT_REVIEW.json', 'ROSTER_IDENTITY_REVIEW.json', 'REPORT_ZH.md']:
        manifest['result_sha256'][name] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    (OUT / 'REPORT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT / 'REPORT_ZH.md'))


if __name__ == '__main__':
    main()
