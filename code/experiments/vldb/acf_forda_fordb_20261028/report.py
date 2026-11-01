"""Produce the complete, unfiltered Chinese report and scientific bar charts."""
from .common import *

def generate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result=json.loads((RUN/'RESULTS.json').read_text())
    protocol=json.loads((RUN/'PROTOCOL.json').read_text())
    downstream=result['downstream'];mechanism=result['mechanism'];contrasts=result['contrasts']
    assert len(downstream)==42 and len(mechanism)==12 and len(contrasts)==24
    index={(r['dataset'],r['backend'],r['arm'],r['consumer']):r for r in downstream}
    groups=[(d,b) for d in DATASETS for b in BACKENDS]
    figure,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True,constrained_layout=True)
    labels=['ER','ERA: ACF <= 0.05','ERA: ACF <= 0.01'];colors=['#7f8c8d','#5ba3d0','#bd477c']
    for ax,consumer,title in zip(axes,['catch22','spectrum32'],['catch22 (22 features)','Relative spectral energy (32 bands)']):
        for j,arm in enumerate(ARMS):
            values=[100*index[d,b,arm,consumer]['balanced_accuracy'] for d,b in groups]
            ax.bar(np.arange(4)+(j-1)*.24,values,width=.23,color=colors[j],label=labels[j])
        raw=[100*index[d,'Raw','Raw',consumer]['balanced_accuracy'] for d,b in groups]
        ax.scatter(np.arange(4),raw,marker='_',s=550,color='black',linewidths=1.8,label='Raw')
        ax.set_xticks(range(4),[d+'\n'+b.upper() for d,b in groups]);ax.set_title(title,fontsize=11)
        ax.set_ylim(45,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Balanced accuracy (%)');axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    figure.savefig(RUN/'acf_ford_classification_bars.pdf')
    figure.savefig(RUN/'acf_ford_classification_bars.png',dpi=220)
    plt.close(figure)
    lines=['# FordA / FordB：ACF 约束的补充实验', '',
      '本实验检验在固定误差与 Pearson correlation 约束后，增加 ACF 约束是否保留更多可用于分类的时序信息。完整保留两个数据集、两个后端、两档 ACF 阈值及所有官方 TEST 记录。', '',
      '## 动机与固定协议', '',
      'Lubba 等（DMKD 2019，catch22，第 3.4 节 / Figure 6）指出 FordA/FordB 的类别差异与低频内容有关，CO_FirstMin_ac 是表现突出的单特征。这提供了事先选择数据的依据，不预设 JointZip 一定改进分类。文献：https://doi.org/10.1007/s10618-019-00647-x 。', '',
      '- 官方划分：FordA TRAIN 3601 / TEST 1320，FordB TRAIN 3636 / TEST 810；每条序列 500 点。FordB 保留其测试工况噪声变化。',
      '- 后端：SZ3 与 ZFP，分别代表预测与变换类；使用现有版本、公式搜索预算和 simple correction。',
      '- E = 1.0，R = 0.99；ER 不约束 A，ERA 分别使用 A = 0.05 和 0.01。所有序列仅按官方已标准化值转为 float32。',
      '- TRAIN 上按传统 ACF 首极小值及其右邻点的 99% 覆盖规则选择共同 L = '+str(protocol['max_lag'])+'；未使用 TEST 选择阈值、特征或分类器。',
      '- 主消费者：固定 pycatch22 0.4.5 的全 22 特征；互补消费者：32 个等宽非 DC 频带的归一化 periodogram 能量。两者均只在 raw TRAIN 拟合中位数插补、标准化及固定 C=1 的平衡 LogisticRegression；测试重建不重训。首极小值单特征仅作机制诊断。',
      '- 完整 JZF1 文件按每个数据集 / 后端 / 臂组装、独立解码并重新计算 E/R/A。CR = 原始 float32 字节 / 完整 JZF1 字节，包含元数据与 correction；批次检查点开销不计入存储格式。',
      '- 2130 条 TEST × 2 后端 × 3 臂 = 12780 个压缩窗口；分类记录相关，不能当作 12780 个独立样本。缓存和并行耗时不用于编码速度比较。', '',
      '## 全部下游结果', '',
      'BA 为类别平衡准确率；Accuracy 为准确率。表中均为百分比，AUROC 为 0–1。', '',
      '| 数据 | 后端 | 臂 | 消费者 | BA % | Accuracy % | AUROC | 相对 Raw 预测翻转数 |',
      '|---|---|---|---|---:|---:|---:|---:|']
    for r in downstream:lines.append(f"| {r['dataset']} | {r['backend']} | {r['arm']} | {r['consumer']} | {100*r['balanced_accuracy']:.3f} | {100*r['accuracy']:.3f} | {r['AUROC']:.5f} | {r['flip_vs_raw']} |")
    lines+=['','## ER → ERA 的配对变化','','BA 变化为百分点，正值表示提升。95% 区间为固定 TEST 内按真实类别分层的成对 bootstrap（5000 次），仅作描述；没有多重比较校正，也不代表其他数据集上的泛化保证。改善 / 持平 / 下降按同一 TEST 记录是否分类正确计数。','','| 数据 | 后端 | 消费者 | ERA 阈值 | ΔBA pp | 描述性 95% 区间 | 改善 / 持平 / 下降 |','|---|---|---|---|---:|---|---|']
    for r in contrasts:lines.append(f"| {r['dataset']} | {r['backend']} | {r['consumer']} | {r['after']} | {r['balanced_accuracy_change_pp']:+.3f} | [{r['descriptive_BA_CI95_low_pp']:+.3f}, {r['descriptive_BA_CI95_high_pp']:+.3f}] | {r['label_improved']} / {r['label_tied']} / {r['label_declined']} |")
    lines+=['','## 存储成本与机制','','A 为 JointZip 的切片 Pearson ACF 最大误差；传统 ACF 列使用全局均值及 lag-zero 归一化，二者定义不同。','','| 数据 | 后端 | 臂 | CR | 完整字节 | 最大 E | 最小 R | 最大 A | 数值通过 / 总数 | 无损窗 | 修复点 | 修复载荷字节 | 传统 ACF MAE | 首极小 lag MAE |','|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in mechanism:lines.append(f"| {r['dataset']} | {r['backend']} | {r['arm']} | {r['CR']:.4f} | {r['full_bytes']} | {r['max_E']:.6f} | {r['min_R']:.6f} | {r['max_slice_ACF']:.6f} | {r['enabled_numerical_pass']} / {r['n']} | {r['lossless_windows']} | {r['repair_values']} | {r['correction_payload_bytes']} | {r['traditional_ACF_MAE']:.6f} | {r['first_minimum_MAE']:.4f} |")
    main=[r for r in contrasts if r['consumer']!='first_minimum']
    npos=sum(r['balanced_accuracy_change_pp']>1e-12 for r in main);nneg=sum(r['balanced_accuracy_change_pp']<-1e-12 for r in main)
    lines+=['','## 解读与边界','',f'预先声明的两个实际分类消费者共 16 个 ER→ERA 比较，BA 有 {npos} 个改善、{16-npos-nneg} 个持平、{nneg} 个下降。各比较及其存储成本均在上表保留。不能从机制误差减小直接推出标签准确率一定提高。', '',
      '本实验隔离增加 A 的效果，但不匹配 CR；更强约束可以要求更多存储，可能通过更精细的原生结果、修复或无损回退达标。两种特征表示不是统计独立的验证。满足 JointZip 的切片 ACF 容差并不保证传统 ACF 的首极小值完全不变，也不保证任意未知下游任务提升。', '',
      'FordA/FordB 根据文献中的时序结构选择，因此是有动机的针对性补充，不是随机数据集总体估计。两档阈值及两后端全部报告，不按测试效果选择子集。', '',
      '![全部主要分类结果](acf_ford_classification_bars.png)', '',
      '完整机器可读结果：downstream.csv、contrasts.csv、mechanism.csv、RESULTS.json；参数与依赖冻结：PROTOCOL.json；每窗口约束与首极小误差：artifacts/*_windows.csv。']
    (RUN/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':generate()
