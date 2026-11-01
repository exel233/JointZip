# JointZip 单后端扩展：SZ2016

状态：已授权，待五后端完整实验及独立计时、导出全部完成后串行执行。不得与上一轮独立计时并行计算。

## 目的与选择理由

在固定数据、阈值、消费者、公式候选和原简单 correction 的条件下，检验框架能否接入第六个后端，并比较加入框架前后的任务表现及存储代价。SZ2016 指官方 SZ 1.1 的历史实现，不是 SZ3，也不标为当前 SOTA。选择依据是原生绝对误差接口、真实一维支持、已有全量原生结果可复用，以及与当前后端不同的预测机制。

已有预测 VAL 汇总显示其原生结果在 strict/medium/loose 下均未全窗通过 ERA，说明有统计保护的检验空间；这不是下游提升的保证。数据已被本项目使用，因此本次属于探索性扩展，不能称独立未见数据的确认实验。不得选择有利 TEST 子集、改变原生设置使其变差，或删除负面结果。

## 不变的科学设置

- 数据来自 `vldb_simple_correction_20260929_v1` 的原冻结输入：142 条预测序列、6 个 UCR 数据集、24 个 NASA 文件，共 172 个输入。
- 两种方法：`sz2016_native` 与 `sz2016_jointzip`。每种 930 条压缩配置，共 1,860 条。原生使用一次用户 epsilon；框架使用当前公式候选，每窗最多 22 次、初始 12 次、divisor=512，沿用五后端实验 SZ3/ZFP/SPERR 的调度，不加入旧网格兜底。
- strict: epsilon=0.1、rho=0.999、tau=0.01；medium: 0.25、0.995、0.02；loose: 0.5、0.99、0.05。UCR stress: 1、0.99、0.05。lag=1–16。
- Pearson 是原始窗口与重建窗口的相关系数；ACF 是各自 lag 切片 Pearson 的差。公式仅指导候选锚点与参数搜索，最终解码、修复、验收提供约束保护，不声称锚点是非线性 ACF 的确定性充分保证。
- correction 保持初始 `posthoc_greedy`、`legacy`、原值及 7/15/31/63/127 修复目标，格式 0–6。保留共同无损回退。不得增加后期复杂编码。
- 两种方法使用同一 JZF1/JZP1 完整文件计费，包含归一化、通道、真实长度及纠正字节。NASA `entry.true_length` 已是 TEST 后缀长度，不再扣除 train_end。
- 同一 epsilon 比较不等价于同一 CR 比较。原生 ERA 不通过是合法评测结果；框架 ERA 数值验收必须通过。

## 官方实现与适配

沿用已使用的 `tools/sz2016/bin/sz2016_batch` 与 `tools/sz2016/sz2016.config`，官方源为 https://github.com/szcompressor/sz2 ，固定提交 `2123f4899b76e6dfe28861da0a70d63a6cd43f83`。已有构建含一处已披露的 zlib 输出缓冲区 `compressBound` 分配补丁；不能称源码逐字未修改。新适配器只拆装一维原生码流、调用原二进制与原配置，不重写压缩算法。

新的生产文件位于 `experiments/vldb/framework_sz2016_20261016`，与运行中的 `framework_full_20261015` 分开。官方二进制和原 correction、anchors 在独立 runtime 中复制一次。保留历史实现和结果。

实现局限沿用 `tools/sz2016/README.md` 的披露：固定历史提交的 best-fit constructor 在完整 ASan 编码检查中有已知 use-after-free；当前为已有实验使用的发布构建，不宣称上游完整内存检查通过。每个结果仍要求独立码流重放与数值验证；若本轮出现压缩故障则停止保留日志，不忽略失败，也不临时修改官方算法以完成测试。

## 复用与消费者

930 条历史原生压缩配置复用原码流，并按本轮容器重封装及独立解码；不照搬旧 CR。输入、模型及运行条件一致且 Raw replay 通过时，复用历史预测。其余计算新结果；完整 row 和 window checkpoint 支持恢复。实际编码耗时另做独立计时，复用不是零成本编码。

固定消费者与五后端实验一致：预测 Ridge、DLinear、PatchTST 三种子、Chronos2、ChronosBoltSmall、TimesFM3；分类 1NN、Logistic、ResNet v1/v2 三种子；检索 shape 与两个 ResNet 嵌入；异常 ridge_svd64、KNN、LSTM 三种子。先完成所有 forecast VAL，再 TEST，不以 TEST 选择政策。

预计消费者记录 17,014：forecast 15,904、classification 144、retrieval 126、anomaly 840，含 Raw、VAL、种子与复用；不是 17,014 个独立样本或全新模型调用。

## 计时与汇报

独立计时采用同六个预测来源的首条序列、前四个 VAL 窗口，strict/loose 两阈值。每个 native/framework 条件一遍预热、三遍正式测量，共 96 次调用（24 预热、72 正式），12 个配对条件。交替调用顺序，不与其他科学任务并行。编码包含公式、候选调用、候选解码、原修复及封装，排除末尾额外独立解码审计；完整解码单列。

不重复五后端实验的 144 条搜索密度消融，其证据仅适用于原 SZ3/ZFP 面板，不当作本次 SZ2016 的消融证据。

报告包含完整来源等权结果及逐配置明细：CR、严格/数值 ERA 通过、预测 MSE、分类准确率、检索 mAP、异常 AP/VUS-PR，以及改善/持平/下降和配对编码、解码耗时。区分后端有损、correction 和无损回退；不只展示有利部分。

## 启动、恢复、完成门槛

前置运行 `vldb_framework_full_20261015_v1` 必须 root COMPLETE、closeout passed、density=144、timing=480 和 Windows DELIVERY complete；在此之前新运行只允许静态准备。

运行名 `vldb_framework_sz2016_20261016_v1`；真实结果在 WSL `/opt/jointzip-home/.local/cth_expansion_runs/` 同名目录。隐藏启动器 `tmp/runtime_framework_sz2016_20261016/run_after_full.ps1 -Execute` 先等待前置任务，然后执行独立烟测、完整科学流水线、计时、closeout、报告、导出。失败停止并保留日志，仅可在确认无旧进程后显式 `-Resume`。不自动跳过未知失败。

完成必须有 root COMPLETE、closeout exact 1,860/17,014 且无缺失错误、timing COMPLETE=96、Windows DELIVERY complete。自动跟进应先交付已有五后端完整报告，继续跟进本扩展，全部交付后再暂停。此轮不自动修改 Overleaf。
