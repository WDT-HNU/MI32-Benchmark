# MI32 Benchmark 技术导读

MI32 Benchmark 围绕三个 EEG 分类任务组织：运动想象三分类、运动想象左手/右手二分类和 P300
二分类。每个任务都使用三类方法进行评测：任务专用深度模型、EEG 基础与长序列模型，以及黎曼
几何方法。

这份导读先说明任务，再说明方法。仓库中的 `deep_learning/` 和 `riemannian/` 是保存方法材料的
文件夹，不是两套彼此独立的数据分支。

## 项目的完整逻辑

```text
EEG Benchmark
├── 任务一 MI 三分类
│   ├── 任务专用深度模型
│   ├── EEG 基础与长序列模型
│   └── 黎曼几何方法
├── 任务二 MI 左右手二分类
│   ├── 任务专用深度模型
│   ├── EEG 基础与长序列模型
│   └── 黎曼几何方法
└── 任务三 P300
    ├── 任务专用深度模型
    ├── EEG 基础与长序列模型
    └── 黎曼几何方法
```

这种组织方式回答两个不同问题：任务轴说明“要识别什么”，方法轴说明“用什么方式识别”。把方法
轴误写成两个数据分支，会让人误以为深度模型没有运行 MI2 和 P300，也会掩盖同一任务内不同方法
协议是否真正可比的问题。

## 三个任务

### MI 三分类

MI3 使用 `left_upper`、`right_upper` 和 `non_upper` 三个标签。MI32 common-32 v4 汇集 8 个
来源、230 名受试者和 97,608 个 trial，三类数量为 24,402、24,402 和 48,804。信号统一为
32 通道、250 Hz、3 秒、单位 V。

缺失目标电极依据模板几何采用 Perrin 球面样条插值，`measured_mask` 保留实测与插值身份。受试者
划分冻结后，模型不得重新拆分数据。

### MI 左手右手二分类

深度模型使用的 MI2 是从 MI3 冻结 release 中只保留 `left_upper` 与 `right_upper` 得到的派生
视图，共 230 名受试者和 48,804 个 trial。派生过程不重复滤波、重采样、重参考或插值，因此它
继承 MI3 的信号几何和受试者身份。

黎曼方法交付材料中的 Binary MI 使用更大的 18-source 冻结 release：389 名受试者、59,216 个
trial、84 个完整受试者独立 folds，以及 36 个代表 folds。二者属于同一任务定义，但不是同一
冻结数据版本。结果可以放在同一任务章节里说明，不能省略 release 和 split 后直接排名。

### P300

P300 是 Target/NonTarget 的 time-locked ERP 任务。深度模型数据汇集 15 个来源和 158,667 个
trial，统一为 16 通道、200 Hz、刺激后 0–600 ms，共 121 个真实采样点。当前深度模型协议从
完整 873 个 LOGO folds 中按预先固定的抽样规则选择 102 folds，7 个模型共完成 714 次运行。

黎曼方法使用 reduced frozen LOGO 协议，当前方法状态表以 85 folds 为分母。P300 不能把 MI 的
plain spatial covariance 机械套用；TSA、MEKT-R 等方法需要 ERP-aware 表示。

## 三类方法

### 任务专用深度模型

这一组包括 EEGNet、TSception、RGNN 和 EEG-Conformer。它们分别代表轻量卷积、多尺度时空卷积、
图神经网络和卷积 Transformer。RGNN 当前按 backbone 或 surrogate 报告，因为现有特征流程不等同
于完整官方 DE+LDS、NodeDAT 和 EmotionDL 流程。

### EEG 基础与长序列模型

这一组包括 LaBraM、EEGMamba、CodeBrain 和 Uni-NTFM。LaBraM、EEGMamba 与 CodeBrain 已在现有
任务协议中留下结果；Uni-NTFM 保留结构与协议路线，但没有把缺少可接受下游权重和分类头契约的
运行写成正式结果。

这类模型对单位、采样率、patch 排列、pooling 和下游 head 很敏感。模型名称相同并不自动代表
输入契约相同，因此每次运行都要保存 adapter manifest、源码身份和权重身份。

### 黎曼几何方法

当前整理 MDRM、TSLDA、KernelSVM、FGMDRM、RA、SPDNet、MFACC、RPA、TSA、
Manifold-SimCLR 和 MEKT-R。它们使用 covariance、SPD 流形、切空间、对齐或迁移学习，但允许访问
的目标域信息不同：

| 信息条件 | 含义 | 代表方法 |
|---|---|---|
| `SOURCE_ONLY` | 只使用训练数据和训练标签拟合 | MDRM、TSLDA、KernelSVM、FGMDRM、SPDNet、MFACC |
| `TRANSDUCTIVE_UNLABELED_TARGET` | 可用无标签目标域统计，不可用目标标签 | RA、MEKT-R |
| `TARGET_CALIBRATION_LABELED` | 只用预先冻结的目标校准标签 | RPA、TSA |

允许访问无标签目标数据不等于标签泄露，但必须在结果中明确披露信息条件。

## 当前实验矩阵

| 任务 | 任务专用深度模型 | EEG 基础与长序列模型 | 黎曼几何方法 |
|---|---|---|---|
| MI3 | EEGNet、TSception、RGNN、EEG-Conformer 已有当前结果 | LaBraM、EEGMamba、CodeBrain 已有当前结果；Uni-NTFM 为协议路线 | 05 三分类代表协议；多数方法 16/16，MFACC 13/16，MEKT-R 为 binary-only |
| MI2 | 4 个模型均完成 fold 0 | LaBraM、EEGMamba、CodeBrain 完成 fold 0 | 代表 36 folds；按方法保存 formal、partial 或 blocked 状态 |
| P300 | 4 个模型均完成 102 sampled LOGO folds | LaBraM、EEGMamba、CodeBrain 完成同一 102 folds | reduced 85-fold protocol；保留 TSLDA 等真实失败 |

所以，深度学习方法并不是只跑三分类。MI2 和 P300 都有深度模型结果；以前的文档只是没有把这些
已经完成的工作纳入总树。

## 数据和方法之间怎样连接

公开 EEG 数据在采样率、通道、单位、事件码、窗口和受试者层级上并不一致。本项目先冻结公共数据，
再让方法 adapter 完成必要转换：

```text
原始数据与来源证据
→ 任务级统一数据
→ 冻结受试者或 LOGO split
→ 方法专用 adapter
→ 模型或方法核心
→ prediction
→ 独立重算 metrics
→ manifest 与 coverage
```

adapter 不负责根据测试分数寻找更有利的预处理。需要拟合的均值、方差、参考点或校准量，只能使用
相应协议允许的数据。

## 怎样避免信息泄露

深度模型中，同一受试者不会同时进入训练、验证和测试；学习率等超参数只依据验证集选择；测试集
在配置冻结后用于最终评估；标准化和校准统计只从训练部分得到。

黎曼方法除 source-only 外，还存在无标签目标域适配和带标签目标域校准。判断是否泄露不能只看
“是否读取目标数据”，而要核对原方法的信息契约、目标标签是否进入拟合，以及 calibration 与
evaluation 是否预先分开。

## 结果应该怎样读

一个数字至少要同时带着五项信息：任务、冻结数据 release、split、方法实现身份和 coverage。

当前深度结果的范围是：

- MI2：7 个模型的 fold 0；
- MI3：模型覆盖不齐，EEGNet、TSception、RGNN、EEG-Conformer 为 fold 0，LaBraM 为 folds 0/2，EEGMamba 与 CodeBrain 为 folds 0/2/3；
- P300：7 个模型的 102 sampled LOGO folds。

当前黎曼结果按任务分别报告 36、16 和 85 folds 的状态与 coverage。`PARTIAL` 表示真实方法或数值
失败被保留，不是删除失败 fold 后只报告成功部分。

三个任务不能平均成一个总分。MI3、MI2 和 P300 的类别、试次单位、折协议与主指标不同；同一任务
内若数据 release 或 split 不同，也不能把分数当作严格配对结果。

## 仓库中的证据

一次可审计运行通常包括结果 CSV、逐试次预测、run manifest、数据清单 SHA-256、训练与测试身份、
随机种子、训练参数、源码提交和权重哈希。模型名称本身不能替代这些内容。

仓库按用途存放文件：

```text
MI32-Benchmark/
├── benchmark/
│   ├── datasets/          任务数据契约与元数据
│   ├── model_adapters/    深度模型输入适配
│   ├── runners/           训练和评测程序
│   ├── tests/             结构、数据语义和泄露门禁
│   └── results/           结果与运行证据
└── docs/
    ├── deep_learning/     深度模型和基础模型
    ├── riemannian/        黎曼方法
    └── project/           获取、复现与发布说明
```

推荐阅读顺序：

1. [数据索引](../benchmark/datasets/README.md)，先确认任务和冻结数据版本；
2. [深度学习方法](deep_learning/README.md)或[黎曼几何方法](riemannian/README.md)，核对方法输入；
3. 结果说明与 method status，确认 coverage；
4. [复现说明](project/REPRODUCIBILITY.md)，开始环境检查和运行。

## 最短复现路径

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark/benchmark

python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

正式运行前还需要取得有权使用的信号包、固定版本的上游源码和权重。每次新增结果都应先固定任务、
release、split 和 adapter，再运行模型；不能用测试集反过来决定这些设置。
