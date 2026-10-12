# 黎曼几何方法汇总

本页按已交付的方法摘要整理。状态字段沿用该摘要的任务和 coverage 口径；方法之间的比较必须同时
保留任务、冻结 split、information regime、有效 fold 数和 fidelity status。

## 总体方法逻辑

EEG 试次 `X ∈ R^(p×T)` 可以转换为空间协方差矩阵。严格正定的协方差属于 SPD 流形，直接采用
逐元素均值、欧氏距离或线性插值会忽略其几何结构。AIRM 距离、Karcher 均值和 tangent-space
mapping 分别提供流形上的距离、参考点和局部线性化表示。

本分支采用的共同边界是：

```text
冻结数据与 split
└── 数据集/方法专用 adapter
    └── 官方方法输入契约
        └── 冻结方法核心
            └── 评测
                └── 同时报告任务、信息边界、coverage 和状态
```

adapter 必须跟随方法：plain centred SCM、official OAS covariance、P300 的 ERP prototype/
augmented covariance 不能互相冒充。允许使用无标签目标域统计的方法，也不能被强行改写成
source-only；需要目标域带标签校准的方法，则必须预先冻结 calibration/evaluation 分区。

Binary MI 的共享输入为 `(B, 32, 750)`、250 Hz、3 秒、V、8–30 Hz，标签为 `0=Left`、
`1=Right`。完整数据说明见[Binary MI 数据契约](../../datasets/binary_mi/DATASET_CARD.md)。

## 方法树

```text
黎曼几何方法
├── 直接分类
│   ├── MDRM：类黎曼均值 + AIRM 最近均值
│   ├── TSLDA：切空间 + 变量选择 + LDA
│   ├── KernelSVM：切空间黎曼核 + 线性 SVM
│   ├── FGMDRM：Fisher geodesic filter + MDM
│   ├── SPDNet：SPD 专用神经网络层
│   └── MFACC：OAS + ADCSP + means field classifier
├── 跨域对齐/迁移
│   ├── RA：无标签目标域黎曼对齐
│   ├── RPA：带标签目标域校准的 Procrustes 分析
│   ├── TSA：带标签目标域切空间对齐
│   └── MEKT-R：无标签目标域分布对齐与伪标签迭代
└── 自监督表示
    └── Manifold-SimCLR：SPD/log-domain 对比学习
```

## 方法卡片

### MDRM

- 依据：Barachant et al., 2012。
- 表示：centred SCM；只有未通过 SPD 数值门禁时才做 scale-relative 最小谱修复。
- 流程：covariance → 各类 Karcher mean → AIRM distance → 最近均值。
- 信息：`SOURCE_ONLY`；类均值只由训练数据估计。
- 状态：Binary 36/36、05 16/16、P300 85/85 均为 formal。

### TSLDA

- 依据：Barachant et al., 2012。
- 流程：训练集黎曼均值 → log map → `sqrt(2)` 对称上三角向量化 → SVD/PCA → ANOVA →
  weighted FDR (`q=0.05`) → LDA。
- 信息：`SOURCE_ONLY`，参考点、降维、特征选择和 LDA 均 train-only。
- 边界：weighted-FDR 选中零变量是真实方法失败，不增加 fallback 或伪造预测。
- 状态：Binary 32/36 valid；05 16/16；P300 35/85 valid。

### KernelSVM

- 依据：Barachant et al., 2013。
- 流程：centred SCM → 训练集 AIRM mean → tangent map → `sqrt(2)` 向量化 → linear SVC。
- 固定约定：`SVC(kernel="linear", C=10)`，不做性能驱动的网格搜索。
- 信息：`SOURCE_ONLY`；05 已检查原生多分类兼容性，AUC 使用连续 decision score。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal。

### FGMDRM / FgMDM

- 依据：Barachant et al., 2010；2019 SSDT-FGMDRM work。
- 流程：AIRM mean → tangent map → Fisher/FGDA filtering → inverse tangent map → MDM。
- 信息：`SOURCE_ONLY`；Fisher 投影只在训练向量和标签上拟合。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal。

### RA

- 依据：Zanini et al., 2018。
- 流程：source/target 分别估计黎曼 reference 并重心化，再由 source classifier 预测。
- 信息：`TRANSDUCTIVE_UNLABELED_TARGET`；目标域协方差可参与 reference，目标标签只在预测后算指标。
- 边界：论文定义的 target self-inclusion 必须披露，但不等同于标签泄露。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal。

### SPDNet

- 依据：Huang & Van Gool, 2017。
- 结构：`BiMap → ReEig → BiMap → ReEig → BiMap → LogEig → vectorization → softmax`，
  验证结构为 `32→16→8→4`。
- 优化：BiMap 权重采用 Stiefel/Riemannian gradient 与 QR retraction；ReEig/LogEig 使用
  author-style spectral backward。
- 信息：`SOURCE_ONLY`。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal；BiMap 参数更新已审计。

### MFACC

- 表示：raw frozen epoch 经 official OAS covariance route。
- 流程：`Covariances(estimator="oas") → ADCSP → MFACC(method_label="lda")`。
- 信息：`SOURCE_ONLY`；不以共享 SCM cache 或人为 SPD repair 替代官方 OAS 路径。
- 状态：Binary 36/36、P300 85/85 formal；05 为 13/16 valid，3 个数值失败单独保留。

### RPA

- 依据：Rodrigues et al., 2019。
- 流程：recentring → stretching → class-wise rotation → source-trained classifier。
- 信息：`TARGET_CALIBRATION_LABELED`；只允许预冻结的目标域 calibration labels，evaluation labels
  仅用于最终指标。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal。

### TSA

- 依据：Bleuzé, Mattout & Congedo, 2022。
- 流程：source/target tangent representation → centering/rescaling → class anchors →
  SVD/Procrustes rotation → source SVC。
- 信息：`TARGET_CALIBRATION_LABELED`；evaluation labels 不进入 alignment。
- 边界：P300 使用 ERP-aware representation，而不是 MI plain SCM。
- 状态：Binary 36/36、05 16/16、P300 85/85 formal。

### Manifold-SimCLR

- 方向：SPD/log-domain augmentation → encoder/projection → contrastive loss → downstream probe/
  fine-tuning。
- 信息：`SELF_SUPERVISED_SOURCE_ONLY`，前提是存在完整可执行的原方法定义。
- 边界：现有摘要将 Binary 36-fold 结果标为 reconstructed exploratory；MXOptimizer、exact backbone
  和 joint fine-tuning schedule 尚不能由作者材料唯一确定，因此不写成 paper-faithful formal result。
- 状态：Binary `PARTIAL_RECONSTRUCTION`；05 与 P300 `IMPLEMENTATION_BLOCKED`。

### MEKT-R

- 依据：Zhang & Wu, 2020。
- 流程：subject alignment → official MTS source concatenation → joint-probability-MMD/generalized
  eigenproblem → 5 次伪标签迭代 → official shrinkage LDA。
- 固定约定：`p=10`、`T=5`、`alpha=0.01`、`beta=0.1`、`rho=20`、target KNN `k=10`、
  heat-kernel `sigma=1`。
- 信息：`TRANSDUCTIVE_UNLABELED_TARGET`；真实目标标签不进入适配、伪标签或 stopping rule。
- 边界：P300 使用 ERP/RSVP-aware route；05 的官方 endpoint 为 binary-only，不自行增加 OvR/OvO。
- 状态：Binary 36/36、P300 85/85 formal；05 `BINARY_ONLY`。

## 信息使用边界

| Information regime | 方法 | 允许拟合的信息 | 防泄露边界 |
|---|---|---|---|
| `SOURCE_ONLY` | MDRM、TSLDA、KernelSVM、FGMDRM、SPDNet、MFACC | source/train EEG、covariance、labels | reference、selector、classifier 均 train-only |
| `TRANSDUCTIVE_UNLABELED_TARGET` | RA、MEKT-R | source labels；target EEG/covariance/statistics | target labels 仅在预测后用于指标 |
| `TARGET_CALIBRATION_LABELED` | RPA、TSA | source data；冻结 calibration subset labels | evaluation labels 不进入对齐或训练 |
| `SELF_SUPERVISED` | Manifold-SimCLR | 按原协议允许的 source-side data | evaluation labels 不参与预训练或选参 |

是否允许访问目标域信息由论文方法契约决定。无标签目标域统计不等于 label leakage；带标签校准也不
能与 source-only 结果在不披露协议的情况下直接并表。

## 三类任务

### Binary Left/Right MI

18 个来源数据集、389 名受试者、59,216 个试次。完整协议 84 folds，代表协议每个数据集抽取两个
既有 fold，共 36 folds。主要表示为空间协方差与 MI 频带信息。

### 05 three-class MI

独立的 8-dataset × 2-fold 代表协议，共 16 folds。必须逐方法检查多分类 endpoint；不能把
binary decision rule 静默扩展。KernelSVM、MFACC 已按摘要记录为支持多分类，MEKT-R 保持
`BINARY_ONLY`。

### P300

P300 是 time-locked ERP 任务，不应机械压缩为 MI-style plain spatial SCM。冻结 reduced LOGO
协议为 85 folds。TSA 和 MEKT-R 使用 ERP-aware route；TSLDA 保留真实 zero-feature failures；
SPDNet 保留经过审计的 spectral backward 和 Stiefel/BiMap update。

## 状态汇总

| Method | Binary MI（36） | 05 三分类 MI（16） | P300 reduced LOGO（85） |
|---|---|---|---|
| MDRM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| TSLDA | PARTIAL 32/36 | FORMAL 16/16 | PARTIAL 35/85 valid |
| KernelSVM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| FGMDRM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| RA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| SPDNet | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| MFACC | FORMAL 36/36 | PARTIAL 13/16 | FORMAL 85/85 |
| RPA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| TSA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| Manifold-SimCLR | PARTIAL_RECONSTRUCTION 36/36 | IMPLEMENTATION_BLOCKED | IMPLEMENTATION_BLOCKED |
| MEKT-R | FORMAL 36/36 | BINARY_ONLY | FORMAL 85/85 |

`FORMAL` 表示摘要记录了按冻结协议完成的结果；`PARTIAL` 保留真实方法或数值失败；
`BINARY_ONLY` 和 `IMPLEMENTATION_BLOCKED` 是 fidelity/scope 状态，不是性能分数。

## 交付摘要记录的工作产物路径

以下路径保留原工作区命名，用于理解材料之间的关系：

| 作用 | 原工作区相对路径 |
|---|---|
| Binary MI 完整 split | `evaluation/within_dataset_splits.json` |
| Binary MI 代表 fold | `evaluation/representative_2fold_subset.json` |
| Adapter 总报告 | `experiments/MASTER_ADAPTER_BENCHMARK_REPORT.md` |
| MEKT-R 规范与报告 | `experiments/MEKT_adapter/MEKT_METHOD_SPEC.md`、`MEKT_FINAL_REPORT.md` |
| P300 benchmark | `P300_benchmark/` |
| 05 三分类 benchmark | `05_benchmark/` |
| P300 fidelity repair | `P300_benchmark/experiments/*_fidelity_repair/` |
| 05 多分类兼容性审计 | `05_benchmark/experiments/MEKT_R_05_MULTICLASS_COMPATIBILITY_AUDIT.md` |

这些路径记录 method/input/protocol provenance。跨任务汇总时仍需保留任务、split、信息边界、
coverage 和 fidelity status。
