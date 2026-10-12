# Binary Left-vs-Right MI 数据契约

本页整理二分类黎曼几何分支已经冻结的数据口径。它保留来源数据集和受试者身份，将可比较的左手、
右手运动想象试次统一到同一输入表示。这里描述冻结后的 benchmark，不重新定义原始公开数据。

## 一眼看懂处理逻辑

```text
18 个公开 MI 数据集
└── 只保留 Left / Right 运动想象试次
    └── 标签统一：0 = Left，1 = Right
        └── 时间统一：3 s × 250 Hz = 750 samples
            └── 通道统一：固定顺序的 32 通道表示
                └── 信号统一：8–30 Hz、单位 V
                    ├── 完整协议：84 个受试者独立 fold
                    └── 代表协议：18 数据集 × 2 个既有 fold = 36 folds
```

## 冻结统计

| 项目 | 冻结值 |
|---|---:|
| component datasets | 18 |
| subjects | 389 |
| trials | 59,216 |
| EEG channels | 32 |
| sampling rate | 250 Hz |
| epoch duration | 3 s |
| samples per trial | 750 |
| trial tensor | `(n_trials, 32, 750)` |
| physical unit | V |
| label `0` | Left |
| label `1` | Right |

| Component dataset | Subjects | Trials | Frozen folds |
|---|---:|---:|---:|
| BNCI2014-001 | 9 | 2,538 | 5 |
| BNCI2014-004 | 9 | 6,520 | 5 |
| Beetl2021-A | 3 | 600 | 3 |
| Beetl2021-B | 2 | 320 | 2 |
| Brandl2020 | 16 | 8,058 | 5 |
| Chang2025 | 28 | 2,260 | 5 |
| Dreyer2023C | 6 | 1,440 | 5 |
| Forenzo2023 | 15 | 2,193 | 5 |
| GuttmannFlury2025-MI | 31 | 2,520 | 5 |
| HefmiIch2025 | 10 | 600 | 5 |
| Kumar2024 | 18 | 7,156 | 5 |
| Lee2019-MI | 54 | 5,400 | 5 |
| Liu2024 | 50 | 2,000 | 5 |
| PhysionetMotorImagery | 109 | 4,918 | 5 |
| Schirrmeister2017 | 7 | 3,248 | 5 |
| Weibo2014 | 10 | 1,580 | 5 |
| Zhou2016 | 4 | 1,199 | 4 |
| Zhou2020 | 8 | 6,666 | 5 |
| **Total** | **389** | **59,216** | **84** |

## 统一规则

### 任务和标签

只保留 Left-hand 与 Right-hand MI。原任务中其余类别不进入该二分类 benchmark；所有方法共享
`0 = Left`、`1 = Right`，标签映射不随方法、fold 或测试表现改变。

### 时间和采样率

每个试次固定为 3 秒，即 250 Hz 下的 750 个采样点。较长 epoch 只从各来源共有的有效时间段
截取；不以补零把短 epoch 凑到 3 秒，以免零段改变协方差的秩和能量分布。不同原始采样率在上游
构建阶段使用带抗混叠处理的重采样。

交付摘要记录的是最终 `250 Hz / 750 samples` 契约，没有重新声明未由原始构建脚本核验的
cue-relative 起止秒数或具体重采样库参数。

### 通道和信号

所有试次采用固定顺序的 32 通道表示。正式契约不使用早期探索过的 156 通道补零方案。冻结信号
限定在 8–30 Hz，采用 Butterworth 带通和零相位处理，单位为 V。

### 数据层与方法适配层

```text
冻结试次 [32,750]、V、8–30 Hz
└── 方法专用 adapter
    ├── centred SCM
    ├── OAS covariance
    └── ERP-aware augmented covariance（P300）
        └── 冻结方法核心
```

共享数据层不替任何方法决定 covariance。多数 SPD/AIRM 方法使用时间中心化 SCM；只有严格 SPD
输入未通过数值门禁时，才进行确定性的最小 PSD→SPD 修复。OAS 或 ERP-specific covariance
属于对应方法的输入契约，不作为全 benchmark 统一替代。

## 文件组织口径

交付材料记录的数据包结构为：

```text
../pooled_mi_riemann_32interp/
├── config.json
├── subjects.csv
└── data/
    └── subject_###.npz
```

| Artifact | 记录内容 |
|---|---|
| `config.json` | 全局数据契约，包括 `n_channels=32`、`n_samples=750` |
| `subjects.csv` | 受试者标识、来源数据集和试次文件相对路径 |
| `data/subject_###.npz` | `X` 试次张量与 `y` 二分类标签 |
| `evaluation/within_dataset_splits.json` | 84 个冻结 train/test 受试者划分 |
| `evaluation/representative_2fold_subset.json` | 36 个代表 fold 及计数审计 |

每个试次的数据集和受试者归属由 subject file 与 `subjects.csv` 关联；运行时不额外假定一个未在
loader 中使用的逐试次 metadata CSV。

## 评测划分

### 完整 84-fold 协议

完整协议为 within-dataset、cross-subject evaluation。训练与测试受试者不重叠：

- Beetl2021-A：3 名受试者，3 个 LOSO folds；
- Beetl2021-B：2 名受试者，2 个 LOSO folds；
- Zhou2016：4 名受试者，4 个 LOSO folds；
- 其余数据集：受试者级 5-fold GroupKFold。

冻结结构覆盖 389 名受试者，每名受试者在所属数据集中恰好一次成为测试对象；记录中无空 fold、
无训练/测试受试者重叠，也无单类别 train/test fold。

### 代表 36-fold 协议

计算成本受限时使用 `18 datasets × 2 already-existing folds`。seed 固定为 `2026`，只从 84 个
已有 fold 中抽取，不重新切分，不改变成员关系，也不读取模型性能。36-fold 是代表性协议，不被
解释为完整 84-fold 评测的等价替代。

对应方法和信息使用边界见[黎曼几何方法汇总](../../../docs/riemannian/METHODS.md)。
