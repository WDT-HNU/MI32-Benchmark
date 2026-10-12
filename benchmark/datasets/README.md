# 数据索引

项目包含三个任务：MI 三分类、MI 左手/右手二分类和 P300。方法类别不决定数据类别；任务专用深度
模型、EEG 基础模型和黎曼几何方法都可以在这三个任务上评测。

## 三个任务

| 任务 | 标签 | 主要冻结数据说明 | 典型输入 |
|---|---|---|---|
| MI3 | `left_upper` / `right_upper` / `non_upper` | MI32 common-32 v4：8 个来源、230 人、97,608 trials | `[32,750] @ 250 Hz` |
| MI2 | Left / Right | 深度模型使用 MI3 的 left/right 派生视图：230 人、48,804 trials | `[32,750] @ 250 Hz` |
| P300 | Target / NonTarget | 15 个来源、158,667 trials；16 通道、0–600 ms ERP | `[16,121] @ 200 Hz` |

## 为什么还会看到另一套 Binary MI

黎曼方法的 Binary MI 交付材料使用一个更大的冻结协议：18 个来源、389 名受试者、59,216 trials，
同样整理为 `[32,750] @ 250 Hz`，完整协议 84 folds，代表协议 36 folds。它与深度模型的 MI2
派生视图属于同一个“左手/右手二分类任务”，但不是同一个冻结 release，不能直接把两边分数当成
严格配对比较。

| 数据入口 | 用途 |
|---|---|
| [MI32 数据卡片](mi32/DATASET_CARD.md) | MI3 主 release 及其 MI2 派生视图的来源、通道和划分依据 |
| [Binary MI 数据卡片](binary_mi/DATASET_CARD.md) | 黎曼 Binary MI 的 18-source 冻结契约与代表协议 |

同理，P300 深度模型的 102-fold sampled LOGO 与黎曼方法的 reduced 85-fold protocol 都属于 P300
任务，但结果必须带着各自的 split identity 解读。任务可以统一组织，数据版本和评测协议不能省略。
