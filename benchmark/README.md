# Benchmark 工作区

这里集中保存真正参与运行和验证的内容。进入本目录后，再沿着“数据 → 方法 → 实验 → 结果”阅读：

```text
benchmark/
├── datasets/          两套数据契约和 MI32 元数据
├── dataset_tools/     32 通道数据构建与检查
├── model_adapters/    8 种深度模型的输入适配
├── runners/           训练与评测程序
├── configs/           数据、模型和评测固定配置
├── scripts/           下载、检查、运行、汇总和发布入口
├── tests/             结构、数据语义和泄露门禁
├── results/           已收录结果和运行证据
├── requirements/      环境依赖
└── THIRD_PARTY_LICENSES/  上游代码许可证原文
```

## 从哪里开始

| 目标 | 入口 |
|---|---|
| 比较两套数据 | [datasets/README.md](datasets/README.md) |
| 查看 8 种深度模型 | [项目文档](../docs/deep_learning/README.md) |
| 查看 11 种黎曼方法 | [项目文档](../docs/riemannian/README.md) |
| 运行模型 | [runners/README.md](runners/README.md) |
| 查看当前结果 | [结果说明](../docs/deep_learning/RESULTS.md) |

所有运行命令均以本目录为工作目录：

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

返回[项目首页](../README.md)。
