# Benchmark 工作区

这里保存参与运行和验证的内容。科学主线是三个任务，代码和证据则按用途归档。

```text
benchmark/
├── datasets/          MI3、MI2、P300 相关数据契约和元数据
├── dataset_tools/     数据构建与检查
├── model_adapters/    深度模型输入适配
├── runners/           训练与评测程序
├── configs/           数据、模型和评测固定配置
├── scripts/           下载、检查、运行、汇总和发布入口
├── tests/             结构、数据语义和泄露门禁
├── results/           各任务结果和运行证据
├── requirements/      环境依赖
└── THIRD_PARTY_LICENSES/  上游代码许可证原文
```

| 目标 | 入口 |
|---|---|
| 查看三个任务的数据 | [datasets/README.md](datasets/README.md) |
| 查看深度模型和基础模型 | [项目文档](../docs/deep_learning/README.md) |
| 查看黎曼方法 | [项目文档](../docs/riemannian/README.md) |
| 查看跨任务实验矩阵 | [技术导读](../docs/TECHNICAL_GUIDE.md) |
| 查看深度模型跨任务结果 | [结果说明](../docs/deep_learning/RESULTS.md) |

所有运行命令均以本目录为工作目录：

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

返回[项目首页](../README.md)。
