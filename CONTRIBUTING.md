# 参与贡献

修改模型定义、输入约定、数据划分、标签或指标，属于 benchmark 变更，不是普通重构。请先提交
issue，说明科学依据、受影响的文件，以及哪些已有结果会因此失效。

每个 pull request 都应当：

1. 说明改动涉及数据语义、数据统一、模型输入适配、模型结构、训练协议，还是仅涉及报告；
2. 新增或更新回归测试；
3. 保证适配器的拟合代码不能访问验证集或测试集；
4. 同步更新机器可读配置和文档；
5. 说明哪些结果会过期、需要重跑；
6. 不得提交 EEG 信号文件、权重、凭据或私有下载地址。

提交前运行轻量测试：

```bash
python -m pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py tests/test_eegconformer_structure.py \
  tests/test_model_adapter_boundary.py model_adapters/tests/test_adapters_contract.py
```

正式接收还要求通过完整 CUDA/数据门禁。
