# 环境配置

正式 fold-0 运行记录的环境为 RTX 4090 D、Python 3.12.3、PyTorch 2.8.0+cu128、
CUDA 12.8、MNE 1.12.1、NumPy 2.3.2、SciPy 1.18.0 和 pandas 3.0.5。

`core.txt` 提供数据工具和 4 种监督模型所需依赖；`foundation.txt` 加入封存配置所需的
LaBraM 和 Mamba 依赖。CUDA wheel 与平台相关，因此请先从 CUDA 12.8 索引安装 PyTorch，
再安装这些依赖文件。

该环境配置记录的是成功运行条件，不表示仅用于 CPU 结构检查的 stub 与 CUDA 内核数值等价。
