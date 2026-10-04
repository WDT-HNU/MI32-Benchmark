# 在 AutoDL 上运行

以下是已记录 GPU 运行采用的配置。24 GB 显存适合作为基础模型预检的起点；fold-0 运行使用了
RTX 4090 级别实例。

请把仓库、数据集、权重和输出放在数据盘，并预留约 40 GB 空间，避免权重或结果包写到一半中断。

```text
/root/autodl-tmp/MI32-Benchmark/
/root/autodl-tmp/datasets/MI32_COMMON32_V4/
/root/autodl-tmp/MI32-Benchmark/third_party/
/root/autodl-tmp/MI32-Benchmark/checkpoints/
/root/autodl-tmp/MI32-Benchmark/outputs/
```

## 安装

```bash
cd /root/autodl-tmp
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

上传或下载有权使用的数据，然后检查完整数据和一个真实 CUDA 批次：

```bash
python scripts/verify_dataset.py /root/autodl-tmp/datasets/MI32_COMMON32_V4 --full
bash scripts/run_full_gate.sh /root/autodl-tmp/datasets/MI32_COMMON32_V4
```

只有两条命令都成功后才能开始完整训练。保留 `outputs/`、日志和环境审计文件，并在停止实例前
把结果包复制出去。

## 避免空闲计费

训练脚本不会自动关闭 AutoDL。确认复制出的结果哈希一致后，请在 AutoDL 控制台停止实例。
如果以后自动关机，请使用服务商正式 API 和自己管理的凭据；不要把 SSH 密码或控制台 token
写入本仓库。
