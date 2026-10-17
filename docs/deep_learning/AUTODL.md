# Running on AutoDL

The recorded GPU runs used an RTX 4090-class instance. A 24 GB GPU is a practical starting point for foundation-model preflight checks.

Keep the repository, dataset, checkpoints, and outputs on the data volume and reserve roughly 40 GB so a checkpoint or result package is not interrupted mid-write.

```text
/root/autodl-tmp/MI32-Benchmark/
/root/autodl-tmp/datasets/MI32_COMMON32_V4/
/root/autodl-tmp/MI32-Benchmark/benchmark/third_party/
/root/autodl-tmp/MI32-Benchmark/benchmark/checkpoints/
/root/autodl-tmp/MI32-Benchmark/benchmark/outputs/
```

## Installation

```bash
cd /root/autodl-tmp
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark/benchmark
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

Upload or download data you are entitled to use, then verify the full release and one real CUDA batch:

```bash
python scripts/verify_dataset.py /root/autodl-tmp/datasets/MI32_COMMON32_V4 --full
bash scripts/run_full_gate.sh /root/autodl-tmp/datasets/MI32_COMMON32_V4
```

Start a full training run only after both commands pass. Preserve `outputs/`, logs, and the environment audit, then copy the result package off the instance before shutdown.

## Avoiding idle charges

Training scripts do not shut down AutoDL automatically. After confirming that the copied result hash matches, stop the instance in the AutoDL console. If automated shutdown is added later, use the provider's official API with privately managed credentials. Never commit an SSH password or console token.
