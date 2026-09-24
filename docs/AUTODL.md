# AutoDL deployment guide

## Recommended instance

- Ubuntu image with Python 3.12 support.
- NVIDIA GPU with at least 24 GB VRAM for the foundation-model preflights; a 4090-class card was
  used for the recorded fold-0 run.
- System disk for the environment and data disk for the 8.7 GB dataset, checkpoints, outputs,
  and caches. Keep at least 40 GB free to avoid incomplete checkpoints/results.

## Directory layout

```text
/root/autodl-tmp/MI32-Benchmark/        repository
/root/autodl-tmp/datasets/MI32_COMMON32_V4/
/root/autodl-tmp/MI32-Benchmark/third_party/
/root/autodl-tmp/MI32-Benchmark/checkpoints/
/root/autodl-tmp/MI32-Benchmark/outputs/
```

## Setup

```bash
cd /root/autodl-tmp
git clone https://github.com/OWNER/MI32-Benchmark.git
cd MI32-Benchmark
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

Upload or download the authorized dataset, then verify it:

```bash
python scripts/verify_dataset.py /root/autodl-tmp/datasets/MI32_COMMON32_V4 --full
bash scripts/run_full_gate.sh /root/autodl-tmp/datasets/MI32_COMMON32_V4
```

Only start full training when the gate exits zero. Keep `outputs/`, logs, and the environment
audit. Download and verify results before shutting down the instance.

## Billing safety

The scripts do not automatically power off the instance. After result hashes are verified and
copied off the machine, stop the instance from the AutoDL console. Automatic shutdown should be
implemented only with a documented provider API and an explicit user-owned credential; do not
embed SSH passwords or console tokens in this repository.
