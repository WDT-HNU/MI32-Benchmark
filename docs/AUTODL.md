# Running on AutoDL

This is the setup used for the recorded GPU runs. A 24 GB card is a comfortable starting point for
the foundation-model preflights; the fold-0 runs used an RTX 4090-class instance.

Keep the repository, dataset, checkpoints, and outputs on the data disk. Leave roughly 40 GB free
so a checkpoint or result archive is not cut off halfway through.

```text
/root/autodl-tmp/MI32-Benchmark/
/root/autodl-tmp/datasets/MI32_COMMON32_V4/
/root/autodl-tmp/MI32-Benchmark/third_party/
/root/autodl-tmp/MI32-Benchmark/checkpoints/
/root/autodl-tmp/MI32-Benchmark/outputs/
```

## Install

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

Upload or download the authorized dataset, then check both the data and one real CUDA batch:

```bash
python scripts/verify_dataset.py /root/autodl-tmp/datasets/MI32_COMMON32_V4 --full
bash scripts/run_full_gate.sh /root/autodl-tmp/datasets/MI32_COMMON32_V4
```

Start full training only after both commands return successfully. Keep `outputs/`, logs, and the
environment audit, and copy the result package off the machine before stopping the instance.

## Avoid paying for an idle machine

The training scripts do not shut down AutoDL. After the copied result hashes match, stop the
instance in the AutoDL console. If you later automate shutdown, use the provider's documented API
and a credential you control; do not put SSH passwords or console tokens in this repository.
