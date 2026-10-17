# Benchmark workspace

This directory contains runnable and verifiable material. The scientific structure is task-first; files are grouped by engineering purpose.

```text
benchmark/
├── datasets/              MI3, MI2, and P300 data contracts and metadata
├── dataset_tools/         dataset construction and validation
├── model_adapters/        deep-model input adapters
├── runners/               training and evaluation programs
├── configs/               pinned data, model, and evaluation settings
├── scripts/               fetch, verify, run, collect, and package entry points
├── tests/                 structure, data-semantic, and leakage gates
├── results/               task results and run evidence
├── requirements/          environment specifications
└── THIRD_PARTY_LICENSES/  retained upstream licence texts
```

| Goal | Entry point |
|---|---|
| Inspect task data contracts | [datasets/README.md](datasets/README.md) |
| Inspect traditional baselines | [documentation](../docs/traditional/README.md) |
| Inspect deep and foundation models | [documentation](../docs/deep_learning/README.md) |
| Inspect Riemannian methods | [documentation](../docs/riemannian/README.md) |
| Read the full experiment matrix | [technical guide](../docs/TECHNICAL_GUIDE.md) |
| Read cross-task deep results | [result notes](../docs/deep_learning/RESULTS.md) |

Run commands from this directory:

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

Return to the [project home](../README.md).
