# Documentation

The documentation follows a task-first tree with three cross-cutting method families.

```text
Tasks
├── MI3
├── MI2
└── P300
    ×
Methods
├── traditional baselines
├── deep models, including foundation models
└── Riemannian geometry methods
```

Folders group related implementation material; they do not create separate datasets.

```text
docs/
├── TECHNICAL_GUIDE.md  complete task × method map
├── traditional/        conventional signal-processing and ML baselines
├── deep_learning/      model structures, adapters, protocols, audits, and results
├── riemannian/         manifold-method contracts, information regimes, and coverage
└── project/            data access, reproducibility, status, release, and licensing
```

| What you want to know | Read |
|---|---|
| Overall project logic | [Technical guide](TECHNICAL_GUIDE.md) |
| Data used by each task | [Dataset index](../benchmark/datasets/README.md) |
| Traditional baselines and the new P300 material | [Traditional methods](traditional/README.md) |
| Deep-model adaptations across all tasks | [Deep learning](deep_learning/README.md) |
| Riemannian methods across all tasks | [Riemannian methods](riemannian/README.md) |
| Reproduction procedure | [Reproducibility](project/REPRODUCIBILITY.md) |
| Current repository scope | [Project status](project/PROJECT_STATUS.md) |
