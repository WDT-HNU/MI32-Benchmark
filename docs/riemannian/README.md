# Riemannian methods

This section records input representations, information regimes, task applicability, and coverage for eleven Riemannian methods. It is not an MI2-only branch: the archived status covers MI2, MI3, and P300.

```text
frozen task data and split
└── method-specific adapter
    └── official input contract
        └── frozen method core
            └── evaluation and coverage report
```

## Cross-task status

| Task | Protocol | Current coverage |
|---|---|---|
| MI2 | Binary MI representative 36 folds | most methods formal; TSLDA partial; Manifold-SimCLR reconstructed exploratory |
| MI3 | representative 16 folds | most methods 16/16; MFACC 13/16; MEKT-R remains binary-only |
| P300 | reduced frozen LOGO 85 folds | most methods 85/85; TSLDA retains zero-feature failures |

The archive covers MDRM, TSLDA, KernelSVM, FGMDRM, RA, SPDNet, MFACC, RPA, TSA, Manifold-SimCLR, and MEKT-R. See the [method summary](METHODS.md) for definitions.

The 18-source Binary MI contract is documented in its [dataset card](../../benchmark/datasets/binary_mi/DATASET_CARD.md). It defines the same binary task as the MI2 view used by deep models but is a different frozen release; the two result sets cannot be ranked without retaining data identity.

Return to the [project overview](../../README.md), or compare with [traditional methods](../traditional/README.md) and [deep models](../deep_learning/README.md).
