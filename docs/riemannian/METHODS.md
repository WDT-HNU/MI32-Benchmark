# Riemannian method summary

This page organizes the delivered method summary. Every comparison must retain task, frozen split, information regime, valid-fold count, and fidelity status.

## Common method logic

An EEG trial `X ∈ R^(p×T)` can be converted into a spatial covariance matrix. Strictly positive-definite covariance matrices lie on the SPD manifold; element-wise means, Euclidean distances, and linear interpolation ignore that geometry. AIRM distance, the Karcher mean, and tangent-space mapping provide a manifold distance, reference point, and local linear representation respectively.

The shared boundary is:

```text
frozen data and split
└── dataset- and method-specific adapter
    └── official method input contract
        └── frozen method core
            └── evaluation
                └── task, information regime, coverage, and status
```

The adapter follows the method. Plain centred SCM, official OAS covariance, and P300 ERP prototype/augmented covariance are not interchangeable. A method defined with unlabelled target statistics must not be silently rewritten as source-only; a method requiring labelled target calibration must use a pre-frozen calibration/evaluation partition.

The shared Binary MI input is `(B,32,750)`, 250 Hz, 3 s, volts, 8–30 Hz, with `0=Left` and `1=Right`. See the [Binary MI contract](../../benchmark/datasets/binary_mi/DATASET_CARD.md).

## Method tree

```text
Riemannian methods
├── direct classification
│   ├── MDRM: class Riemannian means + nearest AIRM mean
│   ├── TSLDA: tangent space + variable selection + LDA
│   ├── KernelSVM: tangent-space Riemannian kernel + linear SVM
│   ├── FGMDRM: Fisher geodesic filter + MDM
│   ├── SPDNet: SPD-specific neural layers
│   └── MFACC: OAS + ADCSP + means-field classifier
├── cross-domain alignment and transfer
│   ├── RA: unlabelled-target Riemannian alignment
│   ├── RPA: labelled-target-calibration Procrustes analysis
│   ├── TSA: labelled-target tangent-space alignment
│   └── MEKT-R: unlabelled-target distribution alignment and pseudo-label iteration
└── self-supervised representation
    └── Manifold-SimCLR: SPD/log-domain contrastive learning
```

## Method cards

### MDRM

- Basis: Barachant et al., 2012.
- Representation: centred SCM; scale-relative minimal spectral repair only after an SPD numerical gate fails.
- Flow: covariance → class Karcher means → AIRM distance → nearest mean.
- Information: `SOURCE_ONLY`; class means are estimated from training data only.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### TSLDA

- Basis: Barachant et al., 2012.
- Flow: training Riemannian mean → log map → `sqrt(2)` symmetric upper-triangle vectorization → SVD/PCA → ANOVA → weighted FDR (`q=0.05`) → LDA.
- Information: `SOURCE_ONLY`; reference, reduction, selection, and LDA are train-only.
- Boundary: selecting zero variables under weighted FDR is a real method failure. No fallback or fabricated prediction is added.
- Status: Binary 32/36 valid; MI3 16/16; P300 35/85 valid.

### KernelSVM

- Basis: Barachant et al., 2013.
- Flow: centred SCM → training AIRM mean → tangent map → `sqrt(2)` vectorization → linear SVC.
- Fixed convention: `SVC(kernel="linear", C=10)` with no performance-driven grid search.
- Information: `SOURCE_ONLY`; native multiclass compatibility is checked for MI3 and AUC uses continuous decision scores.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### FGMDRM / FgMDM

- Basis: Barachant et al., 2010, and the 2019 SSDT-FGMDRM work.
- Flow: AIRM mean → tangent map → Fisher/FGDA filtering → inverse tangent map → MDM.
- Information: `SOURCE_ONLY`; the Fisher projection is fitted on training vectors and labels only.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### RA

- Basis: Zanini et al., 2018.
- Flow: estimate source and target Riemannian references separately, recenter both domains, then predict with a source classifier.
- Information: `TRANSDUCTIVE_UNLABELED_TARGET`; target covariance may estimate the reference, while target labels are used only after prediction for metrics.
- Boundary: target self-inclusion defined by the paper must be disclosed, but it is not label leakage.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### SPDNet

- Basis: Huang and Van Gool, 2017.
- Structure: `BiMap → ReEig → BiMap → ReEig → BiMap → LogEig → vectorization → softmax`, verified as `32→16→8→4`.
- Optimization: Stiefel/Riemannian gradients and QR retraction for BiMap weights; author-style spectral backward for ReEig/LogEig.
- Information: `SOURCE_ONLY`.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal; BiMap parameter updates were audited.

### MFACC

- Representation: raw frozen epoch through the official OAS covariance route.
- Flow: `Covariances(estimator="oas") → ADCSP → MFACC(method_label="lda")`.
- Information: `SOURCE_ONLY`; a shared SCM cache or manual SPD repair does not replace official OAS.
- Status: Binary 36/36 and P300 85/85 formal; MI3 13/16 valid with three numerical failures retained.

### RPA

- Basis: Rodrigues et al., 2019.
- Flow: recentering → stretching → class-wise rotation → source-trained classifier.
- Information: `TARGET_CALIBRATION_LABELED`; only the pre-frozen target calibration labels are allowed. Evaluation labels are used only for final metrics.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### TSA

- Basis: Bleuzé, Mattout, and Congedo, 2022.
- Flow: source/target tangent representations → centering and rescaling → class anchors → SVD/Procrustes rotation → source SVC.
- Information: `TARGET_CALIBRATION_LABELED`; evaluation labels do not enter alignment.
- Boundary: P300 uses an ERP-aware representation rather than MI plain SCM.
- Status: Binary 36/36, MI3 16/16, and P300 85/85 formal.

### Manifold-SimCLR

- Direction: SPD/log-domain augmentation → encoder/projection → contrastive loss → downstream probe or fine-tuning.
- Information: `SELF_SUPERVISED_SOURCE_ONLY`, conditional on a complete executable original-method definition.
- Boundary: the delivered Binary 36-fold result is reconstructed exploratory. MXOptimizer, the exact backbone, and the joint fine-tuning schedule cannot be uniquely determined from the available author material, so it is not labelled paper-faithful.
- Status: Binary `PARTIAL_RECONSTRUCTION`; MI3 and P300 `IMPLEMENTATION_BLOCKED`.

### MEKT-R

- Basis: Zhang and Wu, 2020.
- Flow: subject alignment → official MTS source concatenation → joint-probability-MMD/generalized eigenproblem → five pseudo-label iterations → official shrinkage LDA.
- Fixed values: `p=10`, `T=5`, `alpha=0.01`, `beta=0.1`, `rho=20`, target KNN `k=10`, and heat-kernel `sigma=1`.
- Information: `TRANSDUCTIVE_UNLABELED_TARGET`; true target labels do not enter adaptation, pseudo-labels, or the stopping rule.
- Boundary: P300 uses the ERP/RSVP-aware route. The official MI3 endpoint is binary-only; no unreported OvR/OvO extension is added.
- Status: Binary 36/36 and P300 85/85 formal; MI3 `BINARY_ONLY`.

## Information regimes

| Regime | Methods | Permitted fitted information | Leakage boundary |
|---|---|---|---|
| `SOURCE_ONLY` | MDRM, TSLDA, KernelSVM, FGMDRM, SPDNet, MFACC | source/train EEG, covariance, and labels | reference, selector, and classifier are train-only |
| `TRANSDUCTIVE_UNLABELED_TARGET` | RA, MEKT-R | source labels plus target EEG/covariance/statistics | target labels appear only after prediction for metrics |
| `TARGET_CALIBRATION_LABELED` | RPA, TSA | source data plus labels from a frozen calibration subset | evaluation labels do not enter alignment or training |
| `SELF_SUPERVISED` | Manifold-SimCLR | source-side data allowed by the original protocol | evaluation labels do not enter pretraining or selection |

The paper's method contract determines whether target information is allowed. Unlabelled target statistics are not label leakage, and labelled calibration results cannot be placed beside source-only results without protocol disclosure.

## Tasks

### Binary left/right MI

The 18-source release contains 389 subjects and 59,216 trials. The full protocol has 84 folds; the representative protocol selects two existing folds per dataset, for 36 folds. The primary representation is spatial covariance over the MI band.

### MI3

The archived representative protocol contains `8 datasets × 2 folds = 16 folds`. Multiclass endpoints must be checked method by method; a binary decision rule cannot be silently extended. KernelSVM and MFACC are recorded as multiclass-compatible, while MEKT-R remains `BINARY_ONLY`.

### P300

P300 is a time-locked ERP task and should not be compressed mechanically into MI-style plain spatial SCM. The frozen reduced LOGO protocol contains 85 folds. TSA and MEKT-R use ERP-aware routes; TSLDA preserves genuine zero-feature failures; SPDNet preserves the audited spectral backward and Stiefel/BiMap updates.

## Status matrix

| Method | Binary MI (36) | MI3 (16) | P300 reduced LOGO (85) |
|---|---|---|---|
| MDRM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| TSLDA | PARTIAL 32/36 | FORMAL 16/16 | PARTIAL 35/85 valid |
| KernelSVM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| FGMDRM | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| RA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| SPDNet | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| MFACC | FORMAL 36/36 | PARTIAL 13/16 | FORMAL 85/85 |
| RPA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| TSA | FORMAL 36/36 | FORMAL 16/16 | FORMAL 85/85 |
| Manifold-SimCLR | PARTIAL_RECONSTRUCTION 36/36 | IMPLEMENTATION_BLOCKED | IMPLEMENTATION_BLOCKED |
| MEKT-R | FORMAL 36/36 | BINARY_ONLY | FORMAL 85/85 |

`FORMAL` means that the delivered summary records completion under the frozen protocol. `PARTIAL` retains real method or numerical failures. `BINARY_ONLY` and `IMPLEMENTATION_BLOCKED` are fidelity or scope states, not performance scores.

## Paths recorded by the delivered summary

These names preserve the original workspace relationships:

| Purpose | Original relative path |
|---|---|
| Binary MI full splits | `evaluation/within_dataset_splits.json` |
| Binary MI representative folds | `evaluation/representative_2fold_subset.json` |
| Master adapter report | `experiments/MASTER_ADAPTER_BENCHMARK_REPORT.md` |
| MEKT-R specification and report | `experiments/MEKT_adapter/MEKT_METHOD_SPEC.md`, `MEKT_FINAL_REPORT.md` |
| P300 benchmark | `P300_benchmark/` |
| MI3 benchmark | `05_benchmark/` |
| P300 fidelity repair | `P300_benchmark/experiments/*_fidelity_repair/` |
| MI3 multiclass compatibility audit | `05_benchmark/experiments/MEKT_R_05_MULTICLASS_COMPATIBILITY_AUDIT.md` |

These paths record method, input, and protocol provenance. A cross-task summary must still retain task, split, information regime, coverage, and fidelity status.
