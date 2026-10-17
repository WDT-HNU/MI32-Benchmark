# P300 harmonized dataset contract

This card records the P300/ERP data contract described in the supplied dataset-and-method note and used by the reported traditional benchmark. It is a task-level contract: every model family must still disclose its own frozen split and adapter.

## Task

- Binary epoch classification: `NonTarget = 0`, `Target = 1`.
- Dataset candidates are public P300/ERP datasets exposed through MOABB.
- Dataset-specific event names such as P300 or rare stimulus are mapped to `Target`; all remaining valid task events are mapped to `NonTarget`.

The source note reports 15 eligible datasets. The 2026-09-04 traditional-method snapshot reports 25 evaluable subjects. The deep-model archive separately reports 158,667 trials. These counts describe different recorded artifacts and are not silently treated as one frozen release.

## Common signal representation

| Field | Contract |
|---|---|
| Montage | 16 P300-focused channels |
| Channel order | `Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, FCz, C3, C4, CP3, CPz, CP4` |
| Missing channels | Spatial interpolation from standard scalp coordinates; never zero filling |
| Reference | Common average reference after channel harmonization |
| Band-pass | 0.5–20 Hz |
| Sampling rate | 200 Hz |
| Epoch | 0–600 ms after stimulus onset |
| Tensor convention | trial × channel × time |

At 200 Hz, an endpoint-inclusive 0–600 ms grid contains 121 samples, which matches the deep-model P300 input contract `[B, 16, 121]`. A method must not change labels, trial ownership, or split membership while adapting this representation.

## Evaluation groups

The traditional benchmark uses leave-one-group-out (LOGO) evaluation. Each held-out group is a complete session or run; the remaining groups form the training data for that fold. The report records 873 folds. With 16 methods, this yields 13,968 method-fold evaluations.

Other branches use different frozen subsets:

| Method family | Recorded P300 scope |
|---|---|
| Traditional methods | full 873-fold LOGO report |
| Deep models | fixed 102-fold sample from the 873-fold universe |
| Riemannian methods | reduced frozen 85-fold LOGO protocol |

Results may be discussed under the same P300 task, but they are not paired unless their split identities match.

## Leakage controls

- Spatial filters, feature selectors, templates, dimensionality reduction, and classifiers are fitted inside the training portion of each fold.
- A held-out session or run is used only for evaluation.
- Any learned normalization or calibration quantity must follow the information regime declared by the method.
- Target/NonTarget imbalance must be reflected in the reported metrics; ordinary accuracy is not sufficient on its own.

## Provenance and references

The dataset selection and harmonization route is based on MOABB's separation of datasets, paradigms, evaluations, and pipelines. The P300 paradigm defines Target/NonTarget epoch classification, while the 16-channel montage, 0.5–20 Hz band, 200 Hz rate, and 0–600 ms window are project-specific frozen choices.

- [MOABB P300 paradigm documentation](https://moabb.neurotechx.com/docs/generated/moabb.paradigms.p300.BaseP300.html)
- [MOABB API and dataset concepts](https://moabb.neurotechx.com/docs/api.html)

This public card is an English, reviewed representation of the supplied internal note `P300 merge and methods`. It does not publish the original EEG files or grant redistribution rights for any component dataset.
