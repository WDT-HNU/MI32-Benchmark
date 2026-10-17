# Deep models

This section documents two subgroups within one method family:

- task-specific architectures: EEGNet, TSception, RGNN, and EEG-Conformer;
- EEG foundation and long-sequence models: LaBraM, EEGMamba, CodeBrain, and Uni-NTFM.

These models are not restricted to MI3. Recorded work covers MI3, MI2, and P300.

## Cross-task status

| Task | Models with recorded runs | Current evidence scope |
|---|---|---|
| MI3 | EEGNet, TSception, RGNN, EEG-Conformer, LaBraM, EEGMamba, CodeBrain | fold coverage differs by model |
| MI2 | the same 7 models | fold 0; 45 held-out subjects and 9,462 trials |
| P300 | the same 7 models | 102 fixed sampled LOGO folds; 714 runs |
| Uni-NTFM | structure and protocol route | no run lacking an acceptable downstream-weight and head contract is presented as a formal result |

## Shared task inputs

| Task | Common input | Labels | Evaluation boundary |
|---|---|---|---|
| MI3 | `[B,32,750] @ 250 Hz`, volts | left_upper / right_upper / non_upper | subject-independent folds; validation-only model selection |
| MI2 | left/right view of the same MI release | Left / Right | current formal evidence is fold 0 |
| P300 | `[B,16,121] @ 200 Hz`, volts | NonTarget / Target | 102 frozen sampled LOGO folds; fixed 8 epochs |

Adapters may convert the frozen tensor into a model-required unit, rate, patch layout, graph feature, or standardized representation. They may not change labels, trial ownership, or split membership. Every fitted statistic uses only the training portion allowed by the protocol.

## Documentation and code

| Topic | Location |
|---|---|
| Architecture and classification path | [MODELS.md](MODELS.md) |
| Per-model identity and boundary | [model cards](model_cards/README.md) |
| MI3 split, tuning, and metrics | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) |
| Cross-task results | [RESULTS.md](RESULTS.md) |
| Structural evidence | [audits/EVIDENCE_MATRIX.md](audits/EVIDENCE_MATRIX.md) |
| AutoDL execution | [AUTODL.md](AUTODL.md) |

The public runnable workspace concentrates on MI3 adapters, runners, structural tests, and run-level evidence. MI2 and P300 summaries come from completed independent run packages and retain their original protocol scope.

Return to the [project overview](../../README.md), or compare with [traditional methods](../traditional/README.md) and [Riemannian methods](../riemannian/README.md).
