# Training entry points

This directory contains deep-model training programs. The public runners currently concentrate on MI3. Use [`scripts/benchmark.py`](../scripts/benchmark.py) instead of calling files directly so dataset identity, checkpoint, and run-manifest checks are not skipped.

| File | Responsibility |
|---|---|
| `mi3_eegnet.py` | EEGNet, TSception, RGNN, EEG-Conformer |
| `mi3_foundation.py` | LaBraM, EEGMamba, CodeBrain |
| `uni_mi3_supervised.py` | separate Uni-NTFM protocol benchmark |
| `mi3_deep_models.py` | compatibility exports for task-specific deep models |
| `remote_mi3_deep_models.py` | frozen compatibility implementation for remote runs |

Example:

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full --output outputs/eegnet
```

Input adapters are under [`model_adapters/`](../model_adapters/); structural and data-semantic checks are under [`tests/`](../tests/).
