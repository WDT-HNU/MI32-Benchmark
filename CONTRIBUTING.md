# Contributing

Changes to model definitions, input contracts, splits, labels, or metrics are benchmark changes,
not routine refactors. Open an issue first and describe the scientific reason, affected artifacts,
and invalidated results.

Every pull request should:

1. identify whether it changes dataset semantics, harmonization, model input adaptation, model
   structure, training protocol, or reporting only;
2. add or update a regression test;
3. avoid accessing validation/test data from adapter fitting code;
4. update the machine-readable registry and documentation;
5. state which results become stale and must be rerun;
6. contain no EEG signal files, checkpoints, credentials, or private download URLs.

Run the lightweight suite before submitting:

```bash
pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py tests/test_eegconformer_structure.py \
  tests/test_model_adapter_boundary.py model_adapters/tests/test_adapters_contract.py
```

Formal acceptance additionally requires the full CUDA/data gate.
