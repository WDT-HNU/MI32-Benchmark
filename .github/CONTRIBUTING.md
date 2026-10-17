# Contributing

A change to a model definition, input contract, data split, label, or metric changes the benchmark; it is not an ordinary refactor. Open an issue first and state the scientific basis, affected files, and historical results that would become invalid.

Every pull request should:

1. identify whether it changes data semantics, harmonization, model input, architecture, training protocol, or reporting only;
2. add or update regression tests;
3. prevent adapter fitting code from accessing validation or test data;
4. update machine-readable configuration and documentation together;
5. state which results become stale and require rerunning;
6. exclude EEG signal files, checkpoints, credentials, and private download URLs.

Run lightweight tests before submission:

```bash
cd benchmark
python -m pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py tests/test_eegconformer_structure.py \
  tests/test_model_adapter_boundary.py model_adapters/tests/test_adapters_contract.py
```

Formal acceptance also requires the full CUDA/data gate.
