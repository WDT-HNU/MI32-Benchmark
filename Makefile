PYTHON ?= python
DATA ?= datasets/mi32/full

.PHONY: verify metadata-test cpu-test fetch doctor

verify:
	$(PYTHON) scripts/verify_repository.py

metadata-test:
	$(PYTHON) scripts/verify_dataset.py datasets/mi32/metadata

cpu-test:
	$(PYTHON) -m pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py tests/test_rgnn_structure.py tests/test_eegconformer_structure.py tests/test_mi3rawtrials_semantics.py tests/test_model_adapter_boundary.py model_adapters/tests/test_adapters_contract.py

fetch:
	$(PYTHON) scripts/fetch_upstreams.py
	$(PYTHON) scripts/fetch_checkpoints.py

doctor:
	$(PYTHON) scripts/benchmark.py doctor --data $(DATA)
