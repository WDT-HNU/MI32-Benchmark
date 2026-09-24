#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${1:?usage: run_full_gate.sh /path/to/MI32_COMMON32_V4}"
PYTHON="${PYTHON:-python}"
THIRD_PARTY="${MI32_THIRD_PARTY_ROOT:-$ROOT/third_party}"
CHECKPOINTS="${MI32_CHECKPOINT_ROOT:-$ROOT/checkpoints}"
OUTPUTS="${MI32_OUTPUT_ROOT:-$ROOT/outputs/gate}"

cd "$ROOT"
"$PYTHON" scripts/verify_repository.py
"$PYTHON" scripts/verify_dataset.py "$DATA" --full
"$PYTHON" scripts/benchmark.py doctor --data "$DATA" --third-party "$THIRD_PARTY" --checkpoints "$CHECKPOINTS"

MI32_DATA_ROOT="$DATA" RGNN_TEST_DATA="$DATA" \
  "$PYTHON" -m pytest -q \
    tests/test_eegnet_structure.py \
    tests/test_tsception_structure.py \
    tests/test_rgnn_structure.py \
    tests/test_eegconformer_structure.py \
    tests/test_mi32_data_contract.py \
    tests/test_mi3rawtrials_semantics.py \
    tests/test_model_adapter_boundary.py \
    model_adapters/tests/test_adapters_contract.py

LABRAM_TEST_CHECKPOINT="$CHECKPOINTS/LaBraM/pytorch_model.bin" \
LABRAM_AUTHOR_CHECKPOINT="$CHECKPOINTS/LaBraM/labram-base.pth" \
  "$PYTHON" -m pytest -q tests/test_labram_structure.py

EEGMAMBA_TEST_REPO="$THIRD_PARTY/EEGMamba" \
EEGMAMBA_TEST_CHECKPOINT="$CHECKPOINTS/EEGMamba/pretrained_EEGMamba.pth" \
  "$PYTHON" -m pytest -q tests/test_eegmamba_structure.py

CODEBRAIN_TEST_REPO="$THIRD_PARTY/CodeBrain" \
CODEBRAIN_TEST_CHECKPOINT="$CHECKPOINTS/CodeBrain/CodeBrain.pth" \
  "$PYTHON" -m pytest -q tests/test_codebrain_structure.py

UNINTFM_TEST_REPO="$THIRD_PARTY/Uni-NTFM" \
UNINTFM_TEST_ADAPTER="$ROOT/uni_mi3_supervised.py" \
  "$PYTHON" tests/test_uni_ntfm_structure.py

mkdir -p "$OUTPUTS"
for model in eegnet tsception rgnn eegconformer labram eegmamba codebrain; do
  "$PYTHON" scripts/benchmark.py run --model "$model" --data "$DATA" \
    --output "$OUTPUTS/$model" --third-party "$THIRD_PARTY" --checkpoints "$CHECKPOINTS" \
    --limit 4 --preflight
done
"$PYTHON" scripts/benchmark.py run --model uni_ntfm --data "$DATA" \
  --output "$OUTPUTS/uni_ntfm" --third-party "$THIRD_PARTY" --checkpoints "$CHECKPOINTS" \
  --limit 4 --preflight --allow-protocol-benchmark

touch "$OUTPUTS/ALL_PREFLIGHTS_PASS"
echo "ALL_PREFLIGHTS_PASS $OUTPUTS"
