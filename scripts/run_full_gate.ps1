param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ThirdParty = if ($env:MI32_THIRD_PARTY_ROOT) { $env:MI32_THIRD_PARTY_ROOT } else { Join-Path $Root "third_party" }
$Checkpoints = if ($env:MI32_CHECKPOINT_ROOT) { $env:MI32_CHECKPOINT_ROOT } else { Join-Path $Root "checkpoints" }
$Outputs = if ($env:MI32_OUTPUT_ROOT) { $env:MI32_OUTPUT_ROOT } else { Join-Path $Root "outputs\gate" }

Push-Location $Root
try {
    & $Python scripts/verify_repository.py
    & $Python scripts/verify_dataset.py $DataRoot --full
    & $Python scripts/benchmark.py doctor --data $DataRoot --third-party $ThirdParty --checkpoints $Checkpoints

    $env:MI32_DATA_ROOT = $DataRoot
    $env:RGNN_TEST_DATA = $DataRoot
    & $Python -m pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py `
        tests/test_rgnn_structure.py tests/test_eegconformer_structure.py `
        tests/test_mi32_data_contract.py tests/test_mi3rawtrials_semantics.py `
        tests/test_model_adapter_boundary.py model_adapters/tests/test_adapters_contract.py

    $env:LABRAM_TEST_CHECKPOINT = Join-Path $Checkpoints "LaBraM\pytorch_model.bin"
    $env:LABRAM_AUTHOR_CHECKPOINT = Join-Path $Checkpoints "LaBraM\labram-base.pth"
    & $Python -m pytest -q tests/test_labram_structure.py

    $env:EEGMAMBA_TEST_REPO = Join-Path $ThirdParty "EEGMamba"
    $env:EEGMAMBA_TEST_CHECKPOINT = Join-Path $Checkpoints "EEGMamba\pretrained_EEGMamba.pth"
    & $Python -m pytest -q tests/test_eegmamba_structure.py

    $env:CODEBRAIN_TEST_REPO = Join-Path $ThirdParty "CodeBrain"
    $env:CODEBRAIN_TEST_CHECKPOINT = Join-Path $Checkpoints "CodeBrain\CodeBrain.pth"
    & $Python -m pytest -q tests/test_codebrain_structure.py

    $env:UNINTFM_TEST_REPO = Join-Path $ThirdParty "Uni-NTFM"
    $env:UNINTFM_TEST_ADAPTER = Join-Path $Root "uni_mi3_supervised.py"
    & $Python tests/test_uni_ntfm_structure.py

    New-Item -ItemType Directory -Path $Outputs -Force | Out-Null
    foreach ($Model in @("eegnet", "tsception", "rgnn", "eegconformer", "labram", "eegmamba", "codebrain")) {
        & $Python scripts/benchmark.py run --model $Model --data $DataRoot `
            --output (Join-Path $Outputs $Model) --third-party $ThirdParty `
            --checkpoints $Checkpoints --limit 4 --preflight
    }
    & $Python scripts/benchmark.py run --model uni_ntfm --data $DataRoot `
        --output (Join-Path $Outputs "uni_ntfm") --third-party $ThirdParty `
        --checkpoints $Checkpoints --limit 4 --preflight --allow-protocol-benchmark
    New-Item -ItemType File -Path (Join-Path $Outputs "ALL_PREFLIGHTS_PASS") -Force | Out-Null
    Write-Output "ALL_PREFLIGHTS_PASS $Outputs"
}
finally {
    Pop-Location
}
