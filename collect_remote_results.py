import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/outputs_mi32_common32_v4"))
args_cli = ap.parse_args()
ROOT = args_cli.root
DEST = ROOT / "final_csv"
DEST.mkdir(parents=True, exist_ok=True)

EXPECTED_DATASET_MANIFEST_SHA256 = "1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06"

SPECS = [
    (1, "EEGNet", "eegnet_final_fold0", "Traditional deep learning",
     "all 32 model-valid measured/interpolated MI-32 channels", "none", "supervised from scratch"),
    (2, "TSception", "tsception_final_fold0", "Traditional deep learning",
     "symmetric left-right pairs selected from the common MI-32 input", "none", "supervised from scratch"),
    (3, "RGNN backbone (without NodeDAT/EmotionDL)", "rgnn_final_fold0", "Traditional deep learning",
     "five-band log-power nodes plus learnable signed symmetric MI-32 graph", "none",
     "supervised inductive training from scratch"),
    (4, "EEG-Conformer", "eegconformer_final_fold0", "Traditional deep learning",
     "all 32 model-valid measured/interpolated MI-32 channels", "none", "supervised from scratch"),
    (5, "EEGMamba", "eegmamba_final_fold0", "EEG foundation model",
     "direct topology-ordered MI-32; anti-aliased 250-to-200 Hz; 3x200-point patches",
     "b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe",
     "full fine-tuning of official pretrained backbone with adapted default all_patch_reps head"),
    (6, "LaBraM", "labram_final_fold0", "EEG foundation model",
     "MI-32 canonical channels; 250-to-200 Hz resampling; temporal embedding interpolation",
     "86a40de11088b85291eb47b788b049d784e619a6b3f8e84accbf640b2c59eec3",
     "full fine-tuning of converted official pretrained backbone with new 3-class head"),
    (7, "CodeBrain", "codebrain_final_fold0", "EEG foundation model",
     "official 32-channel pass-through; anti-aliased 250-to-200 Hz; 3x200 flatten-all-patches head",
     "d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352",
     "full fine-tuning of official pretrained backbone with official-style flatten head"),
    (8, "Uni-NTFM", "uni_ntfm_final_fold0", "EEG foundation model",
     "MI-32 input mapped to the official five-region token grid", "none",
     "official-architecture supervised adapter trained from scratch (protocol_benchmark)"),
]


def _fail_closed(message):
    raise RuntimeError(message)


def _require_run_manifest(run_dir, directory):
    """Fail closed unless the run carries a hash-verified source snapshot."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        _fail_closed(f"{directory}: missing run_manifest.json (evidence required)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = manifest.get("source_snapshot", {})
    official_sources = snapshot.get("official_sources", {})
    if not official_sources or not all(
        entry.get("sha256") for entry in official_sources.values()
    ):
        _fail_closed(f"{directory}: run_manifest.json missing verified official source hashes")
    adapter = snapshot.get("adapter", {})
    if not adapter.get("sha256"):
        _fail_closed(f"{directory}: run_manifest.json missing adapter snapshot hash")
    if manifest.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        _fail_closed(f"{directory}: run_manifest dataset identity mismatch")
    if manifest.get("data_artifact_sha256", {}).get("SHA256SUMS") != EXPECTED_DATASET_MANIFEST_SHA256:
        _fail_closed(f"{directory}: run_manifest data artifact identity mismatch")
    return manifest


def _require_traditional_manifest(run_dir, directory, expected_result_model):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        _fail_closed(f"{directory}: missing run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_models = {expected_result_model}
    if expected_result_model == "RGNN":
        allowed_models.add("RGNN backbone (without target-domain NodeDAT/EmotionDL)")
    if manifest.get("model") not in allowed_models:
        _fail_closed(f"{directory}: manifest model mismatch {manifest.get('model')!r}")
    if manifest.get("data_version") != "4.0-common32":
        _fail_closed(f"{directory}: manifest data version mismatch")
    if manifest.get("data_artifact_sha256", {}).get("SHA256SUMS") != EXPECTED_DATASET_MANIFEST_SHA256:
        _fail_closed(f"{directory}: manifest dataset identity mismatch")
    checkpoint = manifest.get("checkpoint_sha256", "")
    if len(checkpoint) != 64:
        _fail_closed(f"{directory}: invalid trained-checkpoint identity")
    return manifest

rows = []
for order, expected_model, directory, group, strategy, checkpoint_hash, regime in SPECS:
    run_dir = ROOT / directory
    result_path = run_dir / "results.csv"
    config_path = run_dir / "run_config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete run: {directory}")
    result = pd.read_csv(result_path)
    expected_result_model = "RGNN" if expected_model.startswith("RGNN backbone") else expected_model
    if set(result["split"]) != {"val", "test"} or len(result) != 2:
        raise RuntimeError(f"{directory}: expected exactly one val and one test row")
    if "model" not in result or set(result["model"].astype(str)) != {expected_result_model}:
        _fail_closed(f"{directory}: result model identity mismatch")
    if "dataset_version" not in result or set(result["dataset_version"].astype(str)) != {"4.0-common32"}:
        _fail_closed(f"{directory}: result is not from sealed dataset version 4.0-common32")
    if "dataset_manifest_sha256" not in result or set(result["dataset_manifest_sha256"].astype(str)) != {EXPECTED_DATASET_MANIFEST_SHA256}:
        _fail_closed(f"{directory}: result is not from the sealed dataset manifest")
    if "model_adapter_kind" not in result or set(result["model_adapter_kind"].astype(str)) != {"model_input_adapter"}:
        _fail_closed(f"{directory}: result is not from the separated Model Input Adapter architecture")
    if set(result["input_channels"].astype(int)) != {32} or set(result["fold"].astype(int)) != {0}:
        _fail_closed(f"{directory}: input channel or fold mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config["args"]
    if int(args.get("fold", -1)) != 0:
        _fail_closed(f"{directory}: run_config fold mismatch")
    if expected_result_model != "Uni-NTFM" and args.get("model") != expected_result_model:
        _fail_closed(f"{directory}: run_config model mismatch {args.get('model')!r}")
    if config.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        _fail_closed(f"{directory}: run_config dataset identity mismatch")
    adapter_manifest = config.get("model_adapter_manifest", {})
    if adapter_manifest.get("adapter_kind") != "model_input_adapter":
        _fail_closed(f"{directory}: missing model_input_adapter manifest identity")
    if not args.get("final_test", False):
        raise RuntimeError(f"{directory}: not marked as a final-test run")
    if "test_was_not_used_for_selection" in config and not config["test_was_not_used_for_selection"]:
        raise RuntimeError(f"{directory}: leakage attestation failed")
    split_sets = [set(map(int, config.get(name, []))) for name in ("train_subjects", "val_subjects", "test_subjects")]
    if not all(split_sets) or any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        _fail_closed(f"{directory}: subject split evidence is empty or overlapping")
    core_metrics = [
        "accuracy", "macro_f1", "balanced_accuracy", "recall_0", "recall_1",
        "recall_2", "upper_vs_non_accuracy", "upper_recall", "non_upper_recall",
        "upper_lr_end_to_end_accuracy",
    ]
    for column in core_metrics:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            _fail_closed(f"{directory}: invalid metric values in {column}")
    losses = pd.to_numeric(result["loss"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        _fail_closed(f"{directory}: invalid loss values")
    for _, metric_row in result.iterrows():
        confusion = np.asarray(json.loads(metric_row["confusion_matrix"]), dtype=int)
        if confusion.shape != (3, 3) or (confusion < 0).any() or int(confusion.sum()) != int(metric_row["n_trials"]):
            _fail_closed(f"{directory}: confusion matrix evidence mismatch")
    result_class = "formal_reproduction"
    if expected_result_model in {"EEGNet", "TSception", "RGNN", "EEG-Conformer"}:
        _require_traditional_manifest(run_dir, directory, expected_result_model)
    if expected_model == "EEGMamba":
        _require_run_manifest(run_dir, directory)
        checkpoint = config.get("checkpoint_load_report", {})
        if checkpoint.get("channel_adapter") is not None:
            _fail_closed(f"{directory}: stale EEGMamba adapter detected (channel_adapter present)")
        if checkpoint.get("classifier_mode") != "all_patch_reps (official default)":
            _fail_closed(f"{directory}: stale EEGMamba classifier_mode {checkpoint.get('classifier_mode')!r}")
    if expected_model == "LaBraM":
        _require_run_manifest(run_dir, directory)
        equivalence = ROOT / "labram_checkpoint_equivalence.json"
        if not equivalence.is_file():
            _fail_closed(f"{directory}: missing labram_checkpoint_equivalence.json evidence")
    if expected_model == "CodeBrain":
        _require_run_manifest(run_dir, directory)
        checkpoint = config.get("checkpoint_load_report", {})
        if checkpoint.get("channel_adapter") is not None:
            _fail_closed(f"{directory}: stale CodeBrain adapter detected (channel_adapter present)")
        if checkpoint.get("classifier_mode") != "flatten all patches (official wrapper style)":
            _fail_closed(f"{directory}: stale CodeBrain classifier_mode {checkpoint.get('classifier_mode')!r}")
        if not checkpoint.get("checkpoint_identity_verified", False):
            _fail_closed(f"{directory}: CodeBrain checkpoint identity not verified")
    if expected_model == "Uni-NTFM":
        benchmark_track = config.get("benchmark_track")
        contract = config.get("benchmark_contract", {})
        if benchmark_track != "protocol_benchmark":
            _fail_closed(f"{directory}: Uni-NTFM benchmark_track {benchmark_track!r} (expected protocol_benchmark)")
        if contract.get("per_trial_zscore") is not False:
            _fail_closed(f"{directory}: Uni-NTFM per-trial z-score detected (forbidden)")
        if contract.get("channel_mapping") != "official five-region token grid":
            _fail_closed(f"{directory}: Uni-NTFM channel mapping {contract.get('channel_mapping')!r}")
        if contract.get("official_repo_commit") != "c0ce0152f94366e59b31b3fb2c108ce909bcd95c":
            _fail_closed(f"{directory}: Uni-NTFM official commit mismatch {contract.get('official_repo_commit')!r}")
        result_class = "protocol_benchmark"
    for _, source in result.iterrows():
        rows.append({
            "model_order": order,
            "model": expected_model,
            "model_group": group,
            "implementation": source.get("implementation", ""),
            "result_class": result_class,
            "training_regime": regime,
            "pretraining": source.get("pretraining", "none supervised from scratch"),
            "checkpoint_sha256": checkpoint_hash,
            "data_version": source.get("dataset_version", "4.0-common32"),
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
            "input_channels": int(source.get("input_channels", 32)),
            "input_strategy": strategy,
            "fold": int(source["fold"]),
            "split": "validation" if source["split"] == "val" else "test",
            "seed": int(source["seed"]),
            "learning_rate": float(args["lr"]),
            "batch_size": int(args["batch_size"]),
            "planned_epochs": int(args["epochs"]),
            "best_epoch": int(source["best_epoch"]),
            "loss": float(source["loss"]),
            "accuracy": float(source["accuracy"]),
            "macro_f1": float(source["macro_f1"]),
            "balanced_accuracy": float(source["balanced_accuracy"]),
            "recall_left_upper": float(source["recall_0"]),
            "recall_right_upper": float(source["recall_1"]),
            "recall_non_upper": float(source["recall_2"]),
            "upper_vs_non_accuracy": float(source["upper_vs_non_accuracy"]),
            "upper_recall": float(source["upper_recall"]),
            "non_upper_recall": float(source["non_upper_recall"]),
            "upper_lr_end_to_end_accuracy": float(source["upper_lr_end_to_end_accuracy"]),
            "upper_lr_conditional_accuracy": float(source["upper_lr_conditional_accuracy"]),
            "upper_lr_conditional_n": int(source["upper_lr_conditional_n"]),
            "n_trials": int(source["n_trials"]),
            "n_subjects": int(source["n_subjects"]),
            "confusion_matrix": source["confusion_matrix"],
            "selection_metric": "validation macro_f1",
            "test_used_for_model_selection": False,
            "status": "completed single subject-held-out fold",
        })

final = pd.DataFrame(rows).sort_values(["model_order", "split"], ascending=[True, False])
if len(final) != 16 or final["model"].nunique() != 8:
    raise RuntimeError("final table completeness check failed")
final.to_csv(DEST / "model_results.csv", index=False)
final.loc[final["split"] == "test"].sort_values("model_order").to_csv(
    DEST / "test_summary.csv", index=False,
)

audit = {
    "models": 8,
    "rows": 16,
    "validation_rows": int((final["split"] == "validation").sum()),
    "test_rows": int((final["split"] == "test").sum()),
    "unique_test_row_per_model": bool(
        (final.loc[final["split"] == "test"].groupby("model").size() == 1).all()
    ),
    "all_test_selection_flags_false": bool((~final["test_used_for_model_selection"]).all()),
}
(DEST / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit))
