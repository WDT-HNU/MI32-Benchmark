import argparse
import json
from pathlib import Path

import numpy as np
from model_adapters.rgnn import RGNNAdapter
import pandas as pd
import torch

from mi3_eegnet import RGNN, RGNN_BANDS


class RunningStats:
    def __init__(self, width):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_square = np.zeros(width, dtype=np.float64)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_square += np.square(values).sum(axis=0)

    def summary(self):
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        return mean, np.sqrt(variance), variance


def choose_evenly(values, count):
    values = list(values)
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[position] for position in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--subjects-per-dataset", type=int, default=2)
    parser.add_argument("--trials-per-subject", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.data)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    subjects = pd.read_csv(
        root / "subjects.csv",
        usecols=["subject_id", "subject_name", "source_dataset"],
    )
    split = pd.read_csv(root / "splits.csv")
    test_subjects = set(split.loc[split.fold == args.fold, "subject_id"].astype(int))
    val_fold = 1 if args.fold == 0 else 0
    val_subjects = set(split.loc[split.fold == val_fold, "subject_id"].astype(int)) - test_subjects
    all_subjects = set(subjects.subject_id.astype(int))
    train_subjects = all_subjects - test_subjects - val_subjects

    selected_subjects = []
    selected_by_dataset = {}
    for dataset, rows in subjects[subjects.subject_id.isin(train_subjects)].groupby("source_dataset"):
        chosen = choose_evenly(sorted(rows.subject_id.astype(int)), args.subjects_per_dataset)
        selected_by_dataset[dataset] = chosen
        selected_subjects.extend(chosen)
    if not selected_subjects or not set(selected_subjects).issubset(train_subjects):
        raise RuntimeError("feature audit attempted to select a non-training subject")

    sfreq = float(config["sfreq"])
    n_times = int(config["n_samples"])
    frequencies = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_masks = [(frequencies >= low) & (frequencies < high) for _name, low, high in RGNN_BANDS]
    diagnostic_masks = {
        "delta_1_4": (frequencies >= 1.0) & (frequencies < 4.0),
        "gamma_retained_30_40": (frequencies >= 30.0) & (frequencies < 40.0),
        "gamma_filtered_40_50": (frequencies >= 40.0) & (frequencies < 50.0),
    }

    node_power = RunningStats(len(RGNN_BANDS))
    node_log_power = RunningStats(len(RGNN_BANDS))
    trial_power = RunningStats(len(RGNN_BANDS))
    trial_log_power = RunningStats(len(RGNN_BANDS))
    diagnostic_power = RunningStats(len(diagnostic_masks))
    sampled_trials = 0
    all_masks_valid = True

    adj = np.load(root / "electrode_adjacency.npy")
    audit_adapter = RGNNAdapter(config["channel_order"], distance_adjacency=adj)
    audit_adapter.fit(
        [(0, np.zeros((1, 32, 750), np.float32), np.zeros(1, dtype=np.int64))],
        "0" * 64,
    )
    feature_model = RGNN(
        torch.from_numpy(adj),
        config["channel_order"],
        sfreq=sfreq,
        n_times=n_times,
        adapter=audit_adapter,
    ).eval()

    subject_index = subjects.set_index("subject_id")
    for subject_id in selected_subjects:
        if subject_id not in train_subjects:
            raise RuntimeError(f"refusing to open non-training subject {subject_id}")
        subject_name = subject_index.loc[subject_id, "subject_name"]
        with np.load(root / "data" / f"{subject_name}.npz", allow_pickle=False) as archive:
            n_trials = int(archive["X"].shape[0])
            trial_indices = np.asarray(choose_evenly(range(n_trials), args.trials_per_subject), dtype=int)
            signals = archive["X"][trial_indices].astype(np.float32, copy=False)
            channel_mask = archive["channel_mask"][trial_indices].astype(bool, copy=False)
        all_masks_valid = all_masks_valid and bool(channel_mask.all())

        mean = signals.mean(axis=-1, keepdims=True)
        std = signals.std(axis=-1, keepdims=True)
        signals = (signals - mean) / np.maximum(std, 1e-6)
        signals[~channel_mask] = 0.0
        spectrum = np.fft.rfft(signals, axis=-1)
        power = np.square(spectrum.real) + np.square(spectrum.imag)
        band_power = np.stack([power[..., mask].mean(axis=-1) for mask in band_masks], axis=-1)
        with torch.no_grad():
            band_log_power = feature_model.adapter.transform(torch.from_numpy(signals)).numpy()
        numpy_log_power = np.log(np.maximum(band_power, 1e-8))
        if not np.allclose(band_log_power, numpy_log_power, atol=2e-5, rtol=2e-5):
            raise RuntimeError("feature audit no longer matches RGNN adapter transform")

        node_power.add(band_power)
        node_log_power.add(band_log_power)
        trial_power.add(band_power.mean(axis=1))
        trial_log_power.add(band_log_power.mean(axis=1))
        diagnostic_power.add(np.stack([
            power[..., mask].mean(axis=-1) for mask in diagnostic_masks.values()
        ], axis=-1))
        sampled_trials += len(trial_indices)

    power_mean, power_std, _ = node_power.summary()
    log_mean, log_std, _ = node_log_power.summary()
    _trial_power_mean, trial_power_std, trial_power_variance = trial_power.summary()
    _trial_log_mean, trial_log_std, trial_log_variance = trial_log_power.summary()
    diagnostic_mean, diagnostic_std, _ = diagnostic_power.summary()

    per_band = {}
    for index, (name, low, high) in enumerate(RGNN_BANDS):
        per_band[name] = {
            "code_band_hz": [low, high],
            "fft_bins": int(band_masks[index].sum()),
            "mean_power": float(power_mean[index]),
            "power_std_across_trial_channels": float(power_std[index]),
            "power_variance_across_trial_means": float(trial_power_variance[index]),
            "mean_log_power_feature": float(log_mean[index]),
            "log_power_std_across_trial_channels": float(log_std[index]),
            "log_power_variance_across_trial_means": float(trial_log_variance[index]),
            "log_power_trial_mean_std": float(trial_log_std[index]),
        }

    report = {
        "status": "PASS",
        "scope": "training subjects only; no validation/test NPZ opened; no labels read",
        "fold": args.fold,
        "training_subject_count": len(train_subjects),
        "validation_subject_count_not_opened": len(val_subjects),
        "test_subject_count_not_opened": len(test_subjects),
        "sampled_training_subjects": selected_subjects,
        "sampled_training_subjects_by_dataset": selected_by_dataset,
        "sampled_trials": sampled_trials,
        "sampled_trial_channels": node_power.count,
        "all_sampled_channel_masks_valid": all_masks_valid,
        "input_shape_per_trial": [int(config["n_channels"]), n_times],
        "node_feature_shape_per_trial": [int(config["n_channels"]), len(RGNN_BANDS)],
        "benchmark_preprocessing_band_hz": [4.0, 40.0],
        "feature_definition": "per-trial/channel FFT mean band power followed by natural logarithm",
        "not_equivalent_claim": "band log-power surrogate for DE; no LDS smoothing",
        "per_band": per_band,
        "filter_conflict_diagnostics": {
            name: {
                "mean_power": float(diagnostic_mean[index]),
                "std_across_trial_channels": float(diagnostic_std[index]),
                "fft_bins": int(list(diagnostic_masks.values())[index].sum()),
            }
            for index, name in enumerate(diagnostic_masks)
        },
    }
    output = Path(args.out)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
