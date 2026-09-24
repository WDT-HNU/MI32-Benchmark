import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.channels.interpolation import _make_interpolation_matrix


TARGET_CHANNELS = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
ALPHA = 1e-5


LOADER = '''from pathlib import Path
import json
import numpy as np
import pandas as pd


class UEHFMI32Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.subjects = pd.read_csv(self.root / "subjects.csv")
        self.trials = pd.read_csv(self.root / "trials.csv", low_memory=False)
        self._index = self.trials.set_index("trial_id", drop=False)

    def load_subject(self, subject_id):
        row = self.subjects.loc[self.subjects.subject_id == int(subject_id)]
        if row.empty:
            raise KeyError(f"unknown subject_id: {subject_id}")
        name = row.iloc[0].subject_name
        with np.load(self.root / "data" / f"{name}.npz", allow_pickle=False) as z:
            return (
                z["X"].copy(),
                z["y"].copy(),
                z["channel_mask"].astype(bool).copy(),
                z["measured_mask"].astype(bool).copy(),
            )

    def load_trial(self, trial_id):
        if int(trial_id) not in self._index.index:
            raise KeyError(f"unknown trial_id: {trial_id}")
        row = self._index.loc[int(trial_id)]
        subject_id = int(row.subject_id)
        name = self.subjects.loc[self.subjects.subject_id == subject_id].iloc[0].subject_name
        idx = int(row.subject_trial_id)
        with np.load(self.root / "data" / f"{name}.npz", allow_pickle=False) as z:
            return (
                z["X"][idx].copy(),
                int(z["y"][idx]),
                z["channel_mask"][idx].astype(bool).copy(),
                z["measured_mask"][idx].astype(bool).copy(),
                subject_id,
            )

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, trial_id):
        return self.load_trial(trial_id)
'''


README = '''# MI-3-32-Interpolated

This is a self-contained 32-channel version of MI-3. Every trial has the same
ordered, symmetric 10-10 motor-imagery montage. A target channel that existed
in the source recording is copied bit-for-bit. A missing target channel is
estimated from all measured source channels with Perrin spherical-spline
interpolation using MNE-Python's standard_1005 coordinates (alpha=1e-5).

`channel_mask` is True for all 32 model-valid channels. `measured_mask` records
provenance: True means directly measured and False means interpolated. Models
must consume all 32 channels and may use `measured_mask` only as an optional
quality/provenance feature; it must not be used to zero interpolated values.
'''


def montage_positions(names):
    montage = mne.channels.make_standard_montage("standard_1005")
    lookup = {
        name.casefold(): xyz
        for name, xyz in montage.get_positions()["ch_pos"].items()
    }
    missing = [name for name in names if name.casefold() not in lookup]
    if missing:
        raise RuntimeError(f"channels without standard_1005 coordinates: {missing}")
    return np.asarray([lookup[name.casefold()] for name in names], dtype=np.float64)


def process_subject(in_file, source_names, source_positions, target_positions, matrix_cache):
    with np.load(in_file, allow_pickle=False) as z:
        X = z["X"]
        y = z["y"].astype(np.int64, copy=True)
        source_mask = z["channel_mask"].astype(bool, copy=False)
        if source_mask.ndim != 2 or not np.all(source_mask == source_mask[0]):
            raise RuntimeError(f"trial-varying or invalid channel mask: {in_file.name}")
        good = source_mask[0]
        good_idx = np.flatnonzero(good)
        if len(good_idx) < 4:
            raise RuntimeError(f"too few measured channels: {in_file.name}")

        source_lookup = {name.casefold(): i for i, name in enumerate(source_names)}
        target_src_idx = np.asarray(
            [source_lookup[name.casefold()] for name in TARGET_CHANNELS], dtype=np.int64
        )
        measured32 = good[target_src_idx]
        bad32_idx = np.flatnonzero(~measured32)
        key = np.packbits(good).tobytes()
        if key not in matrix_cache:
            W = _make_interpolation_matrix(
                source_positions[good], target_positions[bad32_idx], alpha=ALPHA
            ).astype(np.float32)
            matrix_cache[key] = W
        W = matrix_cache[key]

        result = X[:, target_src_idx, :].astype(np.float32, copy=True)
        if len(bad32_idx):
            for start in range(0, len(X), 32):
                stop = min(start + 32, len(X))
                result[start:stop, bad32_idx, :] = np.einsum(
                    "bg,tgs->tbs",
                    W,
                    X[start:stop, good_idx, :],
                    optimize=True,
                )

        direct_ok = bool(
            np.array_equal(
                result[:, measured32, :], X[:, target_src_idx[measured32], :]
            )
        )
        finite_ok = bool(np.isfinite(result).all())
        if len(bad32_idx):
            interp_std = result[:, bad32_idx, :].std(axis=(0, 2))
            interpolated_ok = bool(np.isfinite(interp_std).all() and np.all(interp_std > 0))
        else:
            interpolated_ok = True
        model_mask = np.ones((len(y), len(TARGET_CHANNELS)), dtype=bool)
        measured_mask = np.broadcast_to(measured32, model_mask.shape).copy()
        return result, y, model_mask, measured_mask, {
            "n_trials": int(len(y)),
            "n_source_measured": int(good.sum()),
            "n_target_measured": int(measured32.sum()),
            "n_target_interpolated": int((~measured32).sum()),
            "finite_ok": finite_ok,
            "direct_channels_preserved": direct_ok,
            "interpolated_channels_nonconstant": interpolated_ok,
            "status": "PASS" if finite_ok and direct_ok and interpolated_ok else "FAIL",
        }


def hash_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--smoke-subjects", type=int, default=0)
    args = ap.parse_args()
    src = args.source.resolve()
    if not (src / "data").is_dir():
        raise SystemExit(f"source data directory not found: {src / 'data'}")
    if len(TARGET_CHANNELS) != 32 or len(set(TARGET_CHANNELS)) != 32:
        raise RuntimeError("TARGET_CHANNELS must contain 32 unique names")

    source_config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    source_names = source_config["channel_order"]
    source_positions = montage_positions(source_names)
    target_positions = montage_positions(TARGET_CHANNELS)
    subjects = pd.read_csv(src / "subjects.csv")
    matrix_cache = {}

    if args.smoke_subjects:
        rows = []
        for row in subjects.head(args.smoke_subjects).itertuples(index=False):
            X, y, channel_mask, measured_mask, audit = process_subject(
                src / "data" / f"{row.subject_name}.npz",
                source_names,
                source_positions,
                target_positions,
                matrix_cache,
            )
            rows.append({"subject": row.subject_name, **audit})
            if X.shape != (len(y), 32, 750) or not channel_mask.all():
                raise RuntimeError(f"smoke shape/mask failure: {row.subject_name}")
            del X, y, channel_mask, measured_mask
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        print(json.dumps({"status": "PASS", "smoke_subjects": len(rows)}), flush=True)
        return

    if args.out is None:
        raise SystemExit("--out is required for a full build")
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out}")
    (out / "data").mkdir(parents=True)

    audit_rows = []
    started = time.time()
    for number, row in enumerate(subjects.itertuples(index=False), 1):
        in_file = src / "data" / f"{row.subject_name}.npz"
        X, y, channel_mask, measured_mask, audit = process_subject(
            in_file,
            source_names,
            source_positions,
            target_positions,
            matrix_cache,
        )
        np.savez_compressed(
            out / "data" / in_file.name,
            X=X,
            y=y,
            channel_mask=channel_mask,
            measured_mask=measured_mask,
        )
        audit_rows.append({
            "subject_id": int(row.subject_id),
            "subject_name": row.subject_name,
            "source_dataset": row.source_dataset,
            **audit,
        })
        elapsed = time.time() - started
        eta = elapsed / number * (len(subjects) - number)
        print(
            f"{number}/{len(subjects)} {row.subject_name} {audit['status']} "
            f"measured={audit['n_target_measured']}/32 eta_min={eta / 60:.1f}",
            flush=True,
        )
        del X, y, channel_mask, measured_mask

    for name in ["subjects.csv", "trials.csv", "provenance.csv", "splits.csv"]:
        shutil.copy2(src / name, out / name)
    pd.DataFrame(audit_rows).to_csv(out / "validation.csv", index=False)
    coverage = (
        pd.DataFrame(audit_rows)
        .groupby("source_dataset", as_index=False)
        .agg(
            n_subjects=("subject_id", "count"),
            min_measured=("n_target_measured", "min"),
            max_measured=("n_target_measured", "max"),
            interpolated=("n_target_interpolated", "first"),
        )
    )
    coverage.to_csv(out / "channel_coverage_by_dataset.csv", index=False)
    (out / "loader.py").write_text(LOADER, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

    config = dict(source_config)
    config.update({
        "dataset_name": "UEHF-MI Three-Class Common-32 Montage",
        "version": "4.0-common32",
        "build_status": "RELEASE" if all(x["status"] == "PASS" for x in audit_rows) else "FAIL",
        "n_channels": 32,
        "channel_order": TARGET_CHANNELS,
        "standard_channel_set": "predefined 32-electrode standard_1005 subset shared with PooledMI binary",
        "missing_channel_strategy": "Perrin spherical-spline interpolation from all measured source channels",
        "interpolation_method": "MNE-Python spherical spline interpolation matrix",
        "interpolation_montage": "standard_1005",
        "interpolation_regularization_alpha": ALPHA,
        "data_storage": "self_contained_npz",
        "channel_mask": True,
        "channel_mask_semantics": "True=model-valid channel; fixed True for sealed MI-32, never used as validity mask",
        "measured_mask": True,
        "measured_mask_semantics": "True=original measured source channel; False=interpolated target channel",
        "zero_fill": False,
        "derived_from": src.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": config["build_status"],
        "subjects": int(len(subjects)),
        "trials": int(subjects.n_trials.sum()),
        "channels": 32,
        "unique_source_channel_patterns": int(len(matrix_cache)),
        "method": config["missing_channel_strategy"],
    }
    (out / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checksum_files = sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    )
    with (out / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as f:
        for number, path in enumerate(checksum_files, 1):
            f.write(f"{hash_file(path)}  {path.relative_to(out).as_posix()}\n")
            if number % 25 == 0 or number == len(checksum_files):
                print(f"checksums {number}/{len(checksum_files)}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
