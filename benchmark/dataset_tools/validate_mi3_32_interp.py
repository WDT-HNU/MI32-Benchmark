import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path, block=8 * 1024 * 1024):
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
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    dataset = args.dataset.resolve()
    source_cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    channels = config["channel_order"]
    source_lookup = {name.casefold(): i for i, name in enumerate(source_cfg["channel_order"])}
    target_src = np.asarray([source_lookup[name.casefold()] for name in channels])
    subjects = pd.read_csv(dataset / "subjects.csv")
    trials = pd.read_csv(dataset / "trials.csv", low_memory=False)
    errors = []
    class_counts = np.zeros(3, dtype=np.int64)
    total_bytes = 0

    for number, row in enumerate(subjects.itertuples(index=False), 1):
        out_file = dataset / "data" / f"{row.subject_name}.npz"
        src_file = source / "data" / f"{row.subject_name}.npz"
        with np.load(out_file, allow_pickle=False) as z, np.load(src_file, allow_pickle=False) as s:
            required = {"X", "y", "channel_mask", "measured_mask"}
            keys_ok = required.issubset(z.files)
            if not keys_ok:
                errors.append(f"{row.subject_name}: keys={z.files}")
                continue
            X, y = z["X"], z["y"]
            model_mask = z["channel_mask"].astype(bool)
            measured = z["measured_mask"].astype(bool)
            source_mask = s["channel_mask"].astype(bool)[0]
            expected_measured = source_mask[target_src]
            shape_ok = X.shape == (len(y), 32, 750) and model_mask.shape == measured.shape == (len(y), 32)
            labels_ok = np.array_equal(y, s["y"])
            masks_ok = bool(
                model_mask.all()
                and np.all(measured == measured[0])
                and np.array_equal(measured[0], expected_measured)
            )
            finite_ok = bool(np.isfinite(X).all())
            direct_ok = bool(np.array_equal(X[:, expected_measured, :], s["X"][:, target_src[expected_measured], :]))
            bad = ~expected_measured
            interp_ok = bool(not bad.any() or np.all(X[:, bad, :].std(axis=(0, 2)) > 0))
            meta = trials.loc[trials.subject_id == int(row.subject_id)].sort_values("subject_trial_id")
            trial_meta_ok = bool(
                len(meta) == len(y)
                and np.array_equal(meta.subject_trial_id.to_numpy(), np.arange(len(y)))
                and np.array_equal(meta.label.to_numpy(), y)
            )
            ok = shape_ok and labels_ok and masks_ok and finite_ok and direct_ok and interp_ok and trial_meta_ok
            if not ok:
                errors.append({
                    "subject": row.subject_name,
                    "shape": shape_ok,
                    "labels": labels_ok,
                    "masks": masks_ok,
                    "finite": finite_ok,
                    "direct": direct_ok,
                    "interpolated": interp_ok,
                    "trial_meta": trial_meta_ok,
                })
            class_counts += np.bincount(y, minlength=3)
        total_bytes += out_file.stat().st_size
        if number % 20 == 0 or number == len(subjects):
            print(f"validated {number}/{len(subjects)} errors={len(errors)}", flush=True)

    copied_files = ["subjects.csv", "trials.csv", "provenance.csv", "splits.csv"]
    metadata_identical = {
        name: sha256(source / name) == sha256(dataset / name) for name in copied_files
    }
    summary = {
        "status": "PASS" if not errors and all(metadata_identical.values()) else "FAIL",
        "subjects": int(len(subjects)),
        "trials": int(class_counts.sum()),
        "channels": int(len(channels)),
        "class_counts": class_counts.tolist(),
        "expected_class_counts": [24402, 24402, 48804],
        "metadata_identical_to_source": metadata_identical,
        "npz_bytes": int(total_bytes),
        "errors": errors,
    }
    if summary["class_counts"] != summary["expected_class_counts"]:
        summary["status"] = "FAIL"
    checksum_targets = [
        p
        for p in dataset.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.tmp"}
    ]
    # independent_validation.json is written immediately below and is therefore
    # also part of the final sealed package.
    if not (dataset / "independent_validation.json").exists():
        checksum_target_count = len(checksum_targets) + 1
    else:
        checksum_target_count = len(checksum_targets)
    summary["sha256_entries"] = int(checksum_target_count)
    (dataset / "independent_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checksum_targets = sorted(
        p
        for p in dataset.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.tmp"}
    )
    checksum_tmp = dataset / "SHA256SUMS.tmp"
    with checksum_tmp.open("w", encoding="utf-8", newline="\n") as f:
        for number, path in enumerate(checksum_targets, 1):
            f.write(f"{sha256(path)}  {path.relative_to(dataset).as_posix()}\n")
            if number % 25 == 0 or number == len(checksum_targets):
                print(f"sealed checksums {number}/{len(checksum_targets)}", flush=True)
    checksum_tmp.replace(dataset / "SHA256SUMS")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
