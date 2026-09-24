#!/usr/bin/env python3
"""Pre-publication checks for integrity, accidental large files, and secret-shaped text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "third_party", "checkpoints", "outputs", "full", "__pycache__"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "embedded AutoDL password": re.compile(r"AUTODL_PASSWORD\s*=\s*['\"][^'\"]+['\"]"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tracked_candidates() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true", help="also fail on OWNER/PENDING placeholders")
    args = parser.parse_args()
    errors = []
    warnings = []
    files = tracked_candidates()

    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"large file over 50 MiB: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            warnings.append(f"large file over 5 MiB: {relative}")
        if path.stat().st_size <= 2 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} pattern in {relative}")
            if args.release and path.resolve() != Path(__file__).resolve() and (
                "github.com/OWNER/" in text or ",PENDING," in text
            ):
                errors.append(f"release placeholder in {relative}")

    data_manifest = ROOT / "datasets" / "mi32" / "metadata" / "SHA256SUMS"
    if digest(data_manifest) != "1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06":
        errors.append("dataset metadata manifest identity changed")

    result_manifest = (
        ROOT / "results" / "mi32-common32-v4" /
        "historical-fold0-pre-unit-hardening" / "final_csv" / "SHA256SUMS"
    )
    for line in result_manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = result_manifest.parent / relative
        if not target.is_file() or digest(target) != expected:
            errors.append(f"result package mismatch: {relative}")

    current_results = ROOT / "results" / "mi32-common32-v4" / "current-partial-20260911"
    for result_csv in sorted(current_results.glob("*_results.csv")):
        manifest = result_csv.with_name(result_csv.name.replace("_results.csv", "_run_manifest.json"))
        if not manifest.is_file():
            errors.append(f"current result lacks paired manifest: {result_csv.name}")
            continue
        with result_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if {row.get("split") for row in rows} != {"val", "test"}:
            errors.append(f"current result must contain exactly val/test splits: {result_csv.name}")
        if any(row.get("dataset_manifest_sha256") != digest(data_manifest) for row in rows):
            errors.append(f"dataset identity mismatch in current result: {result_csv.name}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        adapter = payload.get("adapter_manifest", {})
        if result_csv.name.startswith(("current_eegmamba_", "current_codebrain_")):
            output_contract = adapter.get("output_contract", {})
            if output_contract.get("unit") != "uV/100" or output_contract.get("unit_scale_from_volts") != 10_000.0:
                errors.append(f"foundation unit contract mismatch: {manifest.name}")

    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "files_checked": len(files),
        "errors": errors,
        "warnings": warnings,
    }, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
