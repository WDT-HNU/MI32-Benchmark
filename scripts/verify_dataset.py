#!/usr/bin/env python3
"""Verify a metadata-only snapshot or the complete sealed MI32 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST = "1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06"
EXPECTED_COUNTS = {"subjects": 230, "trials": 97608, "channels": 32}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        rows.append((expected, relative.strip().lstrip("*")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--full", action="store_true", help="require and hash every manifest entry")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}")
    manifest_hash = digest(manifest)
    if manifest_hash != EXPECTED_MANIFEST:
        raise SystemExit(f"manifest identity mismatch: {manifest_hash}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    observed = {
        "subjects": int(config["n_subjects"]),
        "trials": int(config["n_trials"]),
        "channels": int(config["n_channels"]),
    }
    if config.get("version") != "4.0-common32" or observed != EXPECTED_COUNTS:
        raise SystemExit(f"dataset contract mismatch: version={config.get('version')} counts={observed}")

    checked = 0
    missing = []
    mismatched = []
    for expected, relative in parse_manifest(manifest):
        target = root / relative
        if not target.is_file():
            missing.append(relative)
            continue
        checked += 1
        actual = digest(target)
        if actual != expected:
            mismatched.append((relative, expected, actual))
    if mismatched:
        raise SystemExit(f"hash mismatch: {mismatched[:5]}")
    if args.full and missing:
        raise SystemExit(f"full release is incomplete; missing {len(missing)} files, first={missing[:5]}")

    mode = "full" if not missing else "metadata-only"
    print(json.dumps({
        "status": "PASS",
        "mode": mode,
        "manifest_sha256": manifest_hash,
        "checked_files": checked,
        "missing_manifest_entries": len(missing),
        **observed,
    }, indent=2))


if __name__ == "__main__":
    main()
