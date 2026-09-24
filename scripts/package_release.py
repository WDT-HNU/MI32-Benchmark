#!/usr/bin/env python3
"""Copy auditable run artifacts into a hash-sealed release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_NAMES = {
    "results.csv", "history.csv", "run_config.json", "run_manifest.json",
    "model_results.csv", "test_summary.csv", "audit.json", "environment_audit.json",
}
ALLOWED_PREFIXES = ("adapter_manifest_", "source_snapshot")


def wanted(relative: Path) -> bool:
    return relative.name in ALLOWED_NAMES or any(part.startswith(ALLOWED_PREFIXES) for part in relative.parts)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise SystemExit(f"refusing to overwrite release: {destination}")
    destination.mkdir(parents=True)

    copied = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if wanted(relative):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(target)
    if not copied:
        raise SystemExit("no auditable artifacts found")

    manifest_rows = []
    for path in sorted(copied):
        manifest_rows.append(f"{digest(path)}  {path.relative_to(destination).as_posix()}")
    (destination / "SHA256SUMS").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (destination / "RELEASE.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "files": len(copied),
        "policy": "weights, datasets, secrets, and private URLs excluded",
    }, indent=2), encoding="utf-8")
    print(f"RELEASE_READY {destination} files={len(copied)}")


if __name__ == "__main__":
    main()
