#!/usr/bin/env python3
"""Download checkpoint artifacts atomically and verify byte length plus SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, record: dict) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing checkpoint: {path}")
    size = path.stat().st_size
    digest = sha256(path)
    if size != int(record["bytes"]) or digest != record["sha256"]:
        raise RuntimeError(
            f"checkpoint mismatch: {path} size={size}/{record['bytes']} sha256={digest}/{record['sha256']}"
        )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "MI32-Benchmark/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model slugs; default: all models with checkpoints")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--third-party", type=Path, default=ROOT / "third_party")
    args = parser.parse_args()

    registry = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"]
    selected = args.models or [name for name, entry in registry.items() if entry.get("checkpoint")]
    for slug in selected:
        entry = registry.get(slug)
        if entry is None or not entry.get("checkpoint"):
            raise SystemExit(f"model has no downloadable checkpoint record: {slug}")
        record = entry["checkpoint"]
        destination = args.out / record["path"]
        try:
            verify(destination, record)
            print(f"PASS existing {slug}: {destination}")
        except (RuntimeError, FileNotFoundError):
            print(f"download {slug}: {record['url']}", flush=True)
            download(record["url"], destination)
            verify(destination, record)
            print(f"PASS downloaded {slug}: {destination}")

        author = entry.get("author_checkpoint")
        if author:
            source = args.third_party / author["source_repo_path"]
            target = args.out / author["path"]
            if not target.exists():
                if not source.is_file():
                    raise SystemExit(
                        f"missing author checkpoint {source}; run fetch_upstreams.py and git lfs pull in LaBraM"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            verify(target, author)
            print(f"PASS author checkpoint {slug}: {target}")

    print("CHECKPOINTS_PASS")


if __name__ == "__main__":
    main()
