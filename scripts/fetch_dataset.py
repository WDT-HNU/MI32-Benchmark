#!/usr/bin/env python3
"""Fetch an approved MI32 archive, extract safely, and verify the sealed release."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if not inside(destination, destination / member.filename):
                    raise RuntimeError(f"unsafe archive path: {member.filename}")
            package.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as package:
            for member in package.getmembers():
                if not inside(destination, destination / member.name):
                    raise RuntimeError(f"unsafe archive path: {member.name}")
            package.extractall(destination, filter="data")
        return
    raise RuntimeError(f"unsupported archive: {archive}")


def download(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("", "file"):
        source = Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else url)
        shutil.copy2(source, destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "MI32-Benchmark/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def locate_release(extracted: Path) -> Path:
    candidates = [extracted] + [path.parent for path in extracted.rglob("SHA256SUMS")]
    matches = [path for path in candidates if (path / "config.json").is_file() and (path / "data").is_dir()]
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected one dataset root, found {unique}")
    return unique[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("MI32_DATA_URL"))
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "mi32" / "full")
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("provide --url or MI32_DATA_URL; no private endpoint is embedded")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mi32-download-") as temporary_name:
        temporary = Path(temporary_name)
        archive = temporary / "dataset.archive"
        extracted = temporary / "extracted"
        extracted.mkdir()
        print(f"downloading {args.url}", flush=True)
        download(args.url, archive)
        safe_extract(archive, extracted)
        release = locate_release(extracted)
        subprocess.run(
            [os.fspath(Path(os.sys.executable)), os.fspath(ROOT / "scripts" / "verify_dataset.py"), os.fspath(release), "--full"],
            check=True,
        )
        shutil.move(os.fspath(release), os.fspath(args.out))
    print(f"DATASET_READY {args.out.resolve()}")


if __name__ == "__main__":
    main()
