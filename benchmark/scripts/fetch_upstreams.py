#!/usr/bin/env python3
"""Clone and freeze all model repositories declared in configs/models.json."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model slugs; default: all entries")
    parser.add_argument("--out", type=Path, default=ROOT / "third_party")
    args = parser.parse_args()

    registry = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"]
    selected = args.models or list(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    for slug in selected:
        entry = registry[slug]
        repository = entry["repository"]
        commit = entry["commit"]
        destination = args.out / entry.get("repo_dir", slug)
        identity = (repository, commit)
        if identity in seen:
            continue
        seen.add(identity)

        if not destination.exists():
            print(f"clone {slug}: {repository} -> {destination}", flush=True)
            run(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)])
        if not (destination / ".git").is_dir():
            raise SystemExit(f"refusing non-git destination: {destination}")

        run(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
        run(["git", "checkout", "--detach", commit], cwd=destination)
        actual = run(["git", "rev-parse", "HEAD"], cwd=destination)
        dirty = run(["git", "status", "--short", "--untracked-files=no"], cwd=destination)
        if actual != commit or dirty:
            raise SystemExit(f"identity check failed for {slug}: expected={commit} actual={actual} dirty={dirty!r}")
        print(f"PASS {slug} {actual}")

    print("UPSTREAMS_PASS")


if __name__ == "__main__":
    main()
