#!/usr/bin/env python3
"""Portable front door for MI32 diagnostics and single-model runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATA_MANIFEST = "1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry() -> dict:
    return json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))["models"]


def git_identity(repo: Path) -> tuple[str | None, str | None]:
    if not (repo / ".git").is_dir():
        return None, None
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
        text=True,
        capture_output=True,
    )
    return (commit.stdout.strip() if commit.returncode == 0 else None,
            dirty.stdout.strip() if dirty.returncode == 0 else None)


def doctor(args: argparse.Namespace) -> int:
    models = registry()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "dataset": {},
        "upstreams": {},
        "checkpoints": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - environment diagnosis
        errors.append(f"torch import failed: {exc}")

    data = args.data.resolve()
    manifest = data / "SHA256SUMS"
    if manifest.is_file():
        actual = sha256(manifest)
        report["dataset"] = {"root": str(data), "manifest_sha256": actual}
        if actual != EXPECTED_DATA_MANIFEST:
            errors.append(f"dataset manifest mismatch: {actual}")
    else:
        errors.append(f"dataset manifest missing: {manifest}")

    for slug, entry in models.items():
        repo = args.third_party / entry.get("repo_dir", slug)
        commit, dirty = git_identity(repo)
        status = commit == entry["commit"] and dirty == ""
        report["upstreams"][slug] = {  # type: ignore[index]
            "path": str(repo), "expected": entry["commit"], "actual": commit, "clean": dirty == "", "ok": status
        }
        if not status:
            errors.append(f"upstream not ready: {slug}")
        checkpoint = entry.get("checkpoint")
        if checkpoint:
            path = args.checkpoints / checkpoint["path"]
            ok = path.is_file() and path.stat().st_size == checkpoint["bytes"] and sha256(path) == checkpoint["sha256"]
            report["checkpoints"][slug] = {"path": str(path), "ok": ok}  # type: ignore[index]
            if not ok:
                errors.append(f"checkpoint not ready: {slug}")

    report["status"] = "PASS" if not errors else "FAIL"
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def command_for(args: argparse.Namespace) -> list[str]:
    entry = registry()[args.model]
    batch = args.batch_size or entry["default_batch_size"]
    lr = args.lr if args.lr is not None else entry["default_learning_rate"]
    common = [
        "--data", str(args.data.resolve()),
        "--out", str(args.output.resolve()),
        "--fold", str(args.fold),
        "--epochs", str(args.epochs),
        "--batch-size", str(batch),
        "--lr", str(lr),
    ]
    if args.limit:
        common += ["--max-trials-per-subject", str(args.limit)]
    if args.preload:
        common.append("--preload")
    if args.balanced_loss:
        common.append("--balanced-loss")
    if args.final_test:
        common.append("--final-test")
    if args.preflight:
        common.append("--preflight")

    if entry["runner"] == "deep":
        return [sys.executable, str(ROOT / "mi3_eegnet.py"), "--model", entry["runner_model"], *common]
    if entry["runner"] == "foundation":
        checkpoint = args.checkpoints / entry["checkpoint"]["path"]
        command = [
            sys.executable, str(ROOT / "mi3_foundation.py"),
            "--model", entry["runner_model"],
            "--weights", str(checkpoint.resolve()),
            *common,
        ]
        if entry.get("repo_dir") and args.model != "labram":
            command += ["--repo", str((args.third_party / entry["repo_dir"]).resolve())]
        return command
    if entry["runner"] == "uni_ntfm":
        if not args.allow_protocol_benchmark:
            raise SystemExit(
                "Uni-NTFM is a protocol benchmark, not paper reproduction; add --allow-protocol-benchmark"
            )
        return [
            sys.executable, str(ROOT / "uni_mi3_supervised.py"),
            "--repo", str((args.third_party / entry["repo_dir"]).resolve()),
            "--allow-custom-adapter",
            *common,
        ]
    raise SystemExit(f"unknown runner type: {entry['runner']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    diag = subparsers.add_parser("doctor")
    diag.add_argument("--data", type=Path, required=True)
    diag.add_argument("--third-party", type=Path, default=ROOT / "third_party")
    diag.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", choices=sorted(registry()), required=True)
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--third-party", type=Path, default=ROOT / "third_party")
    run_parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    run_parser.add_argument("--fold", type=int, default=0)
    run_parser.add_argument("--epochs", type=int, default=5)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--lr", type=float)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--preload", action="store_true")
    run_parser.add_argument("--balanced-loss", action="store_true")
    run_parser.add_argument("--final-test", action="store_true")
    run_parser.add_argument("--preflight", action="store_true")
    run_parser.add_argument("--allow-protocol-benchmark", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.action == "doctor":
        raise SystemExit(doctor(args))
    command = command_for(args)
    print(shlex.join(command))
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
