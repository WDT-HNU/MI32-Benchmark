import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import torch


REPO = Path(os.environ.get(
    "UNINTFM_TEST_REPO",
    str(Path(__file__).resolve().parents[1] / "third_party" / "Uni-NTFM"),
))
ADAPTER = Path(os.environ.get(
    "UNINTFM_TEST_ADAPTER",
    str(Path(__file__).resolve().parents[1] / "uni_mi3_supervised.py"),
))
EXPECTED_COMMIT = "c0ce0152f94366e59b31b3fb2c108ce909bcd95c"
EXPECTED_SOURCE_SHA256 = {
    "model.py": "05e471cc46fb301b0eb6192519ff9e72fe82e9a90aaf1571b20d77558a1eca67",
    "load.py": "48b88244a5ad16c3b209f2be81104846e1fcb19e6a445664193298e831bee85e",
    "train.py": "fecc6ce b740e84392d4728c2006e8649233380cedc7f33f7e41c4277dfc0c09e".replace(" ", ""),
    "README.md": "32e0ffcdec9d9e10beda04c613041a2ace06fe5e06e153638b0003aaf6f3532e",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git():
    """Resolve git from PATH on Linux, macOS, or Windows."""
    import shutil

    exe = shutil.which("git")
    if exe:
        return exe
    raise RuntimeError("git executable not found; Uni-NTFM repository identity cannot be verified")


def check_repo_identity():
    git = _git()
    commit = subprocess.check_output(
        [git, "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == EXPECTED_COMMIT, (EXPECTED_COMMIT, commit)
    assert not subprocess.check_output(
        [git, "-C", str(REPO), "status", "--short", "--untracked-files=no"],
        text=True,
    ).strip()
    # Platform-independent identity: the working-tree file must match the
    # official git blob, allowing only the autocrlf line-ending translation
    # (CRLF on Windows checkouts, LF on Linux). The fixed EXPECTED_SOURCE_SHA256
    # above records the Windows-checkout digest; the blob comparison is the
    # authoritative check across platforms.
    for name in EXPECTED_SOURCE_SHA256:
        blob = subprocess.check_output([git, "-C", str(REPO), "show", f"HEAD:{name}"])
        work = (REPO / name).read_bytes()
        if work != blob and work.replace(b"\r\n", b"\n") != blob:
            raise AssertionError(f"{name} does not match the official git blob")


def load_official_model():
    sys.path.insert(0, str(REPO))
    return importlib.import_module("model")


def check_official_geometry(model_module):
    config = model_module.ModelConfig()
    assert (config.num_regions, config.max_electrodes_per_region) == (5, 24)
    assert config.sequence_length == 1600

    config.embed_dim = 192
    config.num_heads = 6
    config.depth = 1
    config.max_electrodes_per_region = 4
    config.use_moe = False
    config.sequence_length = 64
    model = model_module.DualDomainTransformerMEM(config).eval()
    x = torch.randn(1, 5, 4, 64)
    padding = torch.zeros(1, 5, 4, dtype=torch.bool)
    with torch.no_grad():
        outputs = model(x, padding_mask=padding)
    assert [tuple(value.shape) for value in outputs[:4]] == [
        (1, 20, 64),
        (1, 20, 64),
        (1, 20, 5),
        (1, 20, 5),
    ]
    assert not hasattr(model, "classifier")
    assert not hasattr(model, "num_classes")


def check_adapter_is_explicitly_custom():
    source = ADAPTER.read_text(encoding="utf-8")
    assert "Per measured channel z-normalization" not in source
    assert "--allow-custom-adapter" in source
    assert "protocol-preserving supervised adapter" in source
    assert "protocol_benchmark" in source
    assert "official_model_sequence_length" in source


def main():
    check_repo_identity()
    model_module = load_official_model()
    check_official_geometry(model_module)
    check_adapter_is_explicitly_custom()
    print("Uni-NTFM structure audit: 3/3 PASS")


if __name__ == "__main__":
    main()
