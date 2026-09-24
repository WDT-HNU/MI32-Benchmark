import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


_ORIGINAL_TENSOR_CUDA = torch.Tensor.cuda


def _noop_cuda(self, *args, **kwargs):
    return self


setattr(torch.Tensor, "cuda", _noop_cuda)


def _install_optional_audio_stub_for_architecture_audit():
    try:
        import torchaudio  # noqa: F401
        return
    except Exception:
        pass

    class _IdentityTransform(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    def _identity_function(x, *args, **kwargs):
        return x

    torchaudio = types.ModuleType("torchaudio")
    functional = types.ModuleType("torchaudio.functional")
    transforms = types.ModuleType("torchaudio.transforms")
    for name in ("fftconvolve", "filtfilt", "lfilter", "resample"):
        setattr(functional, name, _identity_function)
    for name in ("Resample", "Spectrogram", "MelSpectrogram", "AmplitudeToDB"):
        setattr(transforms, name, _IdentityTransform)
    torchaudio.__version__ = "architecture-audit-stub"
    torchaudio.functional = functional
    torchaudio.transforms = transforms
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


_install_optional_audio_stub_for_architecture_audit()


try:
    import mi3_deep_models  # noqa: F401
except ModuleNotFoundError:
    import remote_mi3_deep_models as mi3_deep_models
    sys.modules["mi3_deep_models"] = mi3_deep_models

from model_adapters.codebrain import CodeBrainAdapter
from mi3_foundation import (
    CODEBRAIN_OFFICIAL_COMMIT,
    CODEBRAIN_REPOSITORY,
    CODEBRAIN_SOURCE_FILES,
    CodeBrainClassifier,
    sha256_file,
    snapshot_codebrain_sources,
    validate_codebrain_structure,
)


MI32_CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]


def _looks_like_codebrain_repo(path):
    path = Path(path)
    return path.is_dir() and all((path / relative).is_file() for relative in CODEBRAIN_SOURCE_FILES)


def _default_codebrain_repo():
    candidates = []
    for root in (Path(__file__).resolve().parent, Path.cwd()):
        candidates.extend((
            root / "CodeBrain",
            root / "codebrain_repo",
            root / "work" / "codebrain_repo",
        ))
        candidates.extend(sorted(root.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/work/codebrain_repo")))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_codebrain_repo(resolved):
            return resolved
    raise FileNotFoundError(
        "Set CODEBRAIN_TEST_REPO to a frozen CodeBrain checkout with the audited source files."
    )


CODEBRAIN_REPO = Path(os.environ.get("CODEBRAIN_TEST_REPO") or _default_codebrain_repo())
DEFAULT_CHECKPOINT = CODEBRAIN_REPO / "Checkpoints" / "CodeBrain.pth"


class CodeBrainStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = CODEBRAIN_REPO
        checkpoint_override = os.environ.get("CODEBRAIN_TEST_CHECKPOINT")
        if checkpoint_override:
            cls.checkpoint = Path(checkpoint_override)
        elif DEFAULT_CHECKPOINT.is_file():
            cls.checkpoint = DEFAULT_CHECKPOINT
        else:
            raise RuntimeError(
                "CodeBrain checkpoint missing; rerun is blocked until the official checkpoint is present"
            )
        cls.require_verified_checkpoint = True
        adapter = CodeBrainAdapter(MI32_CHANNEL_ORDER)
        fake = torch.randn(2, 32, 750)
        adapter.fit([(0, fake.numpy(), torch.zeros(2, dtype=torch.long).numpy())], "0" * 64)
        cls.model = CodeBrainClassifier(
            MI32_CHANNEL_ORDER,
            cls.repo,
            cls.checkpoint,
            verify_checkpoint_identity=cls.require_verified_checkpoint,
            adapter=adapter,
        )

    @classmethod
    def tearDownClass(cls):
        setattr(torch.Tensor, "cuda", _ORIGINAL_TENSOR_CUDA)

    def test_repository_identity(self):
        report = self.model.load_report["source"]
        self.assertEqual(report["commit"], CODEBRAIN_OFFICIAL_COMMIT)
        self.assertEqual(report["repo"], str(CODEBRAIN_REPO.resolve()))
        self.assertEqual(set(report["source_sha256"]), set(CODEBRAIN_SOURCE_FILES))

    def test_checkpoint_identity_and_load(self):
        report = self.model.load_report
        if self.require_verified_checkpoint:
            self.assertTrue(report["checkpoint_identity_verified"])
        self.assertEqual(report["checkpoint_missing"], [])
        self.assertEqual(report["checkpoint_unexpected"], [])
        self.assertEqual(report["shape_mismatch"], {})
        self.assertEqual(report["numerical_load_max_abs_error"], 0.0)
        self.assertEqual(report["backbone_coverage"], 1.0)
        self.assertEqual(report["exact_loaded_parameter_count"], report["checkpoint_parameter_count"])

    def test_no_adapter_and_flatten_head(self):
        self.assertFalse(hasattr(self.model, "channel_adapter"))
        self.assertIsNone(self.model.load_report["channel_adapter"])
        self.assertEqual(self.model.channel_order, MI32_CHANNEL_ORDER)
        self.assertEqual(self.model.classifier[0].in_features, 19_200)
        self.assertEqual(
            [(m.in_features, m.out_features) for m in self.model.classifier if isinstance(m, nn.Linear)],
            [(19_200, 600), (600, 200), (200, 3)],
        )

    def test_input_preparation_and_forward_path(self):
        audit = validate_codebrain_structure(
            self.model,
            batch_size=2,
            run_full_forward=True,
            require_verified_checkpoint=self.require_verified_checkpoint,
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["input_shape"], [2, 32, 750])
        self.assertEqual(audit["resampled_shape"], [2, 32, 600])
        self.assertEqual(audit["patch_shape"], [2, 32, 3, 200])
        self.assertEqual(audit["classifier_input"], [2, 19_200])
        self.assertEqual(audit["output_shape"], [2, 3])

    def test_source_snapshot_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_codebrain_sources(Path(directory), self.model)
            self.assertEqual(report["official_commit"], CODEBRAIN_OFFICIAL_COMMIT)
            self.assertEqual(report["official_repository"], CODEBRAIN_REPOSITORY)
            for relative in CODEBRAIN_SOURCE_FILES:
                self.assertEqual(report["official_sources"][relative]["sha256"], sha256_file(report["official_sources"][relative]["path"]))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CodeBrainStructureTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "checkpoint": str(CodeBrainStructureTest.checkpoint),
    }, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
