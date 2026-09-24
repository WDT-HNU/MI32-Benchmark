import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

import mi3_foundation
from model_adapters.labram import LaBraMAdapter
from mi3_foundation import (
    LABRAM_BRAINDECODE_VERSION,
    LABRAM_AUTHOR_CHECKPOINT_BYTES,
    LABRAM_AUTHOR_CHECKPOINT_SHA256,
    LABRAM_CHECKPOINT_BYTES,
    LABRAM_CHECKPOINT_SHA256,
    LaBraMClassifier,
    LaBraMMI3Trials,
    _apply_sinc_resample,
    _sinc_resample_kernel,
    sha256_file,
    snapshot_labram_sources,
    validate_labram_structure,
)


MI32_CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

EXPECTED_EMBEDDING_IDS = [
    1, 3, 7, 11, 16, 18, 20, 22, 24, 28, 29, 31, 33, 34, 38, 40,
    42, 44, 46, 50, 51, 53, 55, 56, 60, 62, 64, 66, 68, 81, 82, 83,
]


def _build_synthetic_converted_checkpoint(path):
    from braindecode.models import Labram

    torch.manual_seed(20260817)
    source = Labram(
        n_times=3000,
        n_outputs=0,
        n_chans=128,
        sfreq=200,
        patch_size=200,
        learned_patcher=False,
        embed_dim=200,
        conv_in_channels=1,
        conv_out_channels=8,
        num_layers=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=nn.LayerNorm,
        drop_prob=0.0,
        attn_drop_prob=0.0,
        drop_path_prob=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.1,
        use_abs_pos_emb=True,
        use_mean_pooling=False,
        init_scale=0.001,
        neural_tokenizer=True,
        activation=nn.GELU,
    )
    state = source.state_dict()
    # Make the temporal-axis test independent from random initialization.
    temporal = torch.arange(16, dtype=torch.float32)[None, :, None].expand(1, 16, 200).clone()
    state["temporal_embedding"] = temporal
    torch.save(state, path)
    return state


class LaBraMStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("LABRAM_AUDIT_ALLOW_DUPLICATE_DIST_INFO") == "1":
            # The local audit target contains both old and current dist-info while
            # the imported source is 1.7.0. Formal AutoDL validation never sets this.
            mi3_foundation.package_version = lambda _name: LABRAM_BRAINDECODE_VERSION
        cls.temp = tempfile.TemporaryDirectory()
        real_checkpoint = os.environ.get("LABRAM_TEST_CHECKPOINT")
        if real_checkpoint:
            cls.checkpoint = Path(real_checkpoint)
            cls.raw_state = torch.load(cls.checkpoint, map_location="cpu", weights_only=False)
            cls.verify_identity = True
        else:
            cls.checkpoint = Path(cls.temp.name) / "synthetic_converted_checkpoint.pt"
            cls.raw_state = _build_synthetic_converted_checkpoint(cls.checkpoint)
            cls.verify_identity = False
        adapter = LaBraMAdapter(MI32_CHANNEL_ORDER)
        fake = torch.randn(2, 32, 750)
        adapter.fit([(0, fake.numpy(), torch.zeros(2, dtype=torch.long).numpy())], "0" * 64)
        cls.model = LaBraMClassifier(
            MI32_CHANNEL_ORDER,
            cls.checkpoint,
            verify_checkpoint_identity=cls.verify_identity,
            adapter=adapter,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_checkpoint_identity_and_numerical_coverage(self):
        report = self.model.load_report
        if self.verify_identity:
            self.assertEqual(self.checkpoint.stat().st_size, LABRAM_CHECKPOINT_BYTES)
            self.assertEqual(sha256_file(self.checkpoint), LABRAM_CHECKPOINT_SHA256)
            self.assertTrue(report["checkpoint_identity_verified"])
        self.assertEqual(report["checkpoint_parameter_count"], 5_819_936)
        self.assertEqual(report["model_parameter_count"], 5_817_939)
        self.assertEqual(report["backbone_parameter_count_excluding_new_head"], 5_817_336)
        self.assertEqual(report["exact_loaded_parameter_count"], 5_816_736)
        self.assertEqual(report["temporally_adapted_parameter_count"], 600)
        self.assertEqual(report["backbone_coverage"], 1.0)
        self.assertEqual(report["numerical_load_all_non_temporal_max_abs_error"], 0.0)
        self.assertEqual(report["checkpoint_missing"], ["final_layer.weight", "final_layer.bias"])
        self.assertEqual(report["checkpoint_unexpected"], [])
        self.assertEqual(report["reinitialized_parameters"], {
            "final_layer.weight": 600,
            "final_layer.bias": 3,
        })
        self.assertEqual(report["normalization_key_mapping"], {
            "norm.weight": "fc_norm.weight",
            "norm.bias": "fc_norm.bias",
        })

    def test_author_to_braindecode_conversion_is_tensor_exact(self):
        author_path = os.environ.get("LABRAM_AUTHOR_CHECKPOINT")
        if not author_path:
            self.skipTest("set LABRAM_AUTHOR_CHECKPOINT for independent provenance recheck")
        author_path = Path(author_path)
        self.assertEqual(author_path.stat().st_size, LABRAM_AUTHOR_CHECKPOINT_BYTES)
        self.assertEqual(sha256_file(author_path), LABRAM_AUTHOR_CHECKPOINT_SHA256)
        author = torch.load(author_path, map_location="cpu", weights_only=False)["model"]
        converted = self.raw_state
        reverse = {
            "patch_embed.temporal_conv.conv1.weight": "student.patch_embed.conv1.weight",
            "patch_embed.temporal_conv.conv1.bias": "student.patch_embed.conv1.bias",
            "patch_embed.temporal_conv.norm1.weight": "student.patch_embed.norm1.weight",
            "patch_embed.temporal_conv.norm1.bias": "student.patch_embed.norm1.bias",
            "patch_embed.temporal_conv.conv2.weight": "student.patch_embed.conv2.weight",
            "patch_embed.temporal_conv.conv2.bias": "student.patch_embed.conv2.bias",
            "patch_embed.temporal_conv.norm2.weight": "student.patch_embed.norm2.weight",
            "patch_embed.temporal_conv.norm2.bias": "student.patch_embed.norm2.bias",
            "patch_embed.temporal_conv.conv3.weight": "student.patch_embed.conv3.weight",
            "patch_embed.temporal_conv.conv3.bias": "student.patch_embed.conv3.bias",
            "patch_embed.temporal_conv.norm3.weight": "student.patch_embed.norm3.weight",
            "patch_embed.temporal_conv.norm3.bias": "student.patch_embed.norm3.bias",
            "position_embedding": "student.pos_embed",
            "temporal_embedding": "student.time_embed",
        }
        compared = 0
        used = set()
        for target_key, target in converted.items():
            source_key = reverse.get(target_key, f"student.{target_key}")
            source_key = source_key.replace(".mlp.0.", ".mlp.fc1.")
            source_key = source_key.replace(".mlp.2.", ".mlp.fc2.")
            self.assertIn(source_key, author)
            source = author[source_key]
            self.assertEqual((source.shape, source.dtype), (target.shape, target.dtype))
            torch.testing.assert_close(source, target, rtol=0, atol=0)
            compared += target.numel()
            used.add(source_key)
        self.assertEqual(compared, 5_819_936)
        self.assertEqual(set(author) - used, {
            "logit_scale",
            "student.mask_token",
            "student.lm_head.weight",
            "student.lm_head.bias",
            "lm_head.weight",
            "lm_head.bias",
            "projection_head.0.weight",
            "projection_head.0.bias",
        })

    def test_channel_names_map_to_official_embedding_ids(self):
        self.assertEqual(
            [name.casefold() for name in self.model.channel_names],
            [name.casefold() for name in MI32_CHANNEL_ORDER],
        )
        self.assertEqual(
            [entry["input_name"] for entry in self.model.channel_mapping],
            MI32_CHANNEL_ORDER,
        )
        self.assertEqual(self.model.channel_embedding_ids.tolist(), EXPECTED_EMBEDDING_IDS)
        self.assertNotEqual(EXPECTED_EMBEDDING_IDS, list(range(1, 33)))
        self.assertEqual(self.model.load_report["matched_channels"], 32)
        self.assertEqual(self.model.load_report["unknown_channels"], [])

    def test_bandlimited_resampling_is_deterministic_and_anti_aliased(self):
        kernel, width, old_reduced, new_reduced = _sinc_resample_kernel(250, 200)
        time_axis = torch.arange(750, dtype=torch.float32) / 250
        low = torch.sin(2 * math.pi * 20 * time_axis)[None, None]
        above_new_nyquist = torch.sin(2 * math.pi * 110 * time_axis)[None, None]
        signal = torch.cat((low, above_new_nyquist), dim=1)
        first = _apply_sinc_resample(signal, kernel, width, old_reduced, new_reduced)
        second = _apply_sinc_resample(signal, kernel, width, old_reduced, new_reduced)
        self.assertEqual(tuple(first.shape), (1, 2, 600))
        self.assertTrue(torch.equal(first, second))
        edge = 20
        low_rms = first[0, 0, edge:-edge].square().mean().sqrt().item()
        high_rms = first[0, 1, edge:-edge].square().mean().sqrt().item()
        self.assertGreater(low_rms, 0.65)
        self.assertLess(high_rms, 0.20)
        self.assertLess(high_rms, low_rms * 0.25)

    def test_labram_loader_preserves_physical_microvolt_scale_without_zscore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pd.DataFrame([{
                "subject_id": 7,
                "subject_name": "synthetic_subject",
            }]).to_csv(root / "subjects.csv", index=False)
            signal_volts = np.full((1, 32, 750), 2.5e-6, dtype=np.float32)
            signal_volts[:, 1, :] = np.linspace(-1e-6, 1e-6, 750, dtype=np.float32)
            np.savez(
                root / "data" / "synthetic_subject.npz",
                X=signal_volts,
                y=np.asarray([1], dtype=np.int64),
                channel_mask=np.ones((1, 32), dtype=bool),
            )
            dataset = LaBraMMI3Trials(root, {7}, preload=True)
            x, label, subject = dataset[0]
            self.assertEqual((label, subject), (1, 7))
            torch.testing.assert_close(x, torch.from_numpy(signal_volts[0] * 1e6), rtol=0, atol=0)
            self.assertAlmostEqual(x[0].mean().item(), 2.5, places=6)
            self.assertNotAlmostEqual(x[0].std().item(), 1.0, places=3)

    def test_labram_loader_does_not_zero_masked_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pd.DataFrame([{
                "subject_id": 7,
                "subject_name": "synthetic_subject",
            }]).to_csv(root / "subjects.csv", index=False)
            signal_volts = np.full((1, 32, 750), 2.5e-6, dtype=np.float32)
            channel_mask = np.ones((1, 32), dtype=bool)
            channel_mask[0, 1] = False
            np.savez(
                root / "data" / "synthetic_subject.npz",
                X=signal_volts,
                y=np.asarray([1], dtype=np.int64),
                channel_mask=channel_mask,
            )
            dataset = LaBraMMI3Trials(root, {7}, preload=True)
            x, _, _ = dataset[0]
            self.assertGreater(x[1].abs().max().item(), 0.0)
            torch.testing.assert_close(x[1], torch.full((750,), 2.5, dtype=torch.float32), rtol=0, atol=0)

    def test_temporal_embedding_adapts_only_axis_one(self):
        report = self.model.load_report["temporal_embedding"]
        self.assertEqual(report, {
            "checkpoint_shape": [1, 16, 200],
            "model_shape": [1, 3, 200],
            "interpolation_axis": 1,
            "method": "linear",
            "align_corners": False,
            "other_axes_unchanged": True,
        })
        expected = torch.nn.functional.interpolate(
            self.raw_state["temporal_embedding"].transpose(1, 2),
            size=3,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        torch.testing.assert_close(self.model.backbone.temporal_embedding, expected, rtol=0, atol=0)
        self.assertEqual(tuple(self.model.backbone.cls_token.shape), (1, 1, 200))
        expanded = self.model.backbone._adj_temporal_embedding(
            num_ch=32, batch_size=2, dim_embed=200
        )
        self.assertEqual(tuple(expanded.shape), (2, 96, 200))
        torch.testing.assert_close(
            expanded.reshape(2, 32, 3, 200)[0, 0],
            self.model.backbone.temporal_embedding[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            self.model.backbone.position_embedding,
            self.raw_state["position_embedding"],
            rtol=0,
            atol=0,
        )

    def test_active_patch_token_pooling_and_classifier_path(self):
        audit = validate_labram_structure(
            self.model,
            batch_size=2,
            require_verified_checkpoint=self.verify_identity,
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["input_shape"], [2, 32, 750])
        self.assertEqual(audit["resampled_shape"], [2, 32, 600])
        self.assertEqual(audit["segments"], [2, 32, 3, 200])
        self.assertEqual(audit["channel_time_tokens"], [2, 96, 200])
        self.assertEqual(audit["transformer_tokens_including_cls"], 97)
        self.assertEqual(audit["mean_pool_input"], [2, 200])
        self.assertEqual(audit["output_shape"], [2, 3])
        self.assertIsInstance(self.model.backbone.norm, nn.Identity)
        self.assertIsInstance(self.model.backbone.fc_norm, nn.LayerNorm)
        self.assertEqual(
            (self.model.backbone.final_layer.in_features, self.model.backbone.final_layer.out_features),
            (200, 3),
        )

    def test_full_finetuning_and_optimizer_are_report_only_unchanged(self):
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
        groups = self.model.optimizer_groups(3e-5)
        self.assertAlmostEqual(groups[0]["lr"], 3e-5)
        self.assertAlmostEqual(groups[1]["lr"], 1.5e-4)
        self.assertEqual(
            sum(parameter.numel() for group in groups for parameter in group["params"]),
            5_817_939,
        )

    def test_source_snapshot_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_labram_sources(Path(directory), self.model)
            self.assertEqual(report["braindecode_version"], LABRAM_BRAINDECODE_VERSION)
            self.assertEqual(report["braindecode_source_sha256"], mi3_foundation.LABRAM_BRAINDECODE_SOURCE_SHA256)
            self.assertEqual(report["adapter_sha256"], sha256_file(report["adapter_source"]))
            self.assertEqual(report["braindecode_source_sha256"], sha256_file(report["braindecode_source"]))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LaBraMStructureTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "checkpoint": str(LaBraMStructureTest.checkpoint),
        "checkpoint_identity_verified": LaBraMStructureTest.verify_identity,
    }, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
