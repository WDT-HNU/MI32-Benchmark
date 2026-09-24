import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def _install_mamba_architecture_stub():
    """Provide parameter-exact Mamba2 modules when CUDA wheels are unavailable.

    The stub is only a local architecture/checkpoint audit aid: it preserves
    every official parameter name and shape but does not claim to reproduce the
    Mamba2 numerical kernel. Formal CUDA runs use the real mamba_ssm package.
    """
    try:
        import mamba_ssm  # noqa: F401
        return False
    except ImportError:
        pass

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, **{
                key: value for key, value in kwargs.items() if key in {"device", "dtype"}
            }))
            self.register_parameter("bias", None)
            self.eps = eps

        def forward(self, x):
            return x * x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype) * self.weight

    def layer_norm_fn(hidden_states, weight, bias=None, eps=1e-5, residual=None,
                      prenorm=False, residual_in_fp32=False, is_rms_norm=False, **kwargs):
        del kwargs, is_rms_norm
        combined = hidden_states if residual is None else hidden_states + residual
        residual_out = combined.float() if residual_in_fp32 else combined
        normalized = combined * combined.float().square().mean(-1, keepdim=True).add(eps).rsqrt().to(combined.dtype)
        normalized = normalized * weight
        if bias is not None:
            normalized = normalized + bias
        return (normalized, residual_out) if prenorm else normalized

    class Mamba2(nn.Module):
        def __init__(self, d_model, layer_idx=None, headdim=50, d_state=64,
                     expand=2, ngroups=1, d_conv=4, **kwargs):
            super().__init__()
            del kwargs
            self.d_model = d_model
            self.layer_idx = layer_idx
            self.headdim = headdim
            self.d_state = d_state
            self.d_inner = expand * d_model
            self.nheads = self.d_inner // headdim
            conv_dim = self.d_inner + 2 * ngroups * d_state
            projection_dim = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
            self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
            self.A_log = nn.Parameter(torch.zeros(self.nheads))
            self.D = nn.Parameter(torch.ones(self.nheads))
            self.in_proj = nn.Linear(d_model, projection_dim, bias=False)
            self.conv1d = nn.Conv1d(
                conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim, bias=True
            )
            self.norm = RMSNorm(self.d_inner)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x, **kwargs):
            del kwargs
            projected = self.in_proj(x)[..., :self.d_inner]
            return self.out_proj(self.norm(projected))

    class Block(nn.Module):
        def __init__(self, d_model, mixer_cls, mlp_cls, norm_cls, **kwargs):
            super().__init__()
            del kwargs, mlp_cls
            self.norm = norm_cls(d_model)
            self.mixer = mixer_cls(d_model)
            self.mlp = None

        def forward(self, hidden_states, residual=None, **kwargs):
            del kwargs
            residual = hidden_states if residual is None else hidden_states + residual
            return self.mixer(self.norm(residual)), residual

        def allocate_inference_cache(self, *args, **kwargs):
            return {}

    class _UnusedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    package_names = [
        "mamba_ssm", "mamba_ssm.models", "mamba_ssm.models.config_mamba",
        "mamba_ssm.modules", "mamba_ssm.modules.mamba_simple", "mamba_ssm.modules.mamba2",
        "mamba_ssm.modules.mha", "mamba_ssm.modules.mlp", "mamba_ssm.modules.block",
        "mamba_ssm.utils", "mamba_ssm.utils.generation", "mamba_ssm.utils.hf",
        "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layer_norm",
    ]
    modules = {name: types.ModuleType(name) for name in package_names}
    modules["mamba_ssm"].__version__ = "architecture-audit-stub"
    modules["mamba_ssm.models.config_mamba"].MambaConfig = object
    modules["mamba_ssm.modules.mamba_simple"].Mamba = _UnusedModule
    modules["mamba_ssm.modules.mamba2"].Mamba2 = Mamba2
    modules["mamba_ssm.modules.mha"].MHA = _UnusedModule
    modules["mamba_ssm.modules.mlp"].GatedMLP = _UnusedModule
    modules["mamba_ssm.modules.block"].Block = Block
    modules["mamba_ssm.utils.generation"].GenerationMixin = object
    modules["mamba_ssm.utils.hf"].load_config_hf = lambda *args, **kwargs: {}
    modules["mamba_ssm.utils.hf"].load_state_dict_hf = lambda *args, **kwargs: {}
    layer_norm = modules["mamba_ssm.ops.triton.layer_norm"]
    layer_norm.RMSNorm = RMSNorm
    layer_norm.layer_norm_fn = layer_norm_fn
    layer_norm.rms_norm_fn = layer_norm_fn
    sys.modules.update(modules)
    return True


try:
    import mi3_deep_models  # noqa: F401
except ModuleNotFoundError:
    import remote_mi3_deep_models as mi3_deep_models
    sys.modules["mi3_deep_models"] = mi3_deep_models

from model_adapters.eegmamba import EEGMambaAdapter
from mi3_foundation import (
    EEGMAMBA_CHECKPOINT_BYTES,
    EEGMAMBA_CHECKPOINT_SHA256,
    EEGMAMBA_MI32_CHANNEL_ORDER,
    EEGMAMBA_OFFICIAL_COMMIT,
    EEGMAMBA_SOURCE_SHA256,
    EEGMambaClassifier,
    sha256_file,
    snapshot_eegmamba_sources,
    validate_eegmamba_structure,
)


def _looks_like_eegmamba_repo(path):
    path = Path(path)
    return path.is_dir() and all((path / Path(relative)).is_file() for relative in EEGMAMBA_SOURCE_SHA256)


def _default_eegmamba_repo():
    candidates = []
    for root in (Path(__file__).resolve().parent, Path.cwd()):
        candidates.extend((
            root / "EEGMamba",
            root / "eegmamba_repo",
            root / "work" / "eegmamba_repo",
        ))
        candidates.extend(
            sorted(root.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/work/eegmamba_repo"))
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_eegmamba_repo(resolved):
            return resolved
    raise FileNotFoundError(
        "Set EEGMAMBA_TEST_REPO to a frozen EEGMamba checkout with the audited source files."
    )


class EEGMambaStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(os.environ.get("EEGMAMBA_TEST_REPO") or _default_eegmamba_repo())
        cls.checkpoint = Path(os.environ.get(
            "EEGMAMBA_TEST_CHECKPOINT",
            str(cls.repo / "pretrained_weights" / "pretrained_EEGMamba.pth"),
        ))
        cls.using_architecture_stub = _install_mamba_architecture_stub()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter = EEGMambaAdapter(EEGMAMBA_MI32_CHANNEL_ORDER)
        fake = torch.randn(2, 32, 750)
        adapter.fit([(0, fake.numpy(), torch.zeros(2, dtype=torch.long).numpy())], "0" * 64)
        cls.model = EEGMambaClassifier(
            EEGMAMBA_MI32_CHANNEL_ORDER, cls.repo, cls.checkpoint, adapter=adapter
        ).to(cls.device)

    def test_01_official_source_identity(self):
        report = self.model.load_report["source"]
        self.assertEqual(report["commit"], EEGMAMBA_OFFICIAL_COMMIT)
        self.assertEqual(report["source_sha256"], EEGMAMBA_SOURCE_SHA256)

    def test_02_checkpoint_identity_and_bit_exact_load(self):
        report = self.model.load_report
        self.assertEqual(self.checkpoint.stat().st_size, EEGMAMBA_CHECKPOINT_BYTES)
        self.assertEqual(sha256_file(self.checkpoint), EEGMAMBA_CHECKPOINT_SHA256)
        self.assertEqual(report["checkpoint_tensor_count"], 117)
        self.assertEqual(report["checkpoint_parameter_count"], 3_317_443)
        self.assertEqual(
            report["exact_loaded_parameter_count_before_downstream_proj_replacement"],
            3_317_443,
        )
        self.assertEqual(report["checkpoint_coverage_before_downstream_proj_replacement"], 1.0)
        self.assertEqual(report["checkpoint_missing"], [])
        self.assertEqual(report["checkpoint_unexpected"], [])
        self.assertEqual(report["shape_mismatch"], {})
        self.assertEqual(report["numerical_load_max_abs_error"], 0.0)
        self.assertTrue(all(error == 0.0 for error in report["numerical_load_sample"].values()))

    def test_03_no_unsupported_32_to_60_channel_adapter(self):
        self.assertFalse(hasattr(self.model, "channel_adapter"))
        self.assertIsNone(self.model.load_report["channel_adapter"])
        self.assertEqual(self.model.channel_order, EEGMAMBA_MI32_CHANNEL_ORDER)
        self.assertEqual(self.model.channel_indices.tolist(), list(range(32)))

    def test_04_antialiased_three_patch_input_path(self):
        generator = torch.Generator(device=self.device).manual_seed(20260817)
        x = torch.randn(2, 32, 750, generator=generator, device=self.device)
        first = self.model.prepare_input(x)
        second = self.model.prepare_input(x)
        self.assertEqual(tuple(first.shape), (2, 32, 3, 200))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(self.model.load_report["resampling"]["padding_samples"], 0)
        self.assertTrue(self.model.load_report["resampling"]["anti_aliasing"])

    def test_05_official_patch_embedding(self):
        patch = self.model.backbone.patch_embedding
        conv = patch.proj_in[0]
        self.assertEqual((conv.kernel_size, conv.stride, conv.padding, conv.out_channels),
                         ((1, 49), (1, 25), (0, 24), 25))
        self.assertIsNone(conv.bias)
        self.assertEqual((patch.proj_in[1].num_groups, patch.proj_in[1].num_channels), (5, 25))
        self.assertEqual((patch.spectral_proj[0].in_features,
                          patch.spectral_proj[0].out_features), (101, 200))
        positional = patch.positional_encoding[0]
        self.assertEqual((positional.kernel_size, positional.padding, positional.groups),
                         ((7, 7), (3, 3), 200))

    def test_06_official_mamba2_configuration(self):
        encoder = self.model.backbone.encoder
        self.assertEqual(len(encoder.layers), 12)
        for layer in encoder.layers:
            self.assertEqual(type(layer.mixer).__name__, "Mamba2")
            self.assertEqual(layer.mixer.headdim, 50)
            self.assertEqual(layer.mixer.d_state, 64)
            self.assertEqual(layer.mixer.d_model, 200)
            self.assertIsNone(layer.mlp)
            self.assertEqual(type(layer.norm).__name__, "RMSNorm")
        self.assertTrue(encoder.fused_add_norm)
        self.assertTrue(encoder.residual_in_fp32)
        self.assertEqual(type(encoder.norm_f).__name__, "RMSNorm")

    def test_07_official_downstream_identity_and_classifier(self):
        self.assertIsInstance(self.model.backbone.proj_out, nn.Identity)
        linear = [module for module in self.model.classifier if isinstance(module, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear], [
            (19_200, 600), (600, 200), (200, 3),
        ])
        self.assertEqual(self.model.load_report["removed_pretraining_proj_out_parameters"], 40_200)
        self.assertEqual(self.model.load_report["pretrained_backbone_parameters_used_downstream"], 3_277_243)
        self.assertEqual(self.model.load_report["new_classifier_parameters"], 11_641_403)
        self.assertEqual(self.model.load_report["model_parameter_count"], 14_918_646)

    def test_08_full_shape_and_bf16_compatibility(self):
        audit = validate_eegmamba_structure(self.model, batch_size=2, run_full_forward=True)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["patch_shape"], [2, 32, 3, 200])
        self.assertEqual(audit["output_shape"], [2, 3])
        x = torch.randn(2, 32, 750, device=self.device)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.model(x)
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_09_source_snapshot_and_optimizer_protocol_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            report = snapshot_eegmamba_sources(Path(directory), self.model)
            self.assertEqual(report["official_commit"], EEGMAMBA_OFFICIAL_COMMIT)
            for relative, expected in EEGMAMBA_SOURCE_SHA256.items():
                self.assertEqual(report["official_sources"][relative]["sha256"], expected)
        groups = self.model.optimizer_groups(1e-4)
        self.assertEqual([group["lr"] for group in groups], [1e-4, 5e-4])
        frozen = [name for name, parameter in self.model.named_parameters() if not parameter.requires_grad]
        self.assertEqual(frozen, ["backbone.patch_embedding.mask_encoding"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EEGMambaStructureTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "official_commit": EEGMAMBA_OFFICIAL_COMMIT,
        "checkpoint": os.environ.get("EEGMAMBA_TEST_CHECKPOINT", "official default path"),
        "training_executed": False,
        "dataset_samples_read": 0,
        "mamba_execution": (
            "parameter-exact architecture stub; real CUDA kernel not claimed"
            if getattr(EEGMambaStructureTest, "using_architecture_stub", False)
            else "real mamba_ssm CUDA package"
        ),
    }, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
