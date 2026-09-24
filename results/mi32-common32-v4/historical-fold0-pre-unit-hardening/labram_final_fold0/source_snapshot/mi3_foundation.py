import argparse
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import sys
import time
from types import MethodType
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from mi3_deep_models import (
    MI3RawTrials,
    SubjectBatchSampler,
    evaluate,
    seed_all,
)
from torch.utils.data import DataLoader

from model_adapters.codebrain import CodeBrainAdapter
from model_adapters.eegmamba import EEGMambaAdapter
from model_adapters.labram import LaBraMAdapter


LABRAM_CHECKPOINT_SHA256 = "86a40de11088b85291eb47b788b049d784e619a6b3f8e84accbf640b2c59eec3"
LABRAM_CHECKPOINT_BYTES = 23_351_447
LABRAM_CHECKPOINT_SOURCE = (
    "https://huggingface.co/braindecode/labram-pretrained/resolve/"
    "0563b6c626e7b40d9a36653b763715db94d945d7/pytorch_model.bin"
)
LABRAM_CHECKPOINT_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"
LABRAM_CONVERSION_SOURCE = "https://huggingface.co/braindecode/Labram-Braindecode"
LABRAM_AUTHOR_SOURCE_REVISION = "c431221e6cfd23dbfa9950e0180682fb322b0548"
LABRAM_AUTHOR_CHECKPOINT_SOURCE = (
    "https://github.com/935963004/LaBraM/blob/"
    f"{LABRAM_AUTHOR_SOURCE_REVISION}/checkpoints/labram-base.pth"
)
LABRAM_AUTHOR_CHECKPOINT_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
LABRAM_AUTHOR_CHECKPOINT_BYTES = 96_612_769
LABRAM_BRAINDECODE_VERSION = "1.7.0"
LABRAM_BRAINDECODE_SOURCE_SHA256 = "4102baa7e509c8674c663fa0d30937e389f452e414b33c0588b5f59721eb4e45"

EEGMAMBA_OFFICIAL_COMMIT = "dbc83fa072744201e8897aeb9f65007b952ad323"
EEGMAMBA_REPOSITORY = "https://github.com/wjq-learning/EEGMamba"
EEGMAMBA_CHECKPOINT_REVISION = "0b060d87acd6f23bf1d0b852bf1726064f335f97"
EEGMAMBA_CHECKPOINT_SOURCE = (
    "https://huggingface.co/weighting666/EEGMamba/resolve/"
    f"{EEGMAMBA_CHECKPOINT_REVISION}/pretrained_EEGMamba.pth"
)
EEGMAMBA_CHECKPOINT_SHA256 = "b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe"
EEGMAMBA_CHECKPOINT_BYTES = 13_313_895
EEGMAMBA_SOURCE_SHA256 = {
    "models/eegmamba.py": "0c91c8bfcca1dcbae6565be0bc801014ae408b27ab38ddfd7fac7425f04bf509",
    "models/model_for_bciciv2a.py": "4bd5c95a6059b30f6e5efd8fee67ce9764c3ad50502b1bc5623c513b1338a97b",
    "modules/config_mamba.py": "14deb712edf00a2218af1f6aa72f3e187c4f83a36ef47a543850e91c18318bc8",
    "modules/mixer_seq_simple.py": "4871b070304f4235436d6bc12ef8f8d91fdc3bb88568dbf1ba8814d0e1825e26",
}
EEGMAMBA_MI32_CHANNEL_ORDER = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FCz", "FC4", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP3", "CPz", "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

CODEBRAIN_OFFICIAL_COMMIT = "22d350caf68246d2fda4f630ef837420db3fb130"
CODEBRAIN_REPOSITORY = "https://github.com/YjMajy/CodeBrain"
CODEBRAIN_CHECKPOINT_REVISION = "bef08d2fdb1759685371cc635aad21ce59163689"
CODEBRAIN_CHECKPOINT_SOURCE = (
    "https://huggingface.co/YjMajy/CodeBrain/resolve/"
    f"{CODEBRAIN_CHECKPOINT_REVISION}/CodeBrain.pth"
)
CODEBRAIN_CHECKPOINT_SHA256 = "d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352"
CODEBRAIN_CHECKPOINT_BYTES = 60_901_006
CODEBRAIN_SOURCE_FILES = [
    "Models/SSSM.py",
    "Models/model_for_tuev.py",
    "Models/model_for_tuab.py",
    "Downstream/finetune_main.py",
    "Datasets/tuev_dataset.py",
    "Datasets/tuab_dataset.py",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_git_blob(path):
    """SHA256 of a source file normalized to the git-blob byte stream (LF).

    Windows checkouts under core.autocrlf=true store LF blobs as CRLF on
    disk; normalizing CRLF->LF makes the digest comparable to the official
    frozen blob regardless of checkout convention. Only for text sources;
    binary artifacts (checkpoints) must use sha256_file().
    """
    with Path(path).open("rb") as handle:
        data = handle.read()
    if b"\r\n" in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _sinc_resample_kernel(orig_freq=250, new_freq=200, lowpass_filter_width=6, rolloff=0.99):
    """Create a deterministic Hann-windowed sinc polyphase low-pass kernel.

    This is the standard bandlimited interpolation construction used by
    torchaudio.functional.resample, specialized here so the sealed LaBraM
    input path does not silently change with a torchaudio release.
    """
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("sampling frequencies must be positive")
    if lowpass_filter_width <= 0 or not 0.0 < rolloff <= 1.0:
        raise ValueError("invalid low-pass resampling configuration")
    gcd = math.gcd(int(orig_freq), int(new_freq))
    old = int(orig_freq) // gcd
    new = int(new_freq) // gcd
    cutoff = min(old, new) * rolloff
    width = math.ceil(lowpass_filter_width * old / cutoff)
    index = torch.arange(-width, width + old, dtype=torch.float64)[None, None] / old
    phase = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + index
    phase = (phase * cutoff).clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(phase * math.pi / lowpass_filter_width / 2).square()
    angle = phase * math.pi
    sinc = torch.where(angle == 0, torch.ones_like(angle), angle.sin() / angle)
    kernel = (sinc * window * (cutoff / old)).to(torch.float32)
    return kernel, width, old, new


def _apply_sinc_resample(x, kernel, width, old_reduced=5, new_reduced=4):
    """Apply the fixed polyphase filter independently to every trial/channel."""
    if not x.is_floating_point():
        raise TypeError("LaBraM resampling expects floating-point EEG")
    original_shape = x.shape
    length = original_shape[-1]
    flat = x.reshape(-1, length)
    padded = F.pad(flat, (width, width + old_reduced))
    filtered = F.conv1d(padded[:, None], kernel, stride=old_reduced)
    filtered = filtered.transpose(1, 2).reshape(flat.shape[0], -1)
    target_length = math.ceil(new_reduced * length / old_reduced)
    return filtered[..., :target_length].reshape(original_shape[:-1] + (target_length,))


def _labram_exact_patch_temporal_embedding(backbone, num_ch, batch_size, dim_embed=None):
    """Expand exactly one temporal embedding per active patch and channel.

    Braindecode 1.7.0 allocates ``n_patches + 1`` temporal rows but its neural
    tokenizer adds only rows ``0:n_patches`` to the non-CLS tokens. LaBraM's
    CLS token is a separate parameter and never receives a temporal embedding.
    The MI-32 adapter therefore removes the unused fourth row and deliberately
    expands all three remaining rows over the 32 channels.
    """
    del dim_embed  # The embedding width is already encoded by the parameter.
    expected_patches = backbone.patch_embed[0].n_patchs
    actual_patches = backbone.temporal_embedding.shape[1]
    if actual_patches != expected_patches:
        raise RuntimeError(
            "LaBraM must have exactly one temporal embedding per active patch: "
            f"parameter={actual_patches}, patches={expected_patches}"
        )
    return (
        backbone.temporal_embedding.unsqueeze(1)
        .expand(batch_size, num_ch, expected_patches, -1)
        .flatten(1, 2)
    )


def snapshot_labram_sources(out, model):
    """Seal the executed adapter and dependency source beside each LaBraM run."""
    out = Path(out)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    adapter_source = Path(__file__).resolve()
    dependency_source = Path(inspect.getsourcefile(type(model.backbone))).resolve()
    adapter_copy = snapshot / "mi3_foundation.py"
    dependency_copy = snapshot / "braindecode_labram.py"
    shutil.copyfile(adapter_source, adapter_copy)
    shutil.copyfile(dependency_source, dependency_copy)
    report = {
        "adapter_source": str(adapter_copy),
        "adapter_sha256": sha256_file(adapter_copy),
        "braindecode_source": str(dependency_copy),
        "braindecode_source_sha256": sha256_file(dependency_copy),
        "braindecode_version": package_version("braindecode"),
        "required_braindecode_version": LABRAM_BRAINDECODE_VERSION,
    }
    if report["braindecode_version"] != LABRAM_BRAINDECODE_VERSION:
        raise RuntimeError(
            f"LaBraM runtime must be braindecode=={LABRAM_BRAINDECODE_VERSION}, "
            f"got {report['braindecode_version']}"
        )
    if report["braindecode_source_sha256"] != LABRAM_BRAINDECODE_SOURCE_SHA256:
        raise RuntimeError("installed Braindecode LaBraM source differs from the sealed 1.7.0 source")
    return report


def spherical_channel_adapter(source_order, target_order, alpha=1e-5):
    """Map MI-32 to a pretrained model's canonical montage without learned leakage."""
    import mne
    from mne.channels.interpolation import _make_interpolation_matrix

    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    position_lookup = {name.casefold(): xyz for name, xyz in positions.items()}
    source_lookup = {name.casefold(): index for index, name in enumerate(source_order)}
    missing_coordinates = [
        name for name in list(source_order) + list(target_order)
        if name.casefold() not in position_lookup
    ]
    if missing_coordinates:
        raise RuntimeError(f"missing standard_1005 coordinates: {missing_coordinates}")
    source_xyz = np.asarray([position_lookup[name.casefold()] for name in source_order])
    target_xyz = np.asarray([position_lookup[name.casefold()] for name in target_order])
    matrix = _make_interpolation_matrix(source_xyz, target_xyz, alpha=alpha).astype(np.float32)
    # Channels already present in MI-32 must remain bit-identical, not approximated.
    for row, name in enumerate(target_order):
        if name.casefold() in source_lookup:
            matrix[row] = 0.0
            matrix[row, source_lookup[name.casefold()]] = 1.0
    return torch.from_numpy(matrix)


class LaBraMMI3Trials(MI3RawTrials):
    """MI-32 trials in the official LaBraM amplitude unit (microvolts).

    The sealed MI-32 corpus stores signals in volts. Unlike the shared
    from-scratch benchmark loader, LaBraM must not receive an extra per-trial
    z-score because the author preprocessing specifies physical microvolts.
    Channel masks are provenance only in the sealed MI-32 release and must not
    be used to zero any loaded samples.
    """

    def _load_subject(self, sid):
        x, y, mask = self._read_subject_raw(sid, signals=True)
        chosen = self.selected[sid]
        x, y, mask = x[chosen], y[chosen], mask[chosen]
        x = x.astype(np.float32, copy=False) * np.float32(1_000_000.0)
        return x, y, mask


class LaBraMClassifier(nn.Module):
    """Official LaBraM base checkpoint adapted to fixed-channel MI-32 trials."""

    def __init__(self, channel_order, weights, verify_checkpoint_identity=True, adapter=None):
        super().__init__()
        if adapter is None:
            raise RuntimeError("LaBraMClassifier requires the project LaBraMAdapter (fitted)")
        self.adapter = adapter
        from braindecode.models import Labram
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        installed_version = package_version("braindecode")
        if installed_version != LABRAM_BRAINDECODE_VERSION:
            raise RuntimeError(
                f"LaBraM requires sealed braindecode=={LABRAM_BRAINDECODE_VERSION}; "
                f"got {installed_version}"
            )
        dependency_source = Path(inspect.getsourcefile(Labram)).resolve()
        dependency_hash = sha256_file(dependency_source)
        if dependency_hash != LABRAM_BRAINDECODE_SOURCE_SHA256:
            raise RuntimeError("Braindecode Labram source hash does not match the sealed implementation")

        canonical_names = {name.casefold(): name for name in LABRAM_CHANNEL_ORDER}
        canonical_indices = {name.casefold(): i for i, name in enumerate(LABRAM_CHANNEL_ORDER)}
        unknown = [name for name in channel_order if name.casefold() not in canonical_names]
        duplicate_names = sorted({name.casefold() for name in channel_order if
                                  sum(other.casefold() == name.casefold() for other in channel_order) > 1})
        if len(channel_order) != 32 or unknown or duplicate_names:
            raise RuntimeError(
                "LaBraM MI-32 requires exactly 32 unique canonical channel names: "
                f"count={len(channel_order)}, unknown={unknown}, duplicates={duplicate_names}"
            )
        pairs = [
            (i, name, canonical_names[name.casefold()])
            for i, name in enumerate(channel_order)
        ]
        self.register_buffer("channel_indices", torch.tensor([i for i, _, _ in pairs], dtype=torch.long))
        official_zero_based = [canonical_indices[official.casefold()] for _, _, official in pairs]
        self.register_buffer(
            "channel_embedding_ids",
            torch.tensor([i + 1 for i in official_zero_based], dtype=torch.long),
        )
        self.channel_names = [official for _, _, official in pairs]
        self.channel_mapping = [
            {
                "input_index": source_index,
                "input_name": input_name,
                "official_name": official_name,
                "official_index_zero_based": official_index,
                "model_embedding_id_with_cls_offset": official_index + 1,
            }
            for (source_index, input_name, official_name), official_index
            in zip(pairs, official_zero_based)
        ]
        self.backbone = Labram(
            n_times=600,
            n_outputs=3,
            n_chans=32,
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
            use_mean_pooling=True,
            init_scale=0.001,
            neural_tokenizer=True,
            activation=nn.GELU,
        )
        # Braindecode allocates n_patches+1 rows even though the neural-tokenizer
        # path uses only rows 0:n_patches and has a separate cls_token. Keep no
        # dead temporal parameter in the sealed three-patch MI-32 adaptation.
        active_patches = self.backbone.patch_embed[0].n_patchs
        self.backbone.temporal_embedding = nn.Parameter(
            self.backbone.temporal_embedding[:, :active_patches, :].detach().clone()
        )
        self.backbone._adj_temporal_embedding = MethodType(
            _labram_exact_patch_temporal_embedding, self.backbone
        )
        kernel, width, old_reduced, new_reduced = _sinc_resample_kernel(250, 200)
        self.register_buffer("resample_kernel", kernel, persistent=False)
        self.resample_width = width
        self.resample_old_reduced = old_reduced
        self.resample_new_reduced = new_reduced

        weights = Path(weights)
        checkpoint_bytes = weights.stat().st_size
        checkpoint_hash = sha256_file(weights)
        if verify_checkpoint_identity and (
            checkpoint_bytes != LABRAM_CHECKPOINT_BYTES or checkpoint_hash != LABRAM_CHECKPOINT_SHA256
        ):
            raise RuntimeError(
                "LaBraM checkpoint identity mismatch: "
                f"bytes={checkpoint_bytes}, sha256={checkpoint_hash}"
            )
        raw_state = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(raw_state, dict) or not raw_state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in raw_state.items()
        ):
            raise RuntimeError("LaBraM checkpoint must be a flat tensor state_dict")
        state = dict(raw_state)
        checkpoint_parameter_count = sum(value.numel() for value in raw_state.values())

        # The released converted checkpoint stores the pretrained terminal norm
        # under norm.*. The author downstream default is mean pooling, whose
        # identical LayerNorm lives under fc_norm.*. This is a key-only mapping;
        # the floating-point tensors remain bit-identical.
        normalization_key_mapping = {}
        for suffix in ("weight", "bias"):
            source_key = f"norm.{suffix}"
            target_key = f"fc_norm.{suffix}"
            if source_key in state and target_key not in state:
                state[target_key] = state.pop(source_key)
                normalization_key_mapping[source_key] = target_key

        source_temporal_shape = list(state["temporal_embedding"].shape)
        target_t = self.backbone.temporal_embedding.shape[1]
        if state["temporal_embedding"].shape[1] != target_t:
            state["temporal_embedding"] = F.interpolate(
                state["temporal_embedding"].transpose(1, 2), size=target_t,
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        target_temporal_shape = list(state["temporal_embedding"].shape)

        target_state = self.backbone.state_dict()
        allowed_missing = {"final_layer.weight", "final_layer.bias"}
        missing_before_load = sorted(set(target_state) - set(state))
        unexpected_before_load = sorted(set(state) - set(target_state))
        shape_mismatch = {
            key: {"checkpoint": list(state[key].shape), "model": list(target_state[key].shape)}
            for key in set(state) & set(target_state)
            if state[key].shape != target_state[key].shape
        }
        if set(missing_before_load) != allowed_missing or unexpected_before_load or shape_mismatch:
            raise RuntimeError(
                "LaBraM checkpoint structural mismatch before load: "
                f"missing={missing_before_load}, unexpected={unexpected_before_load}, "
                f"shape_mismatch={shape_mismatch}"
            )
        result = self.backbone.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [key for key in result.missing_keys if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(f"LaBraM checkpoint mismatch: missing={missing}, unexpected={unexpected}")

        loaded_state = self.backbone.state_dict()
        exact_keys = sorted(key for key in state if key != "temporal_embedding")
        exact_max_abs_error = {}
        for key in exact_keys:
            error = (loaded_state[key] - state[key]).abs().max().item()
            if error != 0.0:
                raise RuntimeError(f"LaBraM numerical load mismatch for {key}: max_abs_error={error}")
            exact_max_abs_error[key] = error
        exact_parameter_count = sum(state[key].numel() for key in exact_keys)
        adapted_parameter_count = state["temporal_embedding"].numel()
        backbone_parameter_count = sum(
            parameter.numel() for name, parameter in self.backbone.named_parameters()
            if not name.startswith("final_layer.")
        )
        reinitialized = {
            name: parameter.numel() for name, parameter in self.backbone.named_parameters()
            if name.startswith("final_layer.")
        }
        self.load_report = {
            "checkpoint_path": str(weights.resolve()),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_expected_sha256": LABRAM_CHECKPOINT_SHA256,
            "checkpoint_source": LABRAM_CHECKPOINT_SOURCE,
            "checkpoint_revision": LABRAM_CHECKPOINT_REVISION,
            "checkpoint_conversion_source": LABRAM_CONVERSION_SOURCE,
            "author_checkpoint_source": LABRAM_AUTHOR_CHECKPOINT_SOURCE,
            "author_checkpoint_sha256": LABRAM_AUTHOR_CHECKPOINT_SHA256,
            "author_to_converted_equivalence": (
                "all 221 converted tensors / 5,819,936 parameters independently compared; "
                "max_abs_error=0"
            ),
            "checkpoint_identity_verified": bool(verify_checkpoint_identity),
            "checkpoint_parameter_count": checkpoint_parameter_count,
            "model_parameter_count": sum(p.numel() for p in self.backbone.parameters()),
            "backbone_parameter_count_excluding_new_head": backbone_parameter_count,
            "exact_loaded_parameter_count": exact_parameter_count,
            "temporally_adapted_parameter_count": adapted_parameter_count,
            "backbone_coverage": (exact_parameter_count + adapted_parameter_count) / backbone_parameter_count,
            "bit_exact_backbone_ratio": exact_parameter_count / backbone_parameter_count,
            "reinitialized_parameters": reinitialized,
            "checkpoint_missing": list(result.missing_keys),
            "checkpoint_unexpected": unexpected,
            "shape_mismatch_after_permitted_adaptation": shape_mismatch,
            "normalization_key_mapping": normalization_key_mapping,
            "numerical_load_all_non_temporal_max_abs_error": max(exact_max_abs_error.values()),
            "numerical_load_sample": {
                key: exact_max_abs_error[key]
                for key in [exact_keys[round(i * (len(exact_keys) - 1) / 9)] for i in range(10)]
            },
            "temporal_embedding": {
                "checkpoint_shape": source_temporal_shape,
                "model_shape": target_temporal_shape,
                "interpolation_axis": 1,
                "method": "linear",
                "align_corners": False,
                "other_axes_unchanged": (
                    source_temporal_shape[0] == target_temporal_shape[0]
                    and source_temporal_shape[2] == target_temporal_shape[2]
                ),
            },
            "matched_channels": len(pairs),
            "unknown_channels": unknown,
            "channel_mapping": self.channel_mapping,
            "braindecode_version": installed_version,
            "braindecode_labram_source_sha256": dependency_hash,
            "input_normalization": (
                "LaBraM-specific physical scaling from corpus volts to microvolts (x1e6); "
                "no per-trial or per-channel z-score"
            ),
            "resampling": {
                "input_hz": 250,
                "output_hz": 200,
                "input_samples": 750,
                "output_samples": 600,
                "method": "Hann-windowed sinc polyphase bandlimited interpolation",
                "anti_aliasing": True,
                "lowpass_filter_width": 6,
                "rolloff": 0.99,
                "trial_local": True,
                "deterministic": True,
            },
            "patching": {
                "patch_samples": 200,
                "patch_seconds": 1.0,
                "patches_per_channel": 3,
                "channel_time_tokens": 96,
                "transformer_tokens_including_cls": 97,
                "token_order": "channel-major, then temporal patch",
            },
            "pooling": "mean of all 96 non-CLS channel-time tokens, followed by fc_norm",
            "classifier": "single Linear(200, 3)",
        }

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.adapter.transform(x)
        if x.shape[-1] != 600:
            raise RuntimeError(f"LaBraM adapter must produce 600 samples, got {x.shape[-1]}")
        return self.backbone(x, ch_names=self.channel_names)

    def optimizer_groups(self, lr):
        head_ids = {id(p) for p in self.backbone.final_layer.parameters()}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": backbone, "lr": lr},
                {"params": self.backbone.final_layer.parameters(), "lr": lr * 5}]


def validate_labram_structure(model, batch_size=2, require_verified_checkpoint=True):
    """Fail closed on LaBraM identity, input, token, pooling, and head semantics."""
    report = model.load_report
    if require_verified_checkpoint and not report["checkpoint_identity_verified"]:
        raise AssertionError("formal LaBraM validation requires the sealed public checkpoint")
    if report["backbone_coverage"] != 1.0:
        raise AssertionError(f"incomplete pretrained backbone: {report['backbone_coverage']}")
    if report["checkpoint_missing"] != ["final_layer.weight", "final_layer.bias"]:
        raise AssertionError(f"unexpected missing parameters: {report['checkpoint_missing']}")
    if report["checkpoint_unexpected"] or report["shape_mismatch_after_permitted_adaptation"]:
        raise AssertionError("LaBraM state_dict has unresolved keys or shapes")
    if report["numerical_load_all_non_temporal_max_abs_error"] != 0.0:
        raise AssertionError("LaBraM checkpoint was not loaded bit-exactly")
    if len(model.channel_names) != 32 or len(set(model.channel_names)) != 32:
        raise AssertionError("LaBraM must receive 32 unique MI-32 channel names")
    expected_ids = [entry["model_embedding_id_with_cls_offset"] for entry in model.channel_mapping]
    if model.channel_embedding_ids.tolist() != expected_ids:
        raise AssertionError("LaBraM channel-name to position-embedding mapping changed")

    backbone = model.backbone
    segment = backbone.patch_embed.segment_patch
    if (segment.patch_size, segment.n_patchs, segment.learned_patcher) != (200, 3, False):
        raise AssertionError("LaBraM patch segmentation differs from the sealed MI-32 adaptation")
    if len(backbone.blocks) != 12 or backbone.embed_dim != 200:
        raise AssertionError("LaBraM Transformer depth or hidden size differs")
    if any(block.attn.num_heads != 10 for block in backbone.blocks):
        raise AssertionError("LaBraM attention head count differs")
    if tuple(backbone.temporal_embedding.shape) != (1, 3, 200):
        raise AssertionError("LaBraM must have exactly three active temporal embeddings")
    expanded_temporal = backbone._adj_temporal_embedding(
        num_ch=32, batch_size=batch_size, dim_embed=200
    )
    if tuple(expanded_temporal.shape) != (batch_size, 96, 200):
        raise AssertionError("LaBraM temporal embeddings do not cover all 96 non-CLS tokens")
    if not isinstance(backbone.norm, nn.Identity) or not isinstance(backbone.fc_norm, nn.LayerNorm):
        raise AssertionError("LaBraM must use the author downstream mean-pooling path")
    if not isinstance(backbone.final_layer, nn.Linear) or (
        backbone.final_layer.in_features, backbone.final_layer.out_features
    ) != (200, 3):
        raise AssertionError("LaBraM classifier must be a single Linear(200, 3)")

    generator = torch.Generator().manual_seed(20260817)
    x = torch.randn(batch_size, 32, 750, generator=generator)
    first = _apply_sinc_resample(
        x, model.resample_kernel, model.resample_width,
        model.resample_old_reduced, model.resample_new_reduced,
    )
    second = _apply_sinc_resample(
        x, model.resample_kernel, model.resample_width,
        model.resample_old_reduced, model.resample_new_reduced,
    )
    if first.shape != (batch_size, 32, 600) or not torch.equal(first, second):
        raise AssertionError("LaBraM 250-to-200 Hz resampling is not deterministic [B,32,600]")

    captured = {}
    hooks = [
        backbone.patch_embed.segment_patch.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("segments", list(output.shape))
        ),
        backbone.patch_embed.temporal_conv.register_forward_hook(
            lambda _m, _i, output: captured.__setitem__("channel_time_tokens", list(output.shape))
        ),
        backbone.fc_norm.register_forward_pre_hook(
            lambda _m, inputs: captured.__setitem__("mean_pool_input", list(inputs[0].shape))
        ),
    ]
    model.eval()
    with torch.no_grad():
        output = model(x)
    for hook in hooks:
        hook.remove()
    expected_shapes = {
        "segments": [batch_size, 32, 3, 200],
        "channel_time_tokens": [batch_size, 96, 200],
        "mean_pool_input": [batch_size, 200],
    }
    if captured != expected_shapes or list(output.shape) != [batch_size, 3]:
        raise AssertionError(
            f"LaBraM active tensor path changed: captured={captured}, output={list(output.shape)}"
        )
    return {
        "status": "PASS",
        "input_shape": [batch_size, 32, 750],
        "resampled_shape": list(first.shape),
        **captured,
        "transformer_tokens_including_cls": 97,
        "output_shape": list(output.shape),
        "channel_ids": model.channel_embedding_ids.tolist(),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "backbone_coverage": report["backbone_coverage"],
        "bit_exact_backbone_ratio": report["bit_exact_backbone_ratio"],
        "non_temporal_max_abs_error": report["numerical_load_all_non_temporal_max_abs_error"],
        "pooling": report["pooling"],
        "classifier": report["classifier"],
    }


def _git_executable():
    """Resolve git across environments (PATH first, then common Windows installs).

    AutoDL/Linux: git is on PATH. Windows hosts may have a broken/absent PATH
    entry while git exists under a standard install dir (e.g. F:\\Git\\cmd).
    """
    resolved = shutil.which("git")
    if resolved:
        return resolved
    for candidate in (
        r"C:\Program Files\Git\cmd\git.exe",
        r"F:\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _verify_eegmamba_repository(repo):
    """Fail closed unless the executable EEGMamba sources are the frozen release."""
    repo = Path(repo).resolve()
    hashes = {}
    for relative, expected in EEGMAMBA_SOURCE_SHA256.items():
        source = repo / Path(relative)
        if not source.is_file():
            raise RuntimeError(f"missing frozen EEGMamba source: {source}")
        actual = _sha256_git_blob(source)
        hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(
                f"EEGMamba source hash mismatch for {relative}: expected={expected}, actual={actual}"
            )
    git = _git_executable()
    if git is None:
        raise RuntimeError("git executable not found; EEGMamba runtime repository identity cannot be verified")
    try:
        commit = subprocess.check_output(
            [git, "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("EEGMamba runtime repository must retain its git identity") from error
    if commit != EEGMAMBA_OFFICIAL_COMMIT:
        raise RuntimeError(
            f"EEGMamba commit mismatch: expected={EEGMAMBA_OFFICIAL_COMMIT}, actual={commit}"
        )
    return {"repo": str(repo), "commit": commit, "source_sha256": hashes}


def snapshot_eegmamba_sources(out, model):
    """Copy every audited EEGMamba source into the run artifact."""
    out = Path(out)
    snapshot = out / "source_snapshot_eegmamba"
    snapshot.mkdir(parents=True, exist_ok=True)
    copied = {}
    for relative, expected in EEGMAMBA_SOURCE_SHA256.items():
        source = model.repo / Path(relative)
        destination = snapshot / relative.replace("/", "__")
        with source.open("rb") as handle:
            data = handle.read()
        if b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
        destination.write_bytes(data)
        actual = _sha256_git_blob(destination)
        if actual != expected:
            raise RuntimeError(f"EEGMamba source snapshot changed while copying {relative}")
        copied[relative] = {"path": str(destination), "sha256": actual}
    adapter = snapshot / "mi3_foundation.py"
    shutil.copyfile(Path(__file__).resolve(), adapter)
    return {
        "official_repository": EEGMAMBA_REPOSITORY,
        "official_commit": EEGMAMBA_OFFICIAL_COMMIT,
        "official_sources": copied,
        "adapter": {"path": str(adapter), "sha256": sha256_file(adapter)},
    }


def _verify_codebrain_repository(repo):
    """Fail closed unless the executable CodeBrain sources are the frozen release."""
    repo = Path(repo).resolve()
    git = _git_executable()
    if git is None:
        raise RuntimeError("git executable not found; CodeBrain runtime repository identity cannot be verified")
    try:
        commit = subprocess.check_output(
            [git, "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("CodeBrain runtime repository must retain its git identity") from error
    if commit != CODEBRAIN_OFFICIAL_COMMIT:
        raise RuntimeError(
            f"CodeBrain commit mismatch: expected={CODEBRAIN_OFFICIAL_COMMIT}, actual={commit}"
        )
    try:
        status = subprocess.check_output(
            [git, "-C", str(repo), "status", "--short", "--untracked-files=no"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("CodeBrain runtime repository must retain its git identity") from error
    if status:
        raise RuntimeError("CodeBrain runtime repository must be clean and unmodified")
    hashes = {}
    for relative in CODEBRAIN_SOURCE_FILES:
        source = repo / Path(relative)
        if not source.is_file():
            raise RuntimeError(f"missing frozen CodeBrain source: {source}")
        hashes[relative] = _sha256_git_blob(source)
    return {"repo": str(repo), "commit": commit, "source_sha256": hashes}


def snapshot_codebrain_sources(out, model):
    """Copy every audited CodeBrain source into the run artifact."""
    out = Path(out)
    snapshot = out / "source_snapshot_codebrain"
    snapshot.mkdir(parents=True, exist_ok=True)
    copied = {}
    for relative in CODEBRAIN_SOURCE_FILES:
        source = model.repo / Path(relative)
        destination = snapshot / relative.replace("/", "__")
        with source.open("rb") as handle:
            data = handle.read()
        if b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
        destination.write_bytes(data)
        copied[relative] = {"path": str(destination), "sha256": _sha256_git_blob(destination)}
    adapter = snapshot / "mi3_foundation.py"
    shutil.copyfile(Path(__file__).resolve(), adapter)
    return {
        "official_repository": CODEBRAIN_REPOSITORY,
        "official_commit": CODEBRAIN_OFFICIAL_COMMIT,
        "official_sources": copied,
        "adapter": {"path": str(adapter), "sha256": sha256_file(adapter)},
    }


class EEGMambaClassifier(nn.Module):
    """Official pretrained EEGMamba with the default all-patch downstream head."""

    def __init__(self, channel_order, repo, weights, verify_checkpoint_identity=True, adapter=None):
        super().__init__()
        if adapter is None:
            raise RuntimeError("EEGMambaClassifier requires the project EEGMambaAdapter (fitted)")
        self.adapter = adapter
        from einops.layers.torch import Rearrange

        self.repo = Path(repo).resolve()
        source_report = _verify_eegmamba_repository(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from models.eegmamba import EEGMamba

        imported_source = Path(inspect.getsourcefile(EEGMamba)).resolve()
        expected_source = (self.repo / "models" / "eegmamba.py").resolve()
        if imported_source != expected_source:
            raise RuntimeError(
                f"imported EEGMamba from {imported_source}, expected frozen source {expected_source}"
            )

        observed_order = [name.casefold() for name in channel_order]
        expected_order = [name.casefold() for name in EEGMAMBA_MI32_CHANNEL_ORDER]
        if observed_order != expected_order:
            raise RuntimeError(
                "EEGMamba MI-32 requires the sealed topology-oriented channel order; "
                f"received={list(channel_order)}"
            )
        self.channel_order = list(channel_order)
        self.register_buffer("channel_indices", torch.arange(32, dtype=torch.long))

        self.backbone = EEGMamba(
            in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
            seq_len=3, n_layer=12, nhead=8,
        )
        weights = Path(weights)
        checkpoint_bytes = weights.stat().st_size
        checkpoint_hash = sha256_file(weights)
        if verify_checkpoint_identity and (
            checkpoint_bytes != EEGMAMBA_CHECKPOINT_BYTES
            or checkpoint_hash != EEGMAMBA_CHECKPOINT_SHA256
        ):
            raise RuntimeError(
                "EEGMamba checkpoint identity mismatch: "
                f"bytes={checkpoint_bytes}, sha256={checkpoint_hash}"
            )
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
        ):
            raise RuntimeError("EEGMamba checkpoint must be a flat tensor state_dict")
        target_state = self.backbone.state_dict()
        missing_before_load = sorted(set(target_state) - set(state))
        unexpected_before_load = sorted(set(state) - set(target_state))
        shape_mismatch = {
            key: {"checkpoint": list(state[key].shape), "model": list(target_state[key].shape)}
            for key in set(state) & set(target_state)
            if state[key].shape != target_state[key].shape
        }
        if missing_before_load or unexpected_before_load or shape_mismatch:
            raise RuntimeError(
                "EEGMamba checkpoint structural mismatch: "
                f"missing={missing_before_load}, unexpected={unexpected_before_load}, "
                f"shape_mismatch={shape_mismatch}"
            )
        result = self.backbone.load_state_dict(state, strict=True)
        loaded = self.backbone.state_dict()
        numerical_errors = {
            key: (loaded[key] - value).abs().max().item() for key, value in state.items()
        }
        if max(numerical_errors.values()) != 0.0:
            raise RuntimeError("EEGMamba checkpoint tensors were not loaded bit-exactly")
        checkpoint_parameter_count = sum(value.numel() for value in state.values())
        exact_loaded_parameter_count = sum(loaded[key].numel() for key in state)

        # This is the author's downstream path: remove the pretraining output
        # projection and use the default all_patch_reps classifier. Only C, S,
        # and class-dependent dimensions change for MI-32 (32 channels, 3
        # one-second patches, 3 classes).
        removed_proj_out_parameters = sum(p.numel() for p in self.backbone.proj_out.parameters())
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(32 * 3 * 200, 3 * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(3 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, 3),
        )
        kernel, width, old_reduced, new_reduced = _sinc_resample_kernel(250, 200)
        self.register_buffer("resample_kernel", kernel, persistent=False)
        self.resample_width = width
        self.resample_old_reduced = old_reduced
        self.resample_new_reduced = new_reduced

        exact_keys = sorted(state)
        sample_keys = [exact_keys[round(i * (len(exact_keys) - 1) / 9)] for i in range(10)]
        self.load_report = {
            "model_identity": "EEGMamba pretrained faithful adaptation to MI-32",
            "source": source_report,
            "checkpoint_path": str(weights.resolve()),
            "checkpoint_source": EEGMAMBA_CHECKPOINT_SOURCE,
            "checkpoint_revision": EEGMAMBA_CHECKPOINT_REVISION,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_expected_sha256": EEGMAMBA_CHECKPOINT_SHA256,
            "checkpoint_identity_verified": bool(verify_checkpoint_identity),
            "checkpoint_tensor_count": len(state),
            "checkpoint_parameter_count": checkpoint_parameter_count,
            "exact_loaded_parameter_count_before_downstream_proj_replacement": exact_loaded_parameter_count,
            "checkpoint_coverage_before_downstream_proj_replacement": (
                exact_loaded_parameter_count / checkpoint_parameter_count
            ),
            "checkpoint_missing": list(result.missing_keys),
            "checkpoint_unexpected": list(result.unexpected_keys),
            "shape_mismatch": shape_mismatch,
            "numerical_load_max_abs_error": max(numerical_errors.values()),
            "numerical_load_sample": {key: numerical_errors[key] for key in sample_keys},
            "removed_pretraining_proj_out_parameters": removed_proj_out_parameters,
            "pretrained_backbone_parameters_used_downstream": sum(
                p.numel() for p in self.backbone.parameters()
            ),
            "new_classifier_parameters": sum(p.numel() for p in self.classifier.parameters()),
            "model_parameter_count": sum(p.numel() for p in self.parameters()),
            "channel_adapter": None,
            "source_channels": 32,
            "received_channel_order": self.channel_order,
            "formal_mi32_channel_order": EEGMAMBA_MI32_CHANNEL_ORDER,
            "channel_order_rule": (
                "preserve the released MI-32 topology-oriented order; no alphabetic, "
                "SZBD-60, or invented reordering"
            ),
            "input_scaling": (
                "PENDING: sealed MI-32 loader passes raw volt-scale trials through resampling; "
                "no runtime per-trial or per-channel z-score is applied here"
            ),
            "resampling": {
                "input_hz": 250,
                "output_hz": 200,
                "input_samples": 750,
                "output_samples": 600,
                "method": "Hann-windowed sinc polyphase bandlimited interpolation",
                "anti_aliasing": True,
                "patches": 3,
                "patch_samples": 200,
                "padding_samples": 0,
            },
            "proj_out": "Identity after bit-exact checkpoint load, matching official downstream wrapper",
            "classifier_mode": "all_patch_reps (official default)",
            "classifier": "19200 -> 600 -> 200 -> 3 with ELU and Dropout(0.1)",
        }

    def prepare_input(self, x):
        """Deterministically convert [B,32,750]@250 Hz to [B,32,3,200]@200 Hz.

        The whole transformation (channel pass-through, anti-aliased 250->200 Hz
        resampling, 3x200 patch split) is delegated to the project adapter.
        """
        if x.ndim != 3 or x.shape[1:] != (32, 750):
            raise ValueError(f"EEGMamba expects [B,32,750], got {list(x.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.adapter.transform(x)
        if x.shape[1:] != (32, 3, 200):
            raise RuntimeError(f"EEGMamba adapter must produce [B,32,3,200], got {list(x.shape)}")
        return x

    def forward(self, x):
        x = self.prepare_input(x)
        features = self.backbone(x)
        return self.classifier(features)

    def optimizer_groups(self, lr):
        head_ids = {id(p) for p in self.classifier.parameters()}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": backbone, "lr": lr},
                {"params": self.classifier.parameters(), "lr": lr * 5}]


def validate_eegmamba_structure(model, batch_size=2, run_full_forward=True,
                                require_verified_checkpoint=True):
    """Fail closed on the frozen EEGMamba source, checkpoint and MI-32 path."""
    report = model.load_report
    if require_verified_checkpoint and not report["checkpoint_identity_verified"]:
        raise AssertionError("formal EEGMamba validation requires the official sealed checkpoint")
    if report["source"]["commit"] != EEGMAMBA_OFFICIAL_COMMIT:
        raise AssertionError("EEGMamba official source commit changed")
    if report["source"]["source_sha256"] != EEGMAMBA_SOURCE_SHA256:
        raise AssertionError("EEGMamba official source hashes changed")
    if report["checkpoint_coverage_before_downstream_proj_replacement"] != 1.0:
        raise AssertionError("EEGMamba pretrained checkpoint was not fully loaded")
    if report["checkpoint_missing"] or report["checkpoint_unexpected"] or report["shape_mismatch"]:
        raise AssertionError("EEGMamba checkpoint has unresolved keys or shapes")
    if report["numerical_load_max_abs_error"] != 0.0:
        raise AssertionError("EEGMamba checkpoint load is not bit-exact")
    if hasattr(model, "channel_adapter") or report["channel_adapter"] is not None:
        raise AssertionError("EEGMamba must not contain the historical MI-32 to SZBD-60 adapter")
    if model.channel_order != EEGMAMBA_MI32_CHANNEL_ORDER:
        raise AssertionError("EEGMamba received a changed MI-32 channel order")

    patch = model.backbone.patch_embedding
    temporal = patch.proj_in[0]
    if not isinstance(temporal, nn.Conv2d) or (
        temporal.in_channels, temporal.out_channels, temporal.kernel_size,
        temporal.stride, temporal.padding, temporal.bias
    ) != (1, 25, (1, 49), (1, 25), (0, 24), None):
        raise AssertionError("EEGMamba temporal PatchEmbedding differs from official 49/25/24 Conv2d")
    if not isinstance(patch.proj_in[1], nn.GroupNorm) or (
        patch.proj_in[1].num_groups, patch.proj_in[1].num_channels
    ) != (5, 25) or not isinstance(patch.proj_in[2], nn.GELU):
        raise AssertionError("EEGMamba temporal GroupNorm/GELU differs")
    spectral = patch.spectral_proj[0]
    if not isinstance(spectral, nn.Linear) or (
        spectral.in_features, spectral.out_features, spectral.bias
    ) != (101, 200, None):
        raise AssertionError("EEGMamba rFFT spectral projection must be bias-free 101 -> 200")
    if not isinstance(patch.spectral_proj[1], nn.Dropout) or patch.spectral_proj[1].p != 0.1:
        raise AssertionError("EEGMamba spectral dropout differs from 0.1")
    positional = patch.positional_encoding[0]
    if not isinstance(positional, nn.Conv2d) or (
        positional.in_channels, positional.out_channels, positional.kernel_size,
        positional.stride, positional.padding, positional.groups, positional.bias
    ) != (200, 200, (7, 7), (1, 1), (3, 3), 200, None):
        raise AssertionError("EEGMamba positional encoding is not official 7x7 depthwise Conv2d")

    encoder = model.backbone.encoder
    if len(encoder.layers) != 12:
        raise AssertionError("EEGMamba must have 12 Mamba blocks")
    mixers = [layer.mixer for layer in encoder.layers]
    if any(type(mixer).__name__ != "Mamba2" for mixer in mixers):
        raise AssertionError("EEGMamba encoder contains a non-Mamba2 block")
    if any(getattr(mixer, "headdim", None) != 50 for mixer in mixers):
        raise AssertionError("EEGMamba Mamba2 headdim differs from 50")
    if any(getattr(mixer, "d_state", None) != 64 for mixer in mixers):
        raise AssertionError("EEGMamba Mamba2 d_state differs from 64")
    if any(getattr(mixer, "d_model", None) != 200 for mixer in mixers):
        raise AssertionError("EEGMamba Mamba2 d_model differs from 200")
    if any(getattr(layer, "mlp", None) is not None for layer in encoder.layers):
        raise AssertionError("EEGMamba d_intermediate must remain 0 (no block MLP)")
    if not encoder.fused_add_norm or not encoder.residual_in_fp32:
        raise AssertionError("EEGMamba fused RMSNorm/residual_in_fp32 configuration changed")
    if type(encoder.norm_f).__name__ != "RMSNorm" or any(
        type(layer.norm).__name__ != "RMSNorm" for layer in encoder.layers
    ):
        raise AssertionError("EEGMamba must use RMSNorm")
    if not isinstance(model.backbone.proj_out, nn.Identity):
        raise AssertionError("EEGMamba downstream proj_out must be Identity")
    linear_layers = [module for module in model.classifier if isinstance(module, nn.Linear)]
    if [(layer.in_features, layer.out_features) for layer in linear_layers] != [
        (19_200, 600), (600, 200), (200, 3)
    ]:
        raise AssertionError("EEGMamba classifier is not the adapted official all_patch_reps head")
    dropouts = [module.p for module in model.classifier if isinstance(module, nn.Dropout)]
    if dropouts != [0.1, 0.1]:
        raise AssertionError("EEGMamba all_patch_reps dropout differs")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260817)
    x = torch.randn(batch_size, 32, 750, generator=generator, device=device)
    first = model.prepare_input(x)
    second = model.prepare_input(x)
    if tuple(first.shape) != (batch_size, 32, 3, 200) or not torch.equal(first, second):
        raise AssertionError("EEGMamba input conversion is not deterministic [B,32,3,200]")

    captured = {}
    output_shape = None
    if run_full_forward:
        hooks = [
            patch.proj_in[0].register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("temporal_conv", list(output.shape))
            ),
            patch.spectral_proj[0].register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("spectral_projection", list(output.shape))
            ),
            patch.positional_encoding[0].register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("positional_depthwise_conv", list(output.shape))
            ),
            model.backbone.register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("backbone_output", list(output.shape))
            ),
        ]
        model.eval()
        with torch.no_grad():
            output = model(x)
        for hook in hooks:
            hook.remove()
        expected = {
            "temporal_conv": [batch_size, 25, 96, 8],
            "spectral_projection": [batch_size, 32, 3, 200],
            "positional_depthwise_conv": [batch_size, 200, 32, 3],
            "backbone_output": [batch_size, 32, 3, 200],
        }
        output_shape = list(output.shape)
        if captured != expected or output_shape != [batch_size, 3]:
            raise AssertionError(
                f"EEGMamba active path changed: captured={captured}, output={output_shape}"
            )

    return {
        "status": "PASS",
        "full_forward_executed": bool(run_full_forward),
        "input_shape": [batch_size, 32, 750],
        "resampled_shape": [batch_size, 32, 600],
        "patch_shape": list(first.shape),
        "active_shapes": captured,
        "output_shape": output_shape,
        "mamba_layers": len(encoder.layers),
        "mamba_type": type(mixers[0]).__name__,
        "headdim": mixers[0].headdim,
        "d_state": mixers[0].d_state,
        "rms_norm": True,
        "fused_add_norm": encoder.fused_add_norm,
        "residual_in_fp32": encoder.residual_in_fp32,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_coverage": report["checkpoint_coverage_before_downstream_proj_replacement"],
        "checkpoint_max_abs_error": report["numerical_load_max_abs_error"],
        "channel_adapter": None,
        "classifier_mode": report["classifier_mode"],
    }


class CodeBrainClassifier(nn.Module):
    def __init__(self, channel_order, repo, weights, verify_checkpoint_identity=True, adapter=None):
        super().__init__()
        if adapter is None:
            raise RuntimeError("CodeBrainClassifier requires the project CodeBrainAdapter (fitted)")
        self.adapter = adapter
        self.repo = Path(repo).resolve()
        source_report = _verify_codebrain_repository(self.repo)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from Models.SSSM import SSSM

        observed_order = [name.casefold() for name in channel_order]
        expected_order = [name.casefold() for name in EEGMAMBA_MI32_CHANNEL_ORDER]
        if observed_order != expected_order:
            raise RuntimeError(
                "CodeBrain MI-32 requires the sealed topology-oriented channel order; "
                f"received={list(channel_order)}"
            )
        self.channel_order = list(channel_order)
        self.register_buffer("channel_indices", torch.arange(32, dtype=torch.long))

        self.backbone = SSSM(
            in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
            num_res_layers=8,
            diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200, s4_lmax=570, s4_d_state=64,
            s4_dropout=0.1, s4_bidirectional=True, s4_layernorm=True,
            codebook_size_t=4096, codebook_size_f=4096, if_codebook=False,
        )
        weights = Path(weights)
        checkpoint_bytes = weights.stat().st_size
        checkpoint_hash = sha256_file(weights)
        if verify_checkpoint_identity and CODEBRAIN_CHECKPOINT_BYTES is not None and checkpoint_bytes != CODEBRAIN_CHECKPOINT_BYTES:
            raise RuntimeError(
                "CodeBrain checkpoint identity mismatch: "
                f"bytes={checkpoint_bytes}, expected={CODEBRAIN_CHECKPOINT_BYTES}"
            )
        if verify_checkpoint_identity and CODEBRAIN_CHECKPOINT_SHA256 is not None and checkpoint_hash != CODEBRAIN_CHECKPOINT_SHA256:
            raise RuntimeError(
                "CodeBrain checkpoint identity mismatch: "
                f"sha256={checkpoint_hash}, expected={CODEBRAIN_CHECKPOINT_SHA256}"
            )
        raw = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict) or not raw or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in raw.items()
        ):
            raise RuntimeError("CodeBrain checkpoint must be a flat tensor state_dict")
        state = {key.removeprefix("module."): value for key, value in raw.items()}
        target_state = self.backbone.state_dict()
        missing_before_load = sorted(set(target_state) - set(state))
        unexpected_before_load = sorted(set(state) - set(target_state))
        shape_mismatch = {
            key: {"checkpoint": list(state[key].shape), "model": list(target_state[key].shape)}
            for key in set(state) & set(target_state)
            if state[key].shape != target_state[key].shape
        }
        if missing_before_load or unexpected_before_load or shape_mismatch:
            raise RuntimeError(
                "CodeBrain checkpoint structural mismatch: "
                f"missing={missing_before_load}, unexpected={unexpected_before_load}, "
                f"shape_mismatch={shape_mismatch}"
            )
        result = self.backbone.load_state_dict(state, strict=True)
        loaded_state = self.backbone.state_dict()
        numerical_errors = {}
        for key, value in state.items():
            loaded = loaded_state[key]
            if loaded.is_floating_point() or loaded.is_complex():
                error = (loaded - value).abs().max().item()
                if error != 0.0:
                    raise RuntimeError("CodeBrain checkpoint tensors were not loaded bit-exactly")
            else:
                if not torch.equal(loaded, value):
                    raise RuntimeError("CodeBrain checkpoint tensors were not loaded bit-exactly")
                error = 0.0
            numerical_errors[key] = error
        self.backbone.proj_out = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 200, 3 * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(3 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, 3),
        )
        kernel, width, old_reduced, new_reduced = _sinc_resample_kernel(250, 200)
        self.register_buffer("resample_kernel", kernel, persistent=False)
        self.resample_width = width
        self.resample_old_reduced = old_reduced
        self.resample_new_reduced = new_reduced
        exact_parameter_count = sum(state[key].numel() for key in state)
        self.load_report = {
            "model_identity": "CodeBrain pretrained faithful adaptation to MI-32",
            "source": source_report,
            "checkpoint_path": str(weights.resolve()),
            "checkpoint_source": CODEBRAIN_CHECKPOINT_SOURCE,
            "checkpoint_revision": CODEBRAIN_CHECKPOINT_REVISION,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_expected_sha256": CODEBRAIN_CHECKPOINT_SHA256,
            "checkpoint_identity_verified": bool(
                verify_checkpoint_identity and CODEBRAIN_CHECKPOINT_SHA256 is not None
            ),
            "checkpoint_tensor_count": len(state),
            "checkpoint_parameter_count": exact_parameter_count,
            "exact_loaded_parameter_count": exact_parameter_count,
            "backbone_coverage": 1.0,
            "checkpoint_missing": list(result.missing_keys),
            "checkpoint_unexpected": list(result.unexpected_keys),
            "shape_mismatch": shape_mismatch,
            "numerical_load_max_abs_error": max(numerical_errors.values()),
            "channel_adapter": None,
            "source_channels": len(channel_order),
            "received_channel_order": self.channel_order,
            "formal_mi32_channel_order": EEGMAMBA_MI32_CHANNEL_ORDER,
            "channel_order_rule": "preserve the released MI-32 topology-oriented order; no spatial adapter",
            "input_scaling": "MI-32 volt-scale trials scaled x1e6 to microvolts (official TUEG unit); 250-to-200 Hz sinc resampling",
            "resampling": {
                "input_hz": 250,
                "output_hz": 200,
                "input_samples": 750,
                "output_samples": 600,
                "method": "Hann-windowed sinc polyphase bandlimited interpolation",
                "anti_aliasing": True,
                "patches": 3,
                "patch_samples": 200,
                "padding_samples": 0,
            },
            "proj_out": "Sequential() to match the official downstream wrapper convention",
            "classifier_mode": "flatten all patches (official wrapper style)",
            "classifier": "19200 -> 600 -> 200 -> 3 with ELU and Dropout(0.1)",
        }

    def prepare_input(self, x):
        if x.ndim != 3 or x.shape[1:] != (32, 750):
            raise ValueError(f"CodeBrain expects [B,32,750], got {list(x.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.adapter.transform(x)
        if x.shape[1:] != (32, 3, 200):
            raise RuntimeError(f"CodeBrain adapter must produce [B,32,3,200], got {list(x.shape)}")
        return x

    def forward(self, x):
        x = self.prepare_input(x)
        features = self.backbone(x)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        features = features.contiguous().view(x.shape[0], -1)
        return self.classifier(features)

    def optimizer_groups(self, lr):
        head_ids = {id(p) for p in self.classifier.parameters()}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": backbone, "lr": lr},
                {"params": self.classifier.parameters(), "lr": lr * 5}]


def validate_codebrain_structure(model, batch_size=2, run_full_forward=True,
                                 require_verified_checkpoint=True):
    """Fail closed on the frozen CodeBrain source, checkpoint and MI-32 path."""
    report = model.load_report
    if require_verified_checkpoint and not report["checkpoint_identity_verified"]:
        raise AssertionError("formal CodeBrain validation requires the official sealed checkpoint")
    if report["source"]["commit"] != CODEBRAIN_OFFICIAL_COMMIT:
        raise AssertionError("CodeBrain official source commit changed")
    if set(report["source"]["source_sha256"]) != set(CODEBRAIN_SOURCE_FILES):
        raise AssertionError("CodeBrain source hashes changed")
    if report["checkpoint_missing"] or report["checkpoint_unexpected"] or report["shape_mismatch"]:
        raise AssertionError("CodeBrain checkpoint has unresolved keys or shapes")
    if report["numerical_load_max_abs_error"] != 0.0:
        raise AssertionError("CodeBrain checkpoint load is not bit-exact")
    if hasattr(model, "channel_adapter") or report["channel_adapter"] is not None:
        raise AssertionError("CodeBrain must not contain the historical MI-32 to TUEG-19 adapter")
    if model.channel_order != EEGMAMBA_MI32_CHANNEL_ORDER:
        raise AssertionError("CodeBrain received a changed MI-32 channel order")

    backbone = model.backbone
    patch = backbone.patch_embedding
    temporal = patch.proj_in[0]
    if not isinstance(temporal, nn.Conv2d) or (
        temporal.in_channels, temporal.out_channels, temporal.kernel_size,
        temporal.stride, temporal.padding
    ) != (1, 25, (1, 49), (1, 25), (0, 24)) or temporal.bias is None:
        raise AssertionError("CodeBrain temporal patch stem differs from the frozen release")
    if not isinstance(patch.proj_in[1], nn.GroupNorm) or (
        patch.proj_in[1].num_groups, patch.proj_in[1].num_channels
    ) != (5, 25):
        raise AssertionError("CodeBrain GroupNorm stem differs")
    if not isinstance(patch.proj_in[2], nn.GELU):
        raise AssertionError("CodeBrain patch stem activation differs")
    spectral = patch.spectral_proj[0]
    if not isinstance(spectral, nn.Linear) or (
        spectral.in_features, spectral.out_features
    ) != (101, 200) or spectral.bias is None:
        raise AssertionError("CodeBrain spectral projection differs from the frozen release")
    if not isinstance(patch.spectral_proj[1], nn.Dropout) or patch.spectral_proj[1].p != 0.1:
        raise AssertionError("CodeBrain spectral dropout differs")
    positional = patch.positional_encoding[0]
    if not isinstance(positional, nn.Conv2d) or positional.groups != 200 or (
        positional.in_channels, positional.out_channels, positional.kernel_size,
        positional.padding
    ) != (200, 200, (19, 7), (9, 3)) or positional.bias is None:
        raise AssertionError("CodeBrain positional encoding differs from the frozen release")
    if backbone.proj_out.__class__.__name__ != "Sequential":
        raise AssertionError("CodeBrain downstream proj_out must be Sequential()")
    if not isinstance(model.classifier, nn.Sequential):
        raise AssertionError("CodeBrain classifier must be an official-style MLP head")
    linear_layers = [module for module in model.classifier if isinstance(module, nn.Linear)]
    if [(layer.in_features, layer.out_features) for layer in linear_layers] != [
        (19_200, 600), (600, 200), (200, 3)
    ]:
        raise AssertionError("CodeBrain classifier is not the adapted official flatten-all-patches head")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260817)
    x = torch.randn(batch_size, 32, 750, generator=generator, device=device)
    first = model.prepare_input(x)
    second = model.prepare_input(x)
    if tuple(first.shape) != (batch_size, 32, 3, 200) or not torch.equal(first, second):
        raise AssertionError("CodeBrain input conversion is not deterministic [B,32,3,200]")

    captured = {}
    output_shape = None
    if run_full_forward:
        hooks = [
            backbone.register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("backbone_output", list(output.shape))
            ),
            model.classifier[0].register_forward_pre_hook(
                lambda _m, inputs: captured.__setitem__("classifier_input", list(inputs[0].shape))
            ),
        ]
        model.eval()
        with torch.no_grad():
            output = model(x)
        for hook in hooks:
            hook.remove()
        output_shape = list(output.shape)
        expected = {
            "backbone_output": [batch_size, 32, 3, 200],
            "classifier_input": [batch_size, 19_200],
        }
        if captured != expected or output_shape != [batch_size, 3]:
            raise AssertionError(
                f"CodeBrain active path changed: captured={captured}, output={output_shape}"
            )

    return {
        "status": "PASS",
        "full_forward_executed": bool(run_full_forward),
        "input_shape": [batch_size, 32, 750],
        "resampled_shape": [batch_size, 32, 600],
        "patch_shape": list(first.shape),
        "active_shapes": captured,
        "classifier_input": captured.get("classifier_input"),
        "output_shape": output_shape,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "checkpoint_coverage": 1.0,
        "checkpoint_max_abs_error": report["numerical_load_max_abs_error"],
        "channel_adapter": None,
        "classifier_mode": report["classifier_mode"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["LaBraM", "EEGMamba", "CodeBrain"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--max-trials-per-subject", type=int, default=0)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--balanced-loss", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--preflight-load-only", action="store_true")
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the pretrained backbone (official --frozen variant); train the classifier head only",
    )
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    dataset_manifest_sha256 = sha256_file(root / "SHA256SUMS")
    data_artifact_sha256 = {
        name: sha256_file(root / name)
        for name in (
            "config.json", "subjects.csv", "splits.csv", "trials.csv",
            "provenance.csv", "quality_flags.csv", "release_audit.json",
            "electrode_adjacency.npy", "SHA256SUMS",
        )
    }
    split = pd.read_csv(root / "splits.csv")
    subjects = pd.read_csv(root / "subjects.csv")
    test = set(split.loc[split.fold == args.fold, "subject_id"].astype(int))
    val_fold = 1 if args.fold == 0 else 0
    val = set(split.loc[split.fold == val_fold, "subject_id"].astype(int)) - test
    all_ids = set(subjects.subject_id.astype(int))
    train = all_ids - test - val
    split_sets = [("train", train), ("val", val)]
    if args.final_test:
        split_sets.append(("test", test))
    dataset_class = LaBraMMI3Trials if args.model == "LaBraM" else MI3RawTrials
    datasets = {
        name: dataset_class(args.data, ids, args.max_trials_per_subject, args.preload and name != "test")
        for name, ids in split_sets
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_sampler=SubjectBatchSampler(dataset, args.batch_size, name == "train"),
            num_workers=0, pin_memory=True,
        )
        for name, dataset in datasets.items()
    }
    channel_order = json.loads((root / "config.json").read_text(encoding="utf-8"))["channel_order"]
    source_snapshot_report = None
    structure_audit = None
    split_hash = hashlib.sha256(
        ",".join(map(str, sorted(train))).encode("utf-8")
    ).hexdigest()

    def _train_stream():
        raw = MI3RawTrials(args.data, sorted(train), args.max_trials_per_subject, False)
        for sid in raw.subject_ids:
            x, y, _ = raw._load_subject(sid)
            yield sid, x, y

    if args.model == "LaBraM":
        adapter = LaBraMAdapter(channel_order)
        adapter.fit(_train_stream(), split_hash)
        adapter.save_manifest(out)
        model = LaBraMClassifier(channel_order, args.weights, adapter=adapter)
        structure_audit = validate_labram_structure(model)
        source_snapshot_report = snapshot_labram_sources(out, model)
        (out / "run_manifest.json").write_text(json.dumps({
            "model_identity": "LaBraM pretrained faithful adaptation to MI-32",
            "source_snapshot": source_snapshot_report,
            "checkpoint": model.load_report,
            "structure_audit": structure_audit,
            "adapter_manifest": adapter.manifest(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "data_artifact_sha256": data_artifact_sha256,
            "structure_audit_used_dataset_samples": False,
        }, indent=2), encoding="utf-8")
        implementation = "official LaBraM base via sealed Braindecode 1.7.0 adaptation"
        pretraining = "author checkpoint, state_dict-converted and identity-verified"
    elif args.model == "EEGMamba":
        adapter = EEGMambaAdapter(channel_order)
        adapter.fit(_train_stream(), split_hash)
        adapter.save_manifest(out)
        model = EEGMambaClassifier(channel_order, args.repo, args.weights, adapter=adapter)
        structure_audit = validate_eegmamba_structure(model, run_full_forward=False)
        source_snapshot_report = snapshot_eegmamba_sources(out, model)
        (out / "run_manifest.json").write_text(json.dumps({
            "model_identity": "EEGMamba pretrained faithful adaptation to MI-32",
            "source_snapshot": source_snapshot_report,
            "checkpoint": model.load_report,
            "structure_audit": structure_audit,
            "adapter_manifest": adapter.manifest(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "data_artifact_sha256": data_artifact_sha256,
            "structure_audit_used_dataset_samples": False,
        }, indent=2), encoding="utf-8")
        implementation = "official frozen EEGMamba architecture with faithful MI-32 dimensional adaptation"
        pretraining = "official identity-verified EEGMamba pretrained checkpoint"
    elif args.model == "CodeBrain":
        adapter = CodeBrainAdapter(channel_order)
        adapter.fit(_train_stream(), split_hash)
        adapter.save_manifest(out)
        model = CodeBrainClassifier(channel_order, args.repo, args.weights, adapter=adapter)
        if args.freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
        structure_audit = validate_codebrain_structure(
            model,
            run_full_forward=False,
            require_verified_checkpoint=bool(CODEBRAIN_CHECKPOINT_SHA256),
        )
        source_snapshot_report = snapshot_codebrain_sources(out, model)
        (out / "run_manifest.json").write_text(json.dumps({
            "model_identity": "CodeBrain pretrained faithful adaptation to MI-32",
            "source_snapshot": source_snapshot_report,
            "checkpoint": model.load_report,
            "structure_audit": structure_audit,
            "adapter_manifest": adapter.manifest(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "data_artifact_sha256": data_artifact_sha256,
            "structure_audit_used_dataset_samples": False,
        }, indent=2), encoding="utf-8")
        implementation = "official CodeBrain repository architecture with flatten-all-patches MI-32 adaptation"
        if args.freeze_backbone:
            implementation += " (backbone frozen, official --frozen variant; head-only fine-tune)"
        pretraining = "official CodeBrain pretrained checkpoint"
    else:
        raise RuntimeError(f"unsupported model: {args.model}")
    if args.preflight_load_only:
        print(json.dumps({
            "preflight_load_only": "ok", "model": args.model,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_load_report": model.load_report,
            "structure_audit": structure_audit,
            "source_snapshot": source_snapshot_report,
            "test_dataset_instantiated": False,
        }), flush=True)
        return
    device = torch.device("cuda")
    model = model.to(device)
    if args.preflight:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        preflight_start = time.time()
        x, y, _ = next(iter(loaders["train"]))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if args.model == "EEGMamba":
            structure_audit = validate_eegmamba_structure(model, run_full_forward=True)
            model.train()
        elif args.model == "CodeBrain":
            structure_audit = validate_codebrain_structure(
                model,
                run_full_forward=True,
                require_verified_checkpoint=bool(CODEBRAIN_CHECKPOINT_SHA256),
            )
            model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            preflight_loss = nn.CrossEntropyLoss()(logits.float(), y)
        preflight_loss.backward()
        torch.cuda.synchronize()
        print(json.dumps({
            "preflight": "ok", "model": args.model, "logits_shape": list(logits.shape),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "input_shape": list(x.shape),
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch2_forward_seconds": time.time() - preflight_start,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "checkpoint_load_report": model.load_report,
            "structure_audit": structure_audit,
            "source_snapshot": source_snapshot_report,
            "test_dataset_instantiated": False,
        }), flush=True)
        return

    counts = np.zeros(3, dtype=np.int64)
    for sid in sorted(train):
        _, labels, _ = datasets["train"]._read_subject_raw(sid, signals=False)
        selected_labels = labels[datasets["train"].selected[sid]]
        counts += np.bincount(selected_labels, minlength=3)[:3]
    weights = None
    if args.balanced_loss:
        weights = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32, device=device)
    print(json.dumps({
        "training_class_counts": counts.tolist(),
        "class_weights": weights.cpu().tolist() if weights is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_dataset_instantiated": args.final_test,
        "checkpoint_load_report": model.load_report,
    }), flush=True)

    ce = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(args.lr), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(loaders["train"])), eta_min=max(1e-6, args.lr * 0.05),
    )
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        start = time.time()
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        val_metrics = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": total / n, "seconds": time.time() - start,
               "backbone_lr": optimizer.param_groups[0]["lr"],
               "head_lr": optimizer.param_groups[1]["lr"],
               **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")

    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    rows = []
    for name in (["val", "test"] if args.final_test else ["val"]):
        metrics = evaluate(model, loaders[name], device)
        rows.append({
            "model": args.model, "implementation": implementation, "pretraining": pretraining,
            "dataset_version": json.loads((root / "config.json").read_text(encoding="utf-8")).get("version", "unknown"),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model_adapter_kind": "model_input_adapter",
            "input_channels": len(channel_order),
            "fold": args.fold, "split": name, "seed": args.seed, "best_epoch": checkpoint["epoch"],
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "args": vars(args), "train_subjects": sorted(train), "val_subjects": sorted(val),
        "test_subjects": sorted(test), "checkpoint_load_report": model.load_report,
        "structure_audit": structure_audit, "source_snapshot": source_snapshot_report,
        "test_was_not_used_for_selection": True,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "data_artifact_sha256": data_artifact_sha256,
        "model_adapter_manifest": adapter.manifest(),
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
