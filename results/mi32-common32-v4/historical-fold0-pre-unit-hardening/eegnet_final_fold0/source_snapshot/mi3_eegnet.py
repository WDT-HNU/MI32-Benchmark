import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset, Sampler

from model_adapters.codebrain import CodeBrainAdapter  # noqa: F401  (registry parity)
from model_adapters.eegconformer import EEGConformerAdapter
from model_adapters.eegnet import EEGNetAdapter
from model_adapters.eegmamba import EEGMambaAdapter  # noqa: F401
from model_adapters.labram import LaBraMAdapter  # noqa: F401
from model_adapters.rgnn import RGNNAdapter
from model_adapters.tsception import TSceptionAdapter
from model_adapters.uni_ntfm import UniNTFMAdapter  # noqa: F401


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MI3RawTrials(Dataset):
    """MI-3 fixed-channel trials; channel_mask is provenance only."""

    def __init__(self, root, subject_ids, max_trials_per_subject=0, preload=False):
        self.root = Path(root)
        self.subjects = pd.read_csv(
            self.root / "subjects.csv",
            usecols=["subject_id", "subject_name"],
        ).set_index("subject_id")
        self.items = []
        self.by_subject = []
        self.selected = {}
        self.cache = OrderedDict()
        self.preload = preload
        self.subject_ids = sorted(map(int, subject_ids))
        for sid in self.subject_ids:
            _, labels, _ = self._read_subject_raw(sid, signals=False)
            n = len(labels)
            if max_trials_per_subject and n > max_trials_per_subject:
                # Preserve the agreed 1:1:2 class ratio inside every tuning subject.
                q = max_trials_per_subject // 4
                quotas = [q, q, max_trials_per_subject - 2 * q]
                chosen_parts = []
                for label, quota in enumerate(quotas):
                    candidates = np.flatnonzero(labels == label)
                    if len(candidates) < quota:
                        raise RuntimeError(f"subject {sid}: insufficient class {label} trials")
                    positions = np.linspace(0, len(candidates) - 1, quota, dtype=int)
                    chosen_parts.append(candidates[positions])
                chosen = np.concatenate(chosen_parts)
            else:
                chosen = np.arange(n)
            self.selected[sid] = chosen
            group = []
            for local_trial, _ in enumerate(chosen):
                group.append(len(self.items))
                self.items.append((sid, local_trial))
            self.by_subject.append(group)
        if preload:
            for sid in self.subject_ids:
                self.cache[sid] = self._load_subject(sid)

    def _path(self, sid):
        name = self.subjects.loc[sid, "subject_name"]
        return self.root / "data" / f"{name}.npz"

    def _read_subject_raw(self, sid, signals=True):
        with np.load(self._path(sid), allow_pickle=False) as z:
            x = z["X"].copy() if signals else None
            return x, z["y"].astype(np.int64).copy(), z["channel_mask"].astype(bool).copy()

    def _load_subject(self, sid):
        x, y, mask = self._read_subject_raw(sid, signals=True)
        chosen = self.selected[sid]
        x, y, mask = x[chosen], y[chosen], mask[chosen]
        # Provenance stays attached for audit, but the frozen MI-32 loader must
        # not reinterpret it as validity or zero-fill permission.
        x = x.astype(np.float32, copy=False)
        return x, y, mask

    def _load(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        value = self._load_subject(sid)
        if not self.preload:
            self.cache.clear()
        self.cache[sid] = value
        return value

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sid, tid = self.items[index]
        x, y, _ = self._load(sid)
        return torch.from_numpy(x[tid]), int(y[tid]), sid


class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.groups = dataset.by_subject
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        groups = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(groups)
        for group_id in groups:
            ids = list(self.groups[group_id])
            if self.shuffle:
                random.shuffle(ids)
            for start in range(0, len(ids), self.batch_size):
                yield ids[start:start + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)


class AdapterDataset(MI3RawTrials):
    """MI-32 trials with the project adapter layer applied (adapter.transform).

    The adapter owns every input transformation (unit scale / channel order /
    standardization / feature extraction). Model code stays official.
    """

    def __init__(self, root, subject_ids, max_trials_per_subject=0, preload=False, adapter=None):
        if adapter is None:
            raise ValueError("AdapterDataset requires a project adapter")
        if adapter.stats is None:
            raise ValueError("AdapterDataset requires a fitted adapter (stats from train subjects only)")
        self.adapter = adapter
        super().__init__(root, subject_ids, max_trials_per_subject, preload)

    def _load_subject(self, sid):
        x, y, mask = super()._load_subject(sid)
        x = self.adapter.transform(x)
        return x, y, mask



def _train_eval_loaders(dataset_cls, data_root, split_ids, max_trials_per_subject, preload, batch_size, shuffle):
    return {
        name: DataLoader(
            dataset_cls(data_root, ids, max_trials_per_subject, preload),
            batch_size=batch_size,
            shuffle=shuffle if name == "train" else False,
            num_workers=0,
            pin_memory=True,
        )
        for name, ids in split_ids
    }


class EEGNet(nn.Module):
    """Official-aligned PyTorch EEGNet-8,2 topology for fixed-length EEG trials."""

    OFFICIAL_PAPER_URL = "https://arxiv.org/abs/1611.08024"
    OFFICIAL_REPOSITORY_URL = "https://github.com/vlawhern/arl-eegmodels"

    def __init__(
        self,
        n_chans=128,
        n_times=750,
        n_classes=3,
        dropout=0.5,
        f1=8,
        d=2,
        kern_length=64,
        separable_kernel=16,
        pool1=4,
        pool2=8,
    ):
        super().__init__()
        f2 = f1 * d
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.f1 = int(f1)
        self.d = int(d)
        self.f2 = int(f2)
        self.kern_length = int(kern_length)
        self.separable_kernel = int(separable_kernel)
        self.pool1 = int(pool1)
        self.pool2 = int(pool2)
        self.dropout_rate = float(dropout)
        for name in ("n_chans", "n_times", "n_classes", "f1", "d", "kern_length",
                     "separable_kernel", "pool1", "pool2"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        temporal_left = (self.kern_length - 1) // 2
        temporal_right = self.kern_length - 1 - temporal_left
        separable_left = (self.separable_kernel - 1) // 2
        separable_right = self.separable_kernel - 1 - separable_left
        self.temporal = nn.Sequential(
            nn.ZeroPad2d((temporal_left, temporal_right, 0, 0)),
            nn.Conv2d(1, f1, (1, self.kern_length), bias=False),
            nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f2, (n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool1)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.ZeroPad2d((separable_left, separable_right, 0, 0)),
            nn.Conv2d(f2, f2, (1, self.separable_kernel), groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool2)),
            nn.Dropout(dropout),
        )
        features = self._infer_classifier_features()
        if features < 1:
            raise ValueError("pooling configuration removes the complete temporal dimension")
        self.classifier = nn.Linear(features, n_classes)
        self.reset_parameters()

    def _forward_features(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        return self.separable(x)

    def _infer_classifier_features(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_chans, self.n_times)
            return int(self._forward_features(dummy).flatten(1).shape[1])

    def reset_parameters(self):
        """Match the Glorot/Xavier and BatchNorm initialization used by Keras EEGNet."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.reset_running_stats()

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"EEGNet expects [batch, channels, time], got {tuple(x.shape)}")
        if x.shape[1] != self.n_chans or x.shape[2] != self.n_times:
            raise ValueError(
                f"EEGNet expects [batch, {self.n_chans}, {self.n_times}], got {tuple(x.shape)}"
            )
        return self.classifier(self._forward_features(x).flatten(1))

    def constrain(self):
        # Max-norm constraints used by the original EEGNet formulation.
        with torch.no_grad():
            self.spatial[0].weight.renorm_(p=2, dim=0, maxnorm=1.0)
            self.classifier.weight.renorm_(p=2, dim=0, maxnorm=0.25)


def expected_eegnet_trainable_parameters(model):
    return (
        model.f1 * model.kern_length
        + 2 * model.f1
        + model.f2 * model.n_chans
        + 2 * model.f2
        + model.f2 * model.separable_kernel
        + model.f2 * model.f2
        + 2 * model.f2
        + model.n_classes * model.classifier.in_features
        + model.n_classes
    )


def validate_eegnet_structure(model, batch_size=2):
    """CPU-only, synthetic-input acceptance test for the EEGNet model definition."""
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("EEGNet structure validation must run on CPU")

    expected_parameters = expected_eegnet_trainable_parameters(model)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise AssertionError(f"EEGNet parameters: expected {expected_parameters}, got {actual_parameters}")
    if (model.n_chans, model.n_times, model.n_classes, model.f1, model.d,
            model.kern_length, model.separable_kernel, model.pool1, model.pool2) == (
            32, 750, 3, 8, 2, 64, 16, 4, 8) and actual_parameters != 2723:
        raise AssertionError(f"MI-32 EEGNet must have 2723 trainable parameters, got {actual_parameters}")

    spatial = model.spatial[0]
    separable_depthwise = model.separable[1]
    separable_pointwise = model.separable[2]
    if spatial.groups != model.f1 or spatial.kernel_size != (model.n_chans, 1):
        raise AssertionError("EEGNet spatial convolution is not the expected depthwise full-channel layer")
    if spatial.out_channels != model.f2:
        raise AssertionError("EEGNet spatial depth multiplier does not match F1 * D")
    if separable_depthwise.groups != model.f2 or separable_depthwise.kernel_size != (
            1, model.separable_kernel):
        raise AssertionError("EEGNet separable depthwise convolution is invalid")
    if separable_pointwise.kernel_size != (1, 1) or separable_pointwise.out_channels != model.f2:
        raise AssertionError("EEGNet separable pointwise convolution is invalid")
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if linear_layers != [model.classifier]:
        raise AssertionError("EEGNet contains an unexpected hidden linear layer")

    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if len(batch_norms) != 3:
        raise AssertionError(f"EEGNet must contain exactly three BatchNorm layers, got {len(batch_norms)}")
    for batch_norm in batch_norms:
        if batch_norm.eps != 1e-3 or batch_norm.momentum != 0.01:
            raise AssertionError("EEGNet BatchNorm is not aligned to Keras defaults")
        if not torch.equal(batch_norm.weight.detach(), torch.ones_like(batch_norm.weight)):
            raise AssertionError("EEGNet BatchNorm gamma must initialize to one")
        if not torch.equal(batch_norm.bias.detach(), torch.zeros_like(batch_norm.bias)):
            raise AssertionError("EEGNet BatchNorm beta must initialize to zero")

    original_mode = model.training
    model.eval()
    with torch.no_grad():
        synthetic_input = torch.zeros(batch_size, model.n_chans, model.n_times)
        features = model._forward_features(synthetic_input)
        output = model(synthetic_input)
    model.train(original_mode)
    if output.shape != (batch_size, model.n_classes):
        raise AssertionError(f"EEGNet output shape is {tuple(output.shape)}")
    expected_classifier_features = features.flatten(1).shape[1]
    if model.classifier.in_features != expected_classifier_features:
        raise AssertionError(
            "EEGNet classifier input dimension does not match the actual convolutional feature map"
        )

    spatial_weights = spatial.weight.detach().clone()
    classifier_weights = model.classifier.weight.detach().clone()
    with torch.no_grad():
        spatial.weight.fill_(10.0)
        model.classifier.weight.fill_(10.0)
    model.constrain()
    spatial_norms = spatial.weight.detach().flatten(1).norm(p=2, dim=1)
    classifier_norms = model.classifier.weight.detach().norm(p=2, dim=1)
    with torch.no_grad():
        spatial.weight.copy_(spatial_weights)
        model.classifier.weight.copy_(classifier_weights)
    if torch.any(spatial_norms > 1.000001) or torch.any(classifier_norms > 0.250001):
        raise AssertionError("EEGNet max-norm constraints are ineffective")

    return {
        "status": "PASS",
        "synthetic_input_shape": [batch_size, model.n_chans, model.n_times],
        "output_shape": list(output.shape),
        "feature_map_shape": list(features.shape),
        "classifier_in_features": model.classifier.in_features,
        "trainable_parameters": actual_parameters,
        "spatial_groups": spatial.groups,
        "spatial_kernel": list(spatial.kernel_size),
        "separable_depthwise_groups": separable_depthwise.groups,
        "separable_kernel": list(separable_depthwise.kernel_size),
        "linear_layers": len(linear_layers),
        "spatial_max_norm": float(spatial_norms.max()),
        "classifier_max_norm": float(classifier_norms.max()),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_executed_source(out):
    """Seal the exact Python source used by a formally audited model run."""
    source = Path(__file__).resolve()
    snapshot_dir = Path(out) / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / source.name
    shutil.copy2(source, snapshot)
    source_hash = sha256_file(source)
    if sha256_file(snapshot) != source_hash:
        raise RuntimeError("executed source snapshot hash mismatch")
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "source_file": str(source),
        "source_snapshot": str(snapshot),
        "source_sha256": source_hash,
        "git_commit": git_commit,
    }


def generate_TS_channel_order(original_order):
    """Reproduce the official TSception-v2 ``generate_TS_channel_order`` algorithm.

    Channels with odd numeric suffixes form the first (left) hemisphere and their
    even counterparts form the second (right) hemisphere. Midline channels without
    a numeric suffix are intentionally excluded, as in the authors' implementation.
    """
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        digit_count = sum(character.isdigit() for character in channel)
        if digit_count:
            chan_name.append(channel[:-digit_count])
            chan_num.append(int(channel[-digit_count:]))
            chan_final.append(channel)

    chan_pair = []
    for prefix, number in zip(chan_name, chan_num):
        chan_pair.append(prefix + str(number - 1 if number % 2 == 0 else number + 1))

    paired_interleaved = []
    for channel, pair in zip(chan_final, chan_pair):
        if channel not in paired_interleaved:
            paired_interleaved.extend([channel, pair])
    ordered = paired_interleaved[0::2] + paired_interleaved[1::2]

    lookup = {name.casefold(): name for name in original_order}
    missing = [name for name in ordered if name.casefold() not in lookup]
    if missing:
        raise ValueError(f"TSception channel pairs are incomplete; missing {missing}")
    if len(ordered) < 2 or len(ordered) % 2:
        raise ValueError("TSception requires a non-empty even number of paired channels")
    return [lookup[name.casefold()] for name in ordered]


class TSception(nn.Module):
    """Official TSception-v2 network with an MI-32 input adapter."""

    OFFICIAL_PAPER_URL = "https://doi.org/10.1109/TAFFC.2022.3169001"
    OFFICIAL_REPOSITORY_URL = "https://github.com/yi-ding-cs/TSception"

    def __init__(self, channel_order, n_classes=3, sampling_rate=250, n_times=750,
                 num_t=15, num_s=15, hidden=32, dropout=0.5):
        super().__init__()
        self.original_channel_order = tuple(channel_order)
        self.ts_channel_order = tuple(generate_TS_channel_order(channel_order))
        self.n_input_chans = len(self.original_channel_order)
        self.n_chans = len(self.ts_channel_order)
        self.n_classes = int(n_classes)
        self.sampling_rate = int(sampling_rate)
        self.n_times = int(n_times)
        self.num_t = int(num_t)
        self.num_s = int(num_s)
        self.hidden = int(hidden)
        self.dropout_rate = float(dropout)
        if self.n_times <= 0:
            raise ValueError("n_times must be positive")

        lookup = {name.casefold(): i for i, name in enumerate(self.original_channel_order)}
        self.register_buffer(
            "channel_indices",
            torch.tensor([lookup[name.casefold()] for name in self.ts_channel_order], dtype=torch.long),
        )
        self.temporal_kernels = tuple(int(ratio * self.sampling_rate) for ratio in (0.5, 0.25, 0.125))
        if min(self.temporal_kernels) < 1:
            raise ValueError("sampling_rate is too low for the official TSception temporal windows")

        def block(in_chans, out_chans, kernel, stride, pool):
            return nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=kernel, stride=stride),
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
            )

        self.t1 = block(1, self.num_t, (1, self.temporal_kernels[0]), 1, 8)
        self.t2 = block(1, self.num_t, (1, self.temporal_kernels[1]), 1, 8)
        self.t3 = block(1, self.num_t, (1, self.temporal_kernels[2]), 1, 8)
        self.bn_t = nn.BatchNorm2d(self.num_t)
        self.s1 = block(self.num_t, self.num_s, (self.n_chans, 1), 1, 2)
        self.s2 = block(
            self.num_t,
            self.num_s,
            (self.n_chans // 2, 1),
            (self.n_chans // 2, 1),
            2,
        )
        self.bn_s = nn.BatchNorm2d(self.num_s)
        self.fusion = block(self.num_s, self.num_s, (3, 1), 1, 4)
        self.bn_fusion = nn.BatchNorm2d(self.num_s)
        self.fc = nn.Sequential(
            nn.Linear(self.num_s, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.n_classes),
        )

    def _forward_features(self, x):
        if x.ndim != 3:
            raise ValueError(f"TSception expects [batch, channels, time], got {tuple(x.shape)}")
        if x.shape[1] != self.n_input_chans:
            raise ValueError(
                f"TSception expects [batch, {self.n_input_chans}, time], got {tuple(x.shape)}"
            )
        if x.shape[2] != self.n_times:
            raise ValueError(
                f"TSception expects [batch, {self.n_input_chans}, {self.n_times}], got {tuple(x.shape)}"
            )
        x = x.index_select(1, self.channel_indices).unsqueeze(1)
        out = torch.cat([self.t1(x), self.t2(x), self.t3(x)], dim=-1)
        out = self.bn_t(out)
        out = torch.cat([self.s1(out), self.s2(out)], dim=2)
        out = self.bn_s(out)
        out = self.bn_fusion(self.fusion(out))
        return out

    def forward(self, x):
        out = self._forward_features(x)
        out = out.mean(dim=-1).squeeze(-1)
        return self.fc(out)


def expected_tsception_trainable_parameters(model):
    """Closed-form parameter count for the authors' TSception-v2 implementation."""
    temporal = sum(model.num_t * (kernel + 1) for kernel in model.temporal_kernels)
    spatial_global = model.num_s * (model.num_t * model.n_chans + 1)
    spatial_hemisphere = model.num_s * (model.num_t * (model.n_chans // 2) + 1)
    fusion = model.num_s * (model.num_s * 3 + 1)
    batch_norm = 2 * model.num_t + 4 * model.num_s
    classifier = (
        model.hidden * model.num_s
        + model.hidden
        + model.n_classes * model.hidden
        + model.n_classes
    )
    return temporal + spatial_global + spatial_hemisphere + fusion + batch_norm + classifier


def validate_tsception_structure(model, n_times=750, batch_size=2):
    """CPU-only acceptance test for the official TSception-v2 model definition."""
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("TSception structure validation must run on CPU")
    if model.n_times != n_times:
        raise AssertionError(f"TSception n_times mismatch: expected {n_times}, got {model.n_times}")

    expected_parameters = expected_tsception_trainable_parameters(model)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise AssertionError(
            f"TSception parameters: expected {expected_parameters}, got {actual_parameters}"
        )
    if (
        model.n_input_chans,
        model.n_chans,
        n_times,
        model.n_classes,
        model.sampling_rate,
        model.num_t,
        model.num_s,
        model.hidden,
    ) == (32, 26, 750, 3, 250, 15, 15, 32) and actual_parameters != 13511:
        raise AssertionError(f"MI-32 TSception-v2 must have 13511 parameters, got {actual_parameters}")

    blocks = [model.t1, model.t2, model.t3, model.s1, model.s2, model.fusion]
    for block in blocks:
        if not (
            len(block) == 3
            and isinstance(block[0], nn.Conv2d)
            and isinstance(block[1], nn.LeakyReLU)
            and isinstance(block[2], nn.AvgPool2d)
        ):
            raise AssertionError("every TSception-v2 convolution block must be Conv2d-LeakyReLU-AvgPool2d")
    if [block[0].kernel_size for block in (model.t1, model.t2, model.t3)] != [
        (1, kernel) for kernel in model.temporal_kernels
    ]:
        raise AssertionError("TSception temporal kernels do not follow the sampling-rate windows")
    if model.s1[0].kernel_size != (model.n_chans, 1) or model.s1[0].stride != (1, 1):
        raise AssertionError("TSception global spatial branch is invalid")
    if model.s2[0].kernel_size != (model.n_chans // 2, 1) or model.s2[0].stride != (
        model.n_chans // 2,
        1,
    ):
        raise AssertionError("TSception hemisphere branch is invalid")
    if model.fusion[0].kernel_size != (3, 1) or model.fusion[2].kernel_size != (1, 4):
        raise AssertionError("TSception high-level fusion block is invalid")

    batch_norms = [model.bn_t, model.bn_s, model.bn_fusion]
    for batch_norm in batch_norms:
        if (
            batch_norm.eps != 1e-5
            or batch_norm.momentum != 0.1
            or not batch_norm.affine
            or not batch_norm.track_running_stats
        ):
            raise AssertionError("TSception BatchNorm must retain the official PyTorch defaults")

    original_mode = model.training
    model.eval()
    shapes = {}
    hooks = []
    for name in ("t1", "t2", "t3", "bn_t", "s1", "s2", "bn_s", "fusion", "bn_fusion"):
        module = getattr(model, name)
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, layer=name: shapes.__setitem__(layer, list(output.shape))
            )
        )
    with torch.no_grad():
        synthetic_input = torch.zeros(batch_size, model.n_input_chans, n_times)
        feature_map = model._forward_features(synthetic_input)
        output = model(synthetic_input)
    for hook in hooks:
        hook.remove()
    model.train(original_mode)
    if output.shape != (batch_size, model.n_classes):
        raise AssertionError(f"TSception output shape is {tuple(output.shape)}")
    if feature_map.shape != (batch_size, model.num_s, 1, 31):
        raise AssertionError(f"TSception feature map shape is {tuple(feature_map.shape)}")

    expected_shapes = {
        "t1": [batch_size, model.num_t, model.n_chans, 78],
        "t2": [batch_size, model.num_t, model.n_chans, 86],
        "t3": [batch_size, model.num_t, model.n_chans, 90],
        "bn_t": [batch_size, model.num_t, model.n_chans, 254],
        "s1": [batch_size, model.num_s, 1, 127],
        "s2": [batch_size, model.num_s, 2, 127],
        "bn_s": [batch_size, model.num_s, 3, 127],
        "fusion": [batch_size, model.num_s, 1, 31],
        "bn_fusion": [batch_size, model.num_s, 1, 31],
    }
    if (model.sampling_rate, n_times) == (250, 750) and shapes != expected_shapes:
        raise AssertionError(f"MI-32 TSception-v2 feature shapes differ: {shapes}")

    return {
        "status": "PASS",
        "variant": "TSception-v2 (TAFFC 2022/2023)",
        "synthetic_input_shape": [batch_size, model.n_input_chans, n_times],
        "feature_map_shape": list(feature_map.shape),
        "selected_channel_count": model.n_chans,
        "selected_channel_order": list(model.ts_channel_order),
        "channel_indices": model.channel_indices.tolist(),
        "temporal_kernels": list(model.temporal_kernels),
        "feature_shapes": shapes,
        "output_shape": list(output.shape),
        "trainable_parameters": actual_parameters,
        "initialization": "PyTorch Conv2d/Linear/BatchNorm defaults (official repository behavior)",
        "official_paper_url": TSception.OFFICIAL_PAPER_URL,
        "official_repository_url": TSception.OFFICIAL_REPOSITORY_URL,
    }


class EEGConformerPatchEmbedding(nn.Module):
    """Official spatial-temporal convolutional patch embedding."""

    def __init__(self, n_chans, emb_size=40, dropout=0.5):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_chans, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class EEGConformerMultiHeadAttention(nn.Module):
    """Official attention, including its sqrt(embedding-size) scaling."""

    def __init__(self, emb_size=40, num_heads=10, dropout=0.5):
        super().__init__()
        if emb_size % num_heads:
            raise ValueError("EEG-Conformer embedding size must be divisible by attention heads")
        self.emb_size = int(emb_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.emb_size // self.num_heads
        self.scaling = self.emb_size ** 0.5
        self.keys = nn.Linear(self.emb_size, self.emb_size)
        self.queries = nn.Linear(self.emb_size, self.emb_size)
        self.values = nn.Linear(self.emb_size, self.emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, x, mask=None):
        batch, tokens, _ = x.shape

        def split_heads(projected):
            return projected.reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        queries = split_heads(self.queries(x))
        keys = split_heads(self.keys(x))
        values = split_heads(self.values(x))
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        attention = torch.softmax(energy / self.scaling, dim=-1)
        attention = self.att_drop(attention)
        output = torch.einsum("bhqk,bhkd->bhqd", attention, values)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, self.emb_size)
        return self.projection(output)


class EEGConformerResidualAdd(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return x + self.function(x)


class EEGConformerFeedForward(nn.Sequential):
    def __init__(self, emb_size=40, expansion=4, dropout=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )


class EEGConformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=40, heads=10, dropout=0.5, expansion=4):
        super().__init__()
        self.attention = EEGConformerResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            EEGConformerMultiHeadAttention(emb_size, heads, dropout),
            nn.Dropout(dropout),
        ))
        self.feed_forward = EEGConformerResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            EEGConformerFeedForward(emb_size, expansion, dropout),
            nn.Dropout(dropout),
        ))

    def forward(self, x):
        x = self.attention(x)
        return self.feed_forward(x)


class EEGConformerReduceMean(nn.Module):
    """Parameter-free equivalent of the official einops Reduce('b n e -> b e')."""

    def forward(self, x):
        return x.mean(dim=1)


class EEGConformerClassificationHead(nn.Module):
    """Official active Flatten MLP plus its registered but inactive clshead."""

    def __init__(self, tokens, emb_size=40, n_classes=3):
        super().__init__()
        # conformer.py registers this module but its forward path does not call it.
        self.clshead = nn.Sequential(
            EEGConformerReduceMean(),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(tokens * emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.fc(x.contiguous().view(x.size(0), -1))


class EEGConformer(nn.Module):
    """Faithful EEG-Conformer adaptation for fixed-channel MI-32 trials."""

    OFFICIAL_PAPER_URL = "https://arxiv.org/abs/2106.07336"
    OFFICIAL_REPOSITORY_URL = "https://github.com/eeyhsong/EEG-Conformer"

    def __init__(self, n_chans=32, n_times=750, n_classes=3, dropout=0.5,
                 emb_size=40, depth=6, heads=10, expansion=4):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.emb_size = int(emb_size)
        self.depth = int(depth)
        self.heads = int(heads)
        self.expansion = int(expansion)
        self.dropout_rate = float(dropout)
        self.patch = EEGConformerPatchEmbedding(self.n_chans, self.emb_size, self.dropout_rate)
        temporal_after_conv = self.n_times - 25 + 1
        self.tokens = (temporal_after_conv - 75) // 15 + 1
        if self.tokens < 1:
            raise ValueError("EEG-Conformer convolution/pooling removes the temporal dimension")
        # Independent construction is intentional: the official ModuleList does not clone
        # one already-initialized block across all six depths.
        self.transformer = nn.ModuleList([
            EEGConformerEncoderBlock(
                self.emb_size,
                self.heads,
                self.dropout_rate,
                self.expansion,
            )
            for _ in range(self.depth)
        ])
        self.classifier = EEGConformerClassificationHead(
            self.tokens,
            self.emb_size,
            self.n_classes,
        )

    def forward_features(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(
                f"EEG-Conformer expects [batch, {self.n_chans}, {self.n_times}], got {tuple(x.shape)}"
            )
        x = self.patch(x.unsqueeze(1))
        for block in self.transformer:
            x = block(x)
        return x

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def expected_eegconformer_trainable_parameters(model):
    """Closed-form count for the official graph after the MI-32 task adaptations."""
    patch = (
        40 * 25 + 40
        + 40 * 40 * model.n_chans + 40
        + 2 * 40
        + model.emb_size * 40 + model.emb_size
    )
    attention = 4 * (model.emb_size * model.emb_size + model.emb_size)
    layer_norms = 4 * model.emb_size
    feed_forward = (
        model.emb_size * (model.expansion * model.emb_size)
        + model.expansion * model.emb_size
        + (model.expansion * model.emb_size) * model.emb_size
        + model.emb_size
    )
    transformer = model.depth * (attention + layer_norms + feed_forward)
    inactive_clshead = 2 * model.emb_size + model.emb_size * model.n_classes + model.n_classes
    flattened = model.tokens * model.emb_size
    active_classifier = (
        flattened * 256 + 256
        + 256 * 32 + 32
        + 32 * model.n_classes + model.n_classes
    )
    return patch + transformer + inactive_clshead + active_classifier


def validate_eegconformer_structure(model, batch_size=2):
    """CPU-only synthetic acceptance test for the frozen official EEG-Conformer graph."""
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("EEG-Conformer structure validation must run on CPU")
    if model.dropout_rate != 0.5:
        raise AssertionError("the frozen official EEG-Conformer patch/encoder dropout is 0.5")

    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = expected_eegconformer_trainable_parameters(model)
    if actual_parameters != expected_parameters:
        raise AssertionError(
            f"EEG-Conformer parameters: expected {expected_parameters}, got {actual_parameters}"
        )
    if (
        model.n_chans,
        model.n_times,
        model.n_classes,
        model.emb_size,
        model.depth,
        model.heads,
        model.expansion,
        model.tokens,
    ) == (32, 750, 3, 40, 6, 10, 4, 44) and actual_parameters != 631662:
        raise AssertionError(
            f"MI-32 EEG-Conformer must have 631662 trainable parameters, got {actual_parameters}"
        )

    shallow = model.patch.shallownet
    temporal, spatial, batch_norm, activation, pooling, patch_dropout = shallow
    if not isinstance(temporal, nn.Conv2d) or (
        temporal.in_channels,
        temporal.out_channels,
        temporal.kernel_size,
        temporal.stride,
        temporal.padding,
        temporal.bias is not None,
    ) != (1, 40, (1, 25), (1, 1), (0, 0), True):
        raise AssertionError("EEG-Conformer temporal convolution differs from conformer.py")
    if not isinstance(spatial, nn.Conv2d) or (
        spatial.in_channels,
        spatial.out_channels,
        spatial.kernel_size,
        spatial.stride,
        spatial.padding,
        spatial.bias is not None,
    ) != (40, 40, (model.n_chans, 1), (1, 1), (0, 0), True):
        raise AssertionError("EEG-Conformer spatial convolution does not cover all input channels")
    if not isinstance(batch_norm, nn.BatchNorm2d) or (
        batch_norm.num_features,
        batch_norm.eps,
        batch_norm.momentum,
        batch_norm.affine,
        batch_norm.track_running_stats,
    ) != (40, 1e-5, 0.1, True, True):
        raise AssertionError("EEG-Conformer BatchNorm differs from the official PyTorch defaults")
    if batch_norm.num_batches_tracked.item() != 0:
        raise AssertionError("EEG-Conformer construction must not mutate BatchNorm running statistics")
    if not torch.equal(batch_norm.running_mean, torch.zeros_like(batch_norm.running_mean)):
        raise AssertionError("EEG-Conformer BatchNorm running mean is not pristine")
    if not torch.equal(batch_norm.running_var, torch.ones_like(batch_norm.running_var)):
        raise AssertionError("EEG-Conformer BatchNorm running variance is not pristine")
    if not isinstance(activation, nn.ELU):
        raise AssertionError("EEG-Conformer patch activation must be ELU")
    if not isinstance(pooling, nn.AvgPool2d) or (
        pooling.kernel_size,
        pooling.stride,
        pooling.padding,
    ) != ((1, 75), (1, 15), 0):
        raise AssertionError("EEG-Conformer patch pooling differs from conformer.py")
    if not isinstance(patch_dropout, nn.Dropout) or patch_dropout.p != model.dropout_rate:
        raise AssertionError("EEG-Conformer patch dropout is invalid")
    projection = model.patch.projection
    if not isinstance(projection, nn.Conv2d) or (
        projection.in_channels,
        projection.out_channels,
        projection.kernel_size,
        projection.stride,
        projection.padding,
        projection.bias is not None,
    ) != (40, model.emb_size, (1, 1), (1, 1), (0, 0), True):
        raise AssertionError("EEG-Conformer 1x1 patch projection is missing or invalid")

    if len(model.transformer) != model.depth:
        raise AssertionError("EEG-Conformer encoder depth differs from the frozen official graph")
    if len({id(block) for block in model.transformer}) != model.depth:
        raise AssertionError("EEG-Conformer encoder blocks must be independent module instances")
    first_query = model.transformer[0].attention.function[1].queries.weight
    if model.depth > 1 and all(
        torch.equal(first_query, block.attention.function[1].queries.weight)
        for block in model.transformer[1:]
    ):
        raise AssertionError("EEG-Conformer encoder blocks were initialized as identical clones")
    for block in model.transformer:
        attention_path = block.attention.function
        feed_forward_path = block.feed_forward.function
        if not (
            isinstance(attention_path, nn.Sequential)
            and len(attention_path) == 3
            and isinstance(attention_path[0], nn.LayerNorm)
            and isinstance(attention_path[1], EEGConformerMultiHeadAttention)
            and isinstance(attention_path[2], nn.Dropout)
        ):
            raise AssertionError("EEG-Conformer attention is not a Pre-LN residual block")
        attention = attention_path[1]
        if (
            attention.emb_size,
            attention.num_heads,
            attention.head_dim,
            attention.scaling,
            attention.att_drop.p,
            attention_path[2].p,
        ) != (
            model.emb_size,
            model.heads,
            model.emb_size // model.heads,
            model.emb_size ** 0.5,
            model.dropout_rate,
            model.dropout_rate,
        ):
            raise AssertionError("EEG-Conformer attention heads, scaling, or dropout differ")
        if not (
            isinstance(feed_forward_path, nn.Sequential)
            and len(feed_forward_path) == 3
            and isinstance(feed_forward_path[0], nn.LayerNorm)
            and isinstance(feed_forward_path[1], EEGConformerFeedForward)
            and isinstance(feed_forward_path[2], nn.Dropout)
        ):
            raise AssertionError("EEG-Conformer feed-forward is not a Pre-LN residual block")
        feed_forward = feed_forward_path[1]
        if (
            feed_forward[0].in_features,
            feed_forward[0].out_features,
            feed_forward[2].p,
            feed_forward[3].in_features,
            feed_forward[3].out_features,
            feed_forward_path[2].p,
        ) != (
            model.emb_size,
            model.expansion * model.emb_size,
            model.dropout_rate,
            model.expansion * model.emb_size,
            model.emb_size,
            model.dropout_rate,
        ):
            raise AssertionError("EEG-Conformer feed-forward expansion or dropout differs")

    if any(isinstance(module, nn.TransformerEncoder) for module in model.modules()):
        raise AssertionError("generic TransformerEncoder cannot prove official attention semantics")
    forbidden_names = [
        name for name, _ in model.named_parameters()
        if "pos" in name.casefold() or "cls_token" in name.casefold()
    ]
    forbidden_names.extend(
        name for name, _ in model.named_buffers()
        if "pos" in name.casefold() or "cls_token" in name.casefold()
    )
    if forbidden_names:
        raise AssertionError(f"EEG-Conformer contains forbidden position/CLS parameters: {forbidden_names}")

    clshead = model.classifier.clshead
    fc = model.classifier.fc
    if not (
        len(clshead) == 3
        and isinstance(clshead[0], EEGConformerReduceMean)
        and isinstance(clshead[1], nn.LayerNorm)
        and isinstance(clshead[2], nn.Linear)
    ):
        raise AssertionError("EEG-Conformer inactive clshead does not match the registered official branch")
    expected_fc = [nn.Linear, nn.ELU, nn.Dropout, nn.Linear, nn.ELU, nn.Dropout, nn.Linear]
    if len(fc) != len(expected_fc) or any(
        not isinstance(module, expected_type) for module, expected_type in zip(fc, expected_fc)
    ):
        raise AssertionError("EEG-Conformer active Flatten MLP layer sequence differs")
    if (
        fc[0].in_features,
        fc[0].out_features,
        fc[2].p,
        fc[3].in_features,
        fc[3].out_features,
        fc[5].p,
        fc[6].in_features,
        fc[6].out_features,
    ) != (
        model.tokens * model.emb_size,
        256,
        0.5,
        256,
        32,
        0.3,
        32,
        model.n_classes,
    ):
        raise AssertionError("EEG-Conformer active Flatten MLP dimensions/dropouts differ")

    calls = {"clshead": 0, "fc": 0}
    cls_hook = clshead.register_forward_hook(
        lambda _module, _inputs, _output: calls.__setitem__("clshead", calls["clshead"] + 1)
    )
    fc_hook = fc.register_forward_hook(
        lambda _module, _inputs, _output: calls.__setitem__("fc", calls["fc"] + 1)
    )
    original_mode = model.training
    model.eval()
    with torch.no_grad():
        features = model.forward_features(torch.zeros(batch_size, model.n_chans, model.n_times))
        output = model(torch.zeros(batch_size, model.n_chans, model.n_times))
    cls_hook.remove()
    fc_hook.remove()
    model.train(original_mode)
    if features.shape != (batch_size, model.tokens, model.emb_size):
        raise AssertionError(f"EEG-Conformer token shape is {tuple(features.shape)}")
    if output.shape != (batch_size, model.n_classes):
        raise AssertionError(f"EEG-Conformer output shape is {tuple(output.shape)}")
    if calls != {"clshead": 0, "fc": 1}:
        raise AssertionError(f"EEG-Conformer classifier executed the wrong branch: {calls}")

    return {
        "status": "PASS",
        "variant": "Song et al. EEG-Conformer (TNSRE 2023)",
        "synthetic_input_shape": [batch_size, model.n_chans, model.n_times],
        "patch_shape": [batch_size, model.tokens, model.emb_size],
        "output_shape": list(output.shape),
        "tokens": model.tokens,
        "embedding_size": model.emb_size,
        "attention_heads": model.heads,
        "encoder_depth": model.depth,
        "attention_scaling": "sqrt(embedding_size)",
        "official_paper_url": EEGConformer.OFFICIAL_PAPER_URL,
        "official_repository_url": EEGConformer.OFFICIAL_REPOSITORY_URL,
        "active_classifier": f"Flatten({model.tokens * model.emb_size})-256-32-{model.n_classes}",
        "inactive_clshead_calls": calls["clshead"],
        "trainable_parameters": actual_parameters,
        "initialization": "independent blocks with PyTorch module defaults (official repository behavior)",
    }


RGNN_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
)

class RGNN(nn.Module):
    """Inductive RGNN backbone with MI-32 five-band log-power node features."""

    def __init__(
        self,
        initial_adjacency,
        channel_order,
        n_classes=3,
        hidden=64,
        k=2,
        dropout=0.5,
        sfreq=250,
        n_times=750,
        l1_alpha=1e-5,
        adapter=None,
    ):
        super().__init__()
        if adapter is None:
            raise ValueError("RGNN requires the project RGNNAdapter (graph + features live in the adapter layer)")
        self.adapter = adapter
        self.channel_order = tuple(channel_order)
        self.n_chans = len(self.channel_order)
        self.n_features = len(RGNN_BANDS)
        self.n_classes = int(n_classes)
        self.hidden = int(hidden)
        self.k = int(k)
        self.dropout_rate = float(dropout)
        self.sfreq = float(sfreq)
        self.n_times = int(n_times)
        self.l1_alpha = float(l1_alpha)
        self.bands = RGNN_BANDS
        if self.k < 1:
            raise ValueError("RGNN propagation order k must be positive")
        if self.l1_alpha < 0:
            raise ValueError("RGNN L1 coefficient must be non-negative")

        frequencies = torch.fft.rfftfreq(self.n_times, d=1.0 / self.sfreq)
        band_masks = torch.stack([
            (frequencies >= low) & (frequencies < high)
            for _name, low, high in self.bands
        ])
        if not torch.all(band_masks.any(dim=1)):
            raise ValueError("RGNN has an empty frequency band at the configured sampling rate")
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("band_masks", band_masks)

        distance_adjacency = torch.as_tensor(initial_adjacency, dtype=torch.float32)
        adjusted_adjacency = torch.as_tensor(self.adapter.adjacency(), dtype=torch.float32)
        if adjusted_adjacency.shape != distance_adjacency.shape:
            raise RuntimeError("RGNN adapter adjacency shape mismatch")
        self.global_pairs_applied = self.adapter.applied_pairs
        self.register_buffer("distance_adjacency", distance_adjacency.clone())
        tril = torch.tril_indices(self.n_chans, self.n_chans)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])
        self.edge_values = nn.Parameter(adjusted_adjacency[tril[0], tril[1]].clone())

        # This is the single W in Z = S^L X W. It must be applied after L propagations
        # to match the official NewSGConv implementation when the Linear has a bias.
        self.node_projection = nn.Linear(self.n_features, self.hidden)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden, self.n_classes)

    def raw_adjacency(self):
        adjacency = torch.zeros(
            self.n_chans,
            self.n_chans,
            device=self.edge_values.device,
            dtype=self.edge_values.dtype,
        )
        adjacency[self.tril_rows, self.tril_cols] = self.edge_values
        return adjacency + adjacency.transpose(0, 1) - torch.diag(adjacency.diagonal())

    def normalized_adjacency(self, valid=None):
        adjacency = self.raw_adjacency()
        degree = adjacency.abs().sum(1).clamp_min(1e-6)
        inv = degree.rsqrt()
        normalized = inv[:, None] * adjacency * inv[None, :]
        if valid is None:
            return normalized
        if valid.ndim != 2 or valid.shape[1] != self.n_chans:
            raise ValueError("RGNN valid-node mask has the wrong shape")
        # The sealed MI-32 contract keeps all 32 interpolated nodes valid.
        # The mask is accepted only for API compatibility and must not alter
        # the graph structure or pool away any channels.
        return normalized.unsqueeze(0).expand(valid.shape[0], -1, -1)

    def adjacency(self, valid):
        """Backward-compatible alias for the normalized, signed adjacency."""
        return self.normalized_adjacency(valid)

    def forward(self, features):
        # Features are produced by RGNNAdapter.transform (dataset layer).
        adjacency = self.normalized_adjacency().unsqueeze(0).expand(features.shape[0], -1, -1)
        for _ in range(self.k):
            features = torch.einsum("bij,bjf->bif", adjacency, features)
        features = torch.relu(self.node_projection(features))
        pooled = features.sum(1)
        return self.classifier(self.dropout(pooled))

    def regularization(self):
        """L1 penalty over the independent lower-triangular adjacency parameters."""
        return self.edge_values.abs().sum()


def rgnn_objective(model, logits, targets, criterion):
    """RGNN classification objective with its explicit adjacency L1 penalty."""
    return criterion(logits, targets) + model.l1_alpha * model.regularization()


def expected_rgnn_trainable_parameters(model):
    independent_edges = model.n_chans * (model.n_chans + 1) // 2
    graph_projection = model.hidden * model.n_features + model.hidden
    classifier = model.n_classes * model.hidden + model.n_classes
    return independent_edges + graph_projection + classifier


def validate_rgnn_structure(model, batch_size=2):
    """CPU-only acceptance test for the inductive RGNN backbone."""
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("RGNN structure validation must run on CPU")
    if not isinstance(model.edge_values, nn.Parameter) or not model.edge_values.requires_grad:
        raise AssertionError("RGNN adjacency must be a learnable Parameter")

    expected_parameters = expected_rgnn_trainable_parameters(model)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != expected_parameters:
        raise AssertionError(f"RGNN parameters: expected {expected_parameters}, got {actual_parameters}")
    if (
        model.n_chans,
        model.n_features,
        model.hidden,
        model.k,
        model.n_classes,
    ) == (32, 5, 64, 2, 3) and actual_parameters != 1107:
        raise AssertionError(f"MI-32 RGNN backbone must have 1107 parameters, got {actual_parameters}")

    distance = model.distance_adjacency
    if distance.shape != (model.n_chans, model.n_chans):
        raise AssertionError("RGNN distance adjacency has the wrong shape")
    if not torch.allclose(distance, distance.transpose(0, 1), atol=1e-7, rtol=0):
        raise AssertionError("RGNN distance adjacency is not symmetric")
    if not torch.allclose(distance.diagonal(), torch.ones_like(distance.diagonal())):
        raise AssertionError("RGNN distance adjacency must initialize self-loops to one")

    expected_raw = torch.as_tensor(model.adapter.adjacency(), dtype=torch.float32)
    expected_pairs = model.adapter.applied_pairs
    raw = model.raw_adjacency()
    if model.global_pairs_applied != expected_pairs or not torch.allclose(raw, expected_raw):
        raise AssertionError("RGNN paper global connections are not initialized correctly")
    if not torch.allclose(raw, raw.transpose(0, 1), atol=1e-7, rtol=0):
        raise AssertionError("RGNN learnable adjacency reconstruction is not symmetric")
    for left, right in model.global_pairs_applied:
        left_index = model.channel_order.index(left)
        right_index = model.channel_order.index(right)
        if raw[left_index, right_index] > 0:
            raise AssertionError(f"RGNN global edge {left}-{right} is not negative")

    normalized = model.normalized_adjacency()
    degree = raw.abs().sum(1).clamp_min(1e-6)
    expected_normalized = degree.rsqrt()[:, None] * raw * degree.rsqrt()[None, :]
    if not torch.allclose(normalized, expected_normalized, atol=1e-7, rtol=1e-6):
        raise AssertionError("RGNN normalization does not use the absolute signed degree")

    original_mode = model.training
    model.eval()
    torch.manual_seed(20260817)
    inputs = torch.randn(batch_size, model.n_chans, model.n_times)
    node_features = model.adapter.transform(inputs)
    if node_features.shape != (batch_size, model.n_chans, model.n_features):
        raise AssertionError(f"RGNN node feature shape is {tuple(node_features.shape)}")
    valid = torch.zeros(batch_size, model.n_chans, dtype=torch.bool)
    batch_adjacency = model.normalized_adjacency(valid)
    if not torch.allclose(
        batch_adjacency,
        normalized.unsqueeze(0).expand(batch_size, -1, -1),
        atol=1e-7,
        rtol=1e-6,
    ):
        raise AssertionError("RGNN valid-node masks must be provenance-only and not change adjacency")
    propagated = node_features
    for _ in range(model.k):
        propagated = torch.einsum("bij,bjf->bif", batch_adjacency, propagated)
    node_embeddings = torch.relu(model.node_projection(propagated))
    expected_output = model.classifier(node_embeddings.sum(1))
    output = model(node_features)
    model.train(original_mode)
    if not torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5):
        raise AssertionError("RGNN is not S^L X W -> ReLU -> sum pool -> dropout -> Linear")
    if output.shape != (batch_size, model.n_classes):
        raise AssertionError(f"RGNN output shape is {tuple(output.shape)}")

    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(batch_size) % model.n_classes
    classification_loss = criterion(output, targets)
    objective = rgnn_objective(model, output, targets, criterion)
    expected_l1_component = model.l1_alpha * model.regularization()
    if not torch.allclose(
        objective - classification_loss,
        expected_l1_component,
        atol=5e-6,
        rtol=1e-4,
    ):
        raise AssertionError("RGNN adjacency L1 penalty is absent from its objective")
    edge_gradient = torch.autograd.grad(objective, model.edge_values, retain_graph=False)[0]
    if not torch.isfinite(edge_gradient).all() or not torch.any(edge_gradient != 0):
        raise AssertionError("RGNN objective does not backpropagate to adjacency parameters")

    off_diagonal = ~torch.eye(model.n_chans, dtype=torch.bool)
    return {
        "status": "PASS",
        "identity": "RGNN backbone (without target-domain NodeDAT/EmotionDL)",
        "synthetic_input_shape": list(inputs.shape),
        "node_feature_shape": list(node_features.shape),
        "bands_hz": [[name, low, high] for name, low, high in model.bands],
        "adjacency_shape": list(raw.shape),
        "adjacency_parameterization": "learnable lower triangle reconstructed symmetrically",
        "self_loop_initial_value": float(raw.diagonal().min().detach()),
        "distance_edges_gt_0_1_all_fraction": float((distance > 0.1).float().mean()),
        "distance_edges_gt_0_1_off_diagonal_fraction": float(
            (distance[off_diagonal] > 0.1).float().mean()
        ),
        "paper_global_pairs": [list(pair) for pair in model.adapter.global_pairs()],
        "global_pairs_applied": [list(pair) for pair in model.global_pairs_applied],
        "negative_edge_count_undirected": len(model.global_pairs_applied),
        "signed_degree": "sum(abs(A_ij))",
        "propagation_order": model.k,
        "pooling": "sum",
        "dropout": model.dropout_rate,
        "classifier": f"Linear({model.hidden}, {model.n_classes})",
        "l1_alpha": model.l1_alpha,
        "l1_component": float(expected_l1_component.detach()),
        "output_shape": list(output.shape),
        "trainable_parameters": actual_parameters,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, subjects = [], [], []
    loss_sum = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    for x, y, sid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += ce(logits, y).item() * len(y)
        n += len(y)
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(1).cpu().tolist())
        subjects.extend(sid.tolist())
    recalls = recall_score(ys, ps, labels=[0, 1, 2], average=None, zero_division=0)
    ys_array = np.asarray(ys)
    ps_array = np.asarray(ps)
    true_upper = ys_array != 2
    predicted_upper = ps_array != 2
    correctly_detected_upper = true_upper & predicted_upper
    frame = pd.DataFrame({"y": ys_array, "p": ps_array, "sid": np.asarray(subjects)})
    per_subject = []
    for _sid, group in frame.groupby("sid", sort=False):
        gy = group["y"].to_numpy()
        gp = group["p"].to_numpy()
        per_subject.append({
            "accuracy": accuracy_score(gy, gp),
            "macro_f1": f1_score(gy, gp, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(gy, gp),
        })
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(ys, ps),
        "subject_mean_accuracy": float(np.mean([row["accuracy"] for row in per_subject])) if per_subject else float("nan"),
        "subject_mean_macro_f1": float(np.mean([row["macro_f1"] for row in per_subject])) if per_subject else float("nan"),
        "subject_mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in per_subject])) if per_subject else float("nan"),
        "recall_0": recalls[0],
        "recall_1": recalls[1],
        "recall_2": recalls[2],
        "upper_vs_non_accuracy": float(np.mean(true_upper == predicted_upper)),
        "upper_recall": float(np.mean(predicted_upper[true_upper])),
        "non_upper_recall": float(np.mean(~predicted_upper[~true_upper])),
        "upper_lr_end_to_end_accuracy": float(np.mean(ps_array[true_upper] == ys_array[true_upper])),
        "upper_lr_conditional_accuracy": (
            float(np.mean(ps_array[correctly_detected_upper] == ys_array[correctly_detected_upper]))
            if correctly_detected_upper.any() else float("nan")
        ),
        "upper_lr_conditional_n": int(correctly_detected_upper.sum()),
        "n_trials": n,
        "n_subjects": len(set(subjects)),
        "confusion_matrix": confusion_matrix(ys, ps, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["EEGNet", "TSception", "RGNN", "EEG-Conformer"], default="EEGNet")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--eegnet-kern-length", type=int, default=64)
    ap.add_argument("--eegnet-separable-kernel", type=int, default=16)
    ap.add_argument("--eegnet-pool1", type=int, default=4)
    ap.add_argument("--eegnet-pool2", type=int, default=8)
    ap.add_argument("--rgnn-l1-alpha", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--max-trials-per-subject", type=int, default=0)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--balanced-loss", action="store_true")
    ap.add_argument("--final-test", action="store_true", help="Only use after hyperparameters are frozen.")
    ap.add_argument("--preflight", action="store_true", help="Run one real forward/backward batch and exit.")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_provenance = (
        snapshot_executed_source(out)
        if args.model in {"EEGNet", "TSception", "RGNN", "EEG-Conformer"}
        else {}
    )
    split = pd.read_csv(Path(args.data) / "splits.csv")
    subjects = pd.read_csv(Path(args.data) / "subjects.csv", usecols=["subject_id"])
    test = set(split.loc[split.fold == args.fold, "subject_id"].astype(int))
    val_fold = 1 if args.fold == 0 else 0
    val = set(split.loc[split.fold == val_fold, "subject_id"].astype(int)) - test
    all_ids = set(subjects.subject_id.astype(int))
    train = all_ids - test - val
    if not train or not val or not test:
        raise RuntimeError("empty subject split")

    config = json.loads((Path(args.data) / "config.json").read_text(encoding="utf-8"))
    dataset_manifest_sha256 = sha256_file(Path(args.data) / "SHA256SUMS")
    adjacency_path = Path(args.data) / "electrode_adjacency.npy"
    if args.model == "RGNN" and not adjacency_path.exists():
        raise RuntimeError(f"missing distance-based RGNN adjacency: {adjacency_path}")
    n_chans = int(config["n_channels"])
    n_times = int(config["n_samples"])
    input_adapter = {}
    ts_channel_order = None
    ts_channel_indices = None
    ts_channel_mean = None
    ts_channel_std = None
    eegconformer_mean = None
    eegconformer_std = None
    eegnet_mean = None
    eegnet_std = None

    split_hash = hashlib.sha256(
        ",".join(map(str, sorted(train))).encode("utf-8")
    ).hexdigest()

    def _train_stream(ids):
        raw = MI3RawTrials(args.data, ids, args.max_trials_per_subject, False)
        for sid in ids:
            x, y, _ = raw._load_subject(sid)
            yield sid, x, y

    if args.model == "EEGNet":
        adapter = EEGNetAdapter()
        adapter.fit(_train_stream(train), split_hash)
        adapter.save_manifest(out)
        dataset_factory = lambda ids, preload: AdapterDataset(
            args.data, ids, args.max_trials_per_subject, preload, adapter=adapter
        )
        input_adapter = adapter.manifest()
    elif args.model == "TSception":
        adapter = TSceptionAdapter(config["channel_order"])
        adapter.fit(_train_stream(train), split_hash)
        adapter.save_manifest(out)
        dataset_factory = lambda ids, preload: AdapterDataset(
            args.data, ids, args.max_trials_per_subject, preload, adapter=adapter
        )
        input_adapter = adapter.manifest()
    elif args.model == "RGNN":
        adapter = RGNNAdapter(
            config["channel_order"],
            distance_adjacency=np.load(adjacency_path),
        )
        adapter.fit(_train_stream(train), split_hash)
        adapter.save_manifest(out)
        dataset_factory = lambda ids, preload: AdapterDataset(
            args.data, ids, args.max_trials_per_subject, preload, adapter=adapter
        )
        input_adapter = adapter.manifest()
    elif args.model == "EEG-Conformer":
        adapter = EEGConformerAdapter()
        adapter.fit(_train_stream(train), split_hash)
        adapter.save_manifest(out)
        dataset_factory = lambda ids, preload: AdapterDataset(
            args.data, ids, args.max_trials_per_subject, preload, adapter=adapter
        )
        input_adapter = adapter.manifest()
    else:
        raise RuntimeError(f"unsupported model: {args.model}")

    datasets = {
        "train": dataset_factory(train, args.preload),
        "val": dataset_factory(val, args.preload),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=True,
        )
        for name, dataset in datasets.items()
    }

    device = torch.device("cuda")
    data_artifacts = {
        "config.json": Path(args.data) / "config.json",
        "subjects.csv": Path(args.data) / "subjects.csv",
        "splits.csv": Path(args.data) / "splits.csv",
        "trials.csv": Path(args.data) / "trials.csv",
        "provenance.csv": Path(args.data) / "provenance.csv",
        "quality_flags.csv": Path(args.data) / "quality_flags.csv",
        "release_audit.json": Path(args.data) / "release_audit.json",
        "SHA256SUMS": Path(args.data) / "SHA256SUMS",
        "electrode_adjacency.npy": adjacency_path,
    }
    data_artifact_sha256 = {
        name: sha256_file(path)
        for name, path in data_artifacts.items()
        if path.exists()
    }
    constructors = {
        "EEGNet": lambda: EEGNet(
            n_chans=n_chans,
            n_times=n_times,
            dropout=args.dropout,
            kern_length=args.eegnet_kern_length,
            separable_kernel=args.eegnet_separable_kernel,
            pool1=args.eegnet_pool1,
            pool2=args.eegnet_pool2,
        ),
        "TSception": lambda: TSception(
            config["channel_order"], sampling_rate=int(config["sfreq"]), n_times=n_times,
            dropout=args.dropout,
        ),
        "RGNN": lambda: RGNN(
            torch.from_numpy(np.load(adjacency_path)),
            config["channel_order"],
            dropout=args.dropout,
            sfreq=int(config["sfreq"]),
            n_times=n_times,
            l1_alpha=args.rgnn_l1_alpha,
            adapter=adapter,
        ),
        "EEG-Conformer": lambda: EEGConformer(
            n_chans=n_chans, n_times=n_times, dropout=args.dropout,
        ),
    }
    model = constructors[args.model]()
    if args.model == "EEGNet":
        structure_audit = validate_eegnet_structure(model)
    elif args.model == "TSception":
        structure_audit = validate_tsception_structure(model, n_times=n_times)
    elif args.model == "RGNN":
        structure_audit = validate_rgnn_structure(model)
    elif args.model == "EEG-Conformer":
        structure_audit = validate_eegconformer_structure(model)
    else:
        structure_audit = None
    model = model.to(device)
    if args.preflight:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        x, y, _ = next(iter(loaders["train"]))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preflight_criterion = nn.CrossEntropyLoss()
        if isinstance(model, RGNN):
            preflight_loss = rgnn_objective(model, logits, y, preflight_criterion)
        else:
            preflight_loss = preflight_criterion(logits, y)
        preflight_loss.backward()
        torch.cuda.synchronize()
        print(json.dumps({
            "preflight": "ok",
            "model": args.model,
            "input_shape": list(x.shape),
            "logits_shape": list(logits.shape),
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": time.time() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "dataset_version": config.get("version", "unknown"),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "test_dataset_instantiated": False,
            "structure_audit": structure_audit,
            **source_provenance,
        }), flush=True)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = None
    training_counts = np.zeros(3, dtype=np.int64)
    for sid in sorted(train):
        _, labels, _ = datasets["train"]._read_subject_raw(sid, signals=False)
        selected_labels = labels[datasets["train"].selected[sid]]
        training_counts += np.bincount(selected_labels, minlength=3)[:3]
    if args.balanced_loss:
        class_weights = torch.tensor(
            training_counts.sum() / (3.0 * training_counts), dtype=torch.float32, device=device
        )
    print(json.dumps({
        "training_class_counts": training_counts.tolist(),
        "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_dataset_instantiated": args.final_test,
        "input_adapter": input_adapter,
    }), flush=True)

    ce = nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        start = time.time()
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if isinstance(model, RGNN):
                loss = rgnn_objective(model, logits, y, ce)
            else:
                loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if hasattr(model, "constrain"):
                model.constrain()
            total_loss += loss.item() * len(y)
            n += len(y)
        val_metrics = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n, 1),
            "seconds": time.time() - start,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")

    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if args.final_test:
        datasets["test"] = dataset_factory(test, False)
        loaders["test"] = DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    result_rows = []
    result_splits = ["val"]
    if args.final_test:
        result_splits.append("test")
    for name in result_splits:
        metrics = evaluate(model, loaders[name], device)
        result_rows.append({
            "model": args.model,
            "implementation": {
                "EEGNet": "official-aligned PyTorch EEGNet-8,2 topology",
                "TSception": "official TSception-v2 topology with official MI-32 channel reordering",
                "RGNN": "RGNN backbone without target-domain NodeDAT/EmotionDL; MI-32 log-power features",
                "EEG-Conformer": "official Song et al. EEG-Conformer graph with MI-32 task adaptations",
            }[args.model],
            "pretraining": "none supervised from scratch",
            "dataset_version": config.get("version", "unknown"),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model_adapter_kind": "model_input_adapter",
            "input_channels": n_chans,
            "fold": args.fold,
            "split": name,
            "seed": args.seed,
            "best_epoch": checkpoint["epoch"],
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out / "results.csv", index=False)
    checkpoint_sha256 = sha256_file(out / "best.pt")
    run_config = {
        "args": vars(args),
        "train_subjects": sorted(train),
        "val_subjects": sorted(val),
        "test_subjects": sorted(test),
        "training_class_counts": training_counts.tolist(),
        "test_was_not_used_for_selection": True,
        "data_version": config.get("version", "unknown"),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_config": config,
        "data_artifact_sha256": data_artifact_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "input_adapter": input_adapter,
        "model_adapter_manifest": input_adapter,
        "eegnet_structure_audit": structure_audit if args.model == "EEGNet" else None,
        "tsception_structure_audit": structure_audit if args.model == "TSception" else None,
        "rgnn_structure_audit": structure_audit if args.model == "RGNN" else None,
        "eegconformer_structure_audit": (
            structure_audit if args.model == "EEG-Conformer" else None
        ),
        **source_provenance,
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if args.model == "EEGNet":
        (out / "run_manifest.json").write_text(json.dumps({
            "model": "EEGNet",
            "implementation": "official-aligned PyTorch EEGNet-8,2 topology",
            "official_paper_url": EEGNet.OFFICIAL_PAPER_URL,
            "official_repository_url": EEGNet.OFFICIAL_REPOSITORY_URL,
            "data_version": config.get("version", "unknown"),
            "data_artifact_sha256": data_artifact_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "complete_run_config": str(out / "run_config.json"),
            "structure_audit": structure_audit,
            **source_provenance,
        }, indent=2), encoding="utf-8")
    elif args.model == "TSception":
        (out / "run_manifest.json").write_text(json.dumps({
            "model": "TSception",
            "implementation": "official TSception-v2 topology with official MI-32 channel reordering",
            "official_repository_commit": "9efd666b618d006e32e6da1d30dbc79b1d190604",
            "official_paper_url": TSception.OFFICIAL_PAPER_URL,
            "official_repository_url": TSception.OFFICIAL_REPOSITORY_URL,
            "data_version": config.get("version", "unknown"),
            "data_artifact_sha256": data_artifact_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "complete_run_config": str(out / "run_config.json"),
            "structure_audit": structure_audit,
            **source_provenance,
        }, indent=2), encoding="utf-8")
    elif args.model == "RGNN":
        (out / "run_manifest.json").write_text(json.dumps({
            "model": "RGNN backbone (without target-domain NodeDAT/EmotionDL)",
            "implementation": "signed learnable symmetric SGC backbone with MI-32 log-power features",
            "official_repository_commit": "685b1a2645185bd8129c04a789dbfde8ad896a59",
            "data_version": config.get("version", "unknown"),
            "data_artifact_sha256": data_artifact_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "complete_run_config": str(out / "run_config.json"),
            "structure_audit": structure_audit,
            **source_provenance,
        }, indent=2), encoding="utf-8")
    elif args.model == "EEG-Conformer":
        (out / "run_manifest.json").write_text(json.dumps({
            "model": "EEG-Conformer",
            "implementation": "official Song et al. graph with Chans=32, Samples=750, Classes=3",
            "official_repository_commit": "9ae149ba62487ceae723277d13adac27837113d2",
            "official_paper_url": EEGConformer.OFFICIAL_PAPER_URL,
            "official_repository_url": EEGConformer.OFFICIAL_REPOSITORY_URL,
            "data_version": config.get("version", "unknown"),
            "data_artifact_sha256": data_artifact_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "complete_run_config": str(out / "run_config.json"),
            "structure_audit": structure_audit,
            **source_provenance,
        }, indent=2), encoding="utf-8")
    print(pd.DataFrame(result_rows).to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
