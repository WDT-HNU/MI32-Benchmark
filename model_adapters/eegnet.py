"""EEGNet-8,2 数据适配器。

官方输入条件: Lawhern et al. 2018 的 EEGNet-8,2 期望 µV 量级输入。
MI-32 数据契约: [B,32,750]@250Hz, 单位 V。
本项目适配 (B 类, 任务必要):
  1. 单位换算 V -> mV (x1000, 与当前 mi3_eegnet.py 的 EEGNetScaledTrials 一致);
  2. 训练集全局 affine 标定 (mean/std 只从训练受试者拟合, 与 EEGNetCalibratedTrials 一致)。
消融:
  - none:    无适配 (V 直接输入) -> 数值量级错误, 模型无法正常训练;
  - naive:   逐 trial z-score -> 使用 trial 自身统计, 物理错误且构成泄漏变体;
  - official: 本 adapter (训练集全局标定)。
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


class EEGNetAdapter(ModelAdapter):
    name = "eegnet"
    model_name = "EEGNet-8,2"
    official_reference = (
        "Lawhern et al. 2018 (EEGNet: compact CNN for EEG-BCI); "
        "MI-32 适配: V->mV x1000 + 训练集全局 affine 标定 (B 类, 依据 mi3_eegnet.py EEGNetCalibratedTrials)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "channels": 32, "samples": 750, "fs": 250,
        "unit": "scaled: (V*1000 - train_mean) / train_std", "dtype": "float32",
    }

    UNIT_SCALE = np.float32(1000.0)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        total = np.float64(0.0)
        total_sq = np.float64(0.0)
        n_obs = 0
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            xf = X.astype(np.float64, copy=False) * float(self.UNIT_SCALE)
            total += xf.sum(dtype=np.float64)
            total_sq += np.square(xf).sum(dtype=np.float64)
            n_obs += xf.size
            n_trials += int(X.shape[0])
            n_subjects += 1
        if n_obs <= 0:
            raise RuntimeError("EEGNet fit 需要至少一个训练 trial")
        mean = total / n_obs
        var = total_sq / n_obs - mean * mean
        std = math.sqrt(max(float(var), 0.0))
        if std < 1e-12:
            raise RuntimeError("EEGNet 训练集全局标准差近零 (输入退化?)")
        stats = AdapterStats(
            source_split="train",
            split_hash=split_hash,
            n_subjects=n_subjects,
            n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={"unit_scale": float(self.UNIT_SCALE), "mean": float(np.float32(mean)), "std": float(np.float32(std))},
        )
        self.set_stats(stats)
        return stats

    def transform(self, x: np.ndarray) -> np.ndarray:
        """[B,32,750] V -> scaled input. Accepts numpy or torch (same type out)."""
        if self._stats is None:
            raise RuntimeError("EEGNetAdapter.transform 前必须先 fit (统计量只允许来自训练集)")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        xt = xt * self.UNIT_SCALE
        xt = (xt - np.float32(self._stats.values["mean"])) / np.float32(self._stats.values["std"])
        return xt if is_torch else xt.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveEEGNetAdapter(), "official": self}


class NaiveEEGNetAdapter(ModelAdapter):
    """消融态 naive: 逐 trial z-score (使用 trial 自身统计) -> 物理错误 + 泄漏变体。"""
    name = "eegnet_naive"
    model_name = "EEGNet-8,2 (naive per-trial z-score)"
    official_reference = "消融对照: 逐 trial z-score (禁止项示范)"
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {"note": "per-trial z-score of V*1000 (泄漏变体, 仅消融使用)"}

    UNIT_SCALE = np.float32(1000.0)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            n_trials += int(X.shape[0])
            n_subjects += 1
        stats = AdapterStats(
            source_split="train",
            split_hash=split_hash,
            n_subjects=n_subjects,
            n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={},  # 消融态 naive: 不拟合任何统计 (使用 trial 自身统计)
        )
        self.set_stats(stats)
        return stats

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32) * self.UNIT_SCALE
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return (x - mean) / np.maximum(std, np.float32(1e-6))

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
