"""EEG-Conformer 数据适配器。

官方输入条件: Song et al. 2022 EEG Conformer 官方实现接收 [B,C,T] 原始信号,
官方预处理为训练集全局标准化 (非逐 trial)。
MI-32 适配 (B 类): 训练集全局 mean/std 标准化 (V 直接, 无单位换算);
数值协议: 关闭 AMP/TF32 (在训练入口, 见 mi3_eegnet.py main)。
消融:
  - none:    无适配 (V 直接输入);
  - naive:   逐 trial z-score (泄漏变体);
  - official: 本 adapter (训练集全局标准化)。
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


class EEGConformerAdapter(ModelAdapter):
    name = "eegconformer"
    model_name = "EEG-Conformer"
    official_reference = (
        "Song et al. 2022 (EEG Conformer, IEEE TNSRE); "
        "官方实现训练集全局标准化预处理; MI-32 适配: 全局 mean/std (B 类, 无单位换算)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "channels": 32, "samples": 750, "fs": 250,
        "unit": "standardized: (V - train_mean) / train_std", "dtype": "float32",
    }

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        total = np.float64(0.0)
        total_sq = np.float64(0.0)
        n_obs = 0
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            xf = X.astype(np.float64, copy=False)
            total += xf.sum(dtype=np.float64)
            total_sq += np.square(xf).sum(dtype=np.float64)
            n_obs += xf.size
            n_trials += int(X.shape[0])
            n_subjects += 1
        if n_obs <= 0:
            raise RuntimeError("EEG-Conformer fit 需要至少一个训练 trial")
        mean = total / n_obs
        var = total_sq / n_obs - mean * mean
        std = math.sqrt(max(float(var), 0.0))
        if std < 1e-12:
            raise RuntimeError("EEG-Conformer 训练集全局标准差近零")
        stats = AdapterStats(
            source_split="train",
            split_hash=split_hash,
            n_subjects=n_subjects,
            n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={"mean": float(np.float32(mean)), "std": float(np.float32(std))},
        )
        self.set_stats(stats)
        return stats

    def transform(self, x):
        if self._stats is None:
            raise RuntimeError("EEGConformerAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        xt = (xt - np.float32(self._stats.values["mean"])) / np.float32(self._stats.values["std"])
        return xt if is_torch else xt.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveConformerAdapter(), "official": self}


class NaiveConformerAdapter(ModelAdapter):
    """消融态 naive: 逐 trial z-score (泄漏变体)。"""
    name = "eegconformer_naive"
    model_name = "EEG-Conformer (naive per-trial z-score)"
    official_reference = "消融对照: 逐 trial z-score (禁止项示范)"
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {"note": "per-trial z-score (泄漏变体, 仅消融使用)"}

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
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return (x - mean) / np.maximum(std, np.float32(1e-6))

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
