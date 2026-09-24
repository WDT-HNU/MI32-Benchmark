"""TSception V2 数据适配器。

官方输入条件: TSception V2 (TAFFC 2022) 使用官方 generate_TS_channel_order()
算法生成通道序 (奇数数字后缀=左半球, 偶数=右半球, 无数字后缀中线通道排除),
并对选定通道做按通道标准化。
MI-32 适配 (B 类): 32 通道 -> 官方算法选出 26 个左右成对通道 (排除 Fz/FCz/Cz/CPz/Pz/Oz);
按通道 mean/std 只从训练受试者拟合。
消融:
  - none:    无适配 (V 直接输入);
  - naive:   逐 trial z-score (泄漏变体);
  - official: 本 adapter (官方通道序 + 训练集按通道标准化)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


def generate_TS_channel_order(original_order: Sequence[str]) -> List[str]:
    """复刻官方 TSception-v2 generate_TS_channel_order 算法 (与 mi3_eegnet.py 一致)。"""
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        digit_count = sum(c.isdigit() for c in channel)
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


class TSceptionAdapter(ModelAdapter):
    name = "tsception"
    model_name = "TSception V2"
    official_reference = (
        "Ding et al. 2022 (TSception V2, TAFFC, DOI 10.1109/TAFFC.2022.3169001); "
        "官方仓库 yi-ding-cs/TSception@9efd666; 通道序=官方 generate_TS_channel_order, "
        "MI-32 适配: 26 配对通道 + 训练集按通道标准化 (B 类)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "channels": 32, "samples": 750, "fs": 250,
        "note": "选定通道(官方配对序)按训练集通道统计标准化; 其余通道原样; 模型内部按官方通道序取数",
        "dtype": "float32",
    }

    def __init__(self, channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("TSceptionAdapter 需要 channel_order (数据 config 的 32 通道序)")
        self.channel_order = list(channel_order)
        self.ts_channel_order = generate_TS_channel_order(self.channel_order)
        lookup = {name.casefold(): i for i, name in enumerate(self.channel_order)}
        self.channel_indices = np.asarray(
            [lookup[name.casefold()] for name in self.ts_channel_order], dtype=np.int64
        )
        if len(self.ts_channel_order) != 26:
            raise ValueError(
                f"MI-32 TSception 应为 26 个配对通道, 得到 {len(self.ts_channel_order)} "
                f"-> 请核对 channel_order 与官方算法"
            )

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        total = np.zeros(len(self.channel_indices), dtype=np.float64)
        total_sq = np.zeros(len(self.channel_indices), dtype=np.float64)
        n_obs = 0
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            sel = X[:, self.channel_indices, :].astype(np.float64, copy=False)
            total += sel.sum(axis=(0, 2))
            total_sq += np.square(sel).sum(axis=(0, 2))
            n_obs += sel.shape[0] * sel.shape[2]
            n_trials += int(X.shape[0])
            n_subjects += 1
        if n_obs <= 0:
            raise RuntimeError("TSception fit 需要至少一个训练 trial")
        mean = total / n_obs
        var = total_sq / n_obs - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        if np.any(std < 1e-12):
            raise RuntimeError("TSception 训练集通道标准差含近零值")
        stats = AdapterStats(
            source_split="train",
            split_hash=split_hash,
            n_subjects=n_subjects,
            n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={
                "channel_order": self.ts_channel_order,
                "channel_indices": self.channel_indices.tolist(),
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
            },
        )
        self.set_stats(stats)
        return stats

    def transform(self, x):
        if self._stats is None:
            raise RuntimeError("TSceptionAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            xt = x.clone()
            mean = torch.as_tensor(self._stats.values["mean"], dtype=torch.float32, device=x.device)
            std = torch.as_tensor(self._stats.values["std"], dtype=torch.float32, device=x.device)
            idx = torch.as_tensor(self.channel_indices, dtype=torch.long, device=x.device)
        else:
            xt = np.asarray(x, dtype=np.float32).copy()
            mean = np.asarray(self._stats.values["mean"], dtype=np.float32)
            std = np.asarray(self._stats.values["std"], dtype=np.float32)
            idx = self.channel_indices
        sel = xt[:, idx, :]
        xt[:, idx, :] = (sel - mean[None, :, None]) / std[None, :, None]
        return xt if is_torch else xt.astype(np.float32, copy=False)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {
            "none": IdentityAdapter(),
            "naive": NaiveTSceptionAdapter(self.channel_order),
            "official": self,
        }


class NaiveTSceptionAdapter(TSceptionAdapter):
    """消融态 naive: 逐 trial z-score (使用 trial 自身统计, 泄漏变体)。"""
    name = "tsception_naive"
    model_name = "TSception V2 (naive per-trial z-score)"
    official_reference = "消融对照: 逐 trial z-score (禁止项示范)"

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
        x = np.asarray(x, dtype=np.float32).copy()
        sel = x[:, self.channel_indices, :]
        mean = sel.mean(axis=(1, 2), keepdims=True)
        std = sel.std(axis=(1, 2), keepdims=True)
        x[:, self.channel_indices, :] = (sel - mean) / np.maximum(std, np.float32(1e-6))
        return x

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
