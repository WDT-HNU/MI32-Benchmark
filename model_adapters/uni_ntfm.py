"""Uni-NTFM 数据适配器。

官方输入条件: Chen et al. 2024 Uni-NTFM: 5 区域 (frontal/central/parietal/
temporal/occipital) x 24 电极布局, 时频投影, MoE。官方无可直接复用的下游
分类头/checkpoint -> 本项目为 protocol_benchmark (自定义适配明示)。
MI-32 适配 (C 类明示 + B 类):
  1. 32 通道 -> 5x24 区域映射 (缺失位置 padding);
  2. padding mask 标记无效位置 (模型注意力屏蔽);
  3. 自定义监督头 (UniClassifier) 不在 adapter 内。
单位修正 (2026-08-27): 原实现未做 V->µV 转换。本机 compact 迷你训练验证:
V 输入 4ep 全程塌缩 (val macro_f1 ≤0.22), ×1e6 后 ep3 突破退化 (0.33)。
消融:
  - none:    无适配 -> shape 契约错误;
  - naive:   逐 trial z-score (泄漏变体);
  - official: 本 adapter (V->µV x1e6 + 区域映射 + padding)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso

REGIONS = {
    "frontal": ["Fp1","Fp2","F3","F4","F7","F8","Fz","F1","F2","F5","F6","AF3","AF4","AF7","AF8","AFz","FT7","FT8","FC1","FC2","FC3","FC4","FC5","FC6"],
    "central": ["C3","C4","Cz","C1","C2","C5","C6","CP1","CP2","CP3","CP4","CP5","CP6","CPz"],
    "parietal": ["P3","P4","P7","P8","Pz","P1","P2","P5","P6","PO3","PO4","PO7","PO8","POz","CP1","CP2","CP3","CP4","CP5","CP6","CPz","TP7","TP8","TP9"],
    "temporal": ["T3","T4","T5","T6","T7","T8","FT7","FT8","TP7","TP8","TP9","TP10"],
    "occipital": ["O1","O2","Oz","PO3","PO4","PO7","PO8","POz","CB1","CB2"],
}


class UniNTFMAdapter(ModelAdapter):
    name = "uni_ntfm"
    model_name = "Uni-NTFM (protocol_benchmark)"
    official_reference = (
        "Chen et al. 2024 (Uni-NTFM, commit c0ce0152); 官方无下游分类契约 -> "
        "protocol_benchmark; MI-32 适配: V->µV x1e6 + 32通道->5x24区域映射 + padding mask (C 类明示 + B 类)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "shape": "[B, 5, 24, 750]", "layout": "5 regions x 24 electrodes",
        "note": "µV (x1e6 from V); padding mask 由 padding_mask() 提供 (True=无效)",
    }
    UNIT_SCALE = 1e6  # MI-32 raw volts -> microvolts (official EEG datasets scale)

    def __init__(self, channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("UniNTFMAdapter 需要 channel_order")
        self.channel_order = list(channel_order)
        lookup = {name.casefold(): i for i, name in enumerate(self.channel_order)}
        self.mapping = np.full((5, 24), -1, dtype=np.int64)
        self.region_names = list(REGIONS.keys())
        for r, channels in enumerate(REGIONS.values()):
            for e, ch in enumerate(channels[:24]):
                self.mapping[r, e] = lookup.get(ch.casefold(), -1)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            if X.shape[1:] != (32, 750):
                raise RuntimeError(f"Uni-NTFM 期望 [B,32,750], 得到 {list(X.shape)}")
            n_trials += int(X.shape[0])
            n_subjects += 1
        stats = AdapterStats(
            source_split="train", split_hash=split_hash,
            n_subjects=n_subjects, n_trials=n_trials,
            fit_timestamp=utcnow_iso(), values={},
        )
        self.set_stats(stats)
        return stats

    def transform(self, x):
        if self._stats is None:
            raise RuntimeError("UniNTFMAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            X = x.float()
            device = x.device
            valid = torch.as_tensor(self.mapping >= 0, device=device)
            src = torch.as_tensor(self.mapping[valid.cpu().numpy()], dtype=torch.long, device=device)
        else:
            X = np.asarray(x, dtype=np.float32)
            valid = self.mapping >= 0
            src = self.mapping[valid]
        B = X.shape[0]
        out = torch.zeros((B, 5, 24, 750), dtype=torch.float32, device=X.device) if is_torch else np.zeros((B, 5, 24, 750), np.float32)
        out[:, valid] = X[:, src] * self.UNIT_SCALE  # V -> µV
        return out if is_torch else out.astype(np.float32)

    def padding_mask(self, batch_size: int) -> np.ndarray:
        """(B,5,24) bool: True=padding/无效位置 (模型注意力屏蔽)。确定性。"""
        valid = self.mapping >= 0
        return np.broadcast_to(~valid, (batch_size, 5, 24))

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveUniNTFMAdapter(self.channel_order), "official": self}


class NaiveUniNTFMAdapter(UniNTFMAdapter):
    """消融态 naive: 逐 trial z-score on valid channels (泄漏变体)。"""
    name = "uni_ntfm_naive"
    model_name = "Uni-NTFM (naive per-trial z-score)"
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
        X = np.asarray(x, dtype=np.float32)
        out = np.zeros((X.shape[0], 5, 24, 750), np.float32)
        valid = self.mapping >= 0
        src = self.mapping[valid]
        sel = X[:, src].astype(np.float32)
        mean = sel.mean(axis=(1, 2), keepdims=True)
        std = sel.std(axis=(1, 2), keepdims=True)
        out[:, valid] = (sel - mean) / np.maximum(std, np.float32(1e-6))
        return out

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
