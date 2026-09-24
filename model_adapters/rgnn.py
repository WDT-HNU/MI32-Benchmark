"""RGNN backbone 数据适配器。

官方: Zhong et al. 2019 RGNN 使用预计算 DE+LDS 节点特征 + 距离/全局邻接。
本项目 (明示差异): 归纳式 backbone-only (无 NodeDAT/EmotionDL);
节点特征 = 每通道五频带 log-power surrogate (delta/theta/alpha/beta/gamma,
与 mi3_eegnet.py RGNN_BANDS 一致); 邻接 = 距离初始化 + 论文全局对。
适配职责 (C 类明示 + B 类):
  1. [B,32,750] -> [B,32,5] 五频带 log-power (本项目 surrogate, 论文措辞必须注明);
  2. 邻接矩阵生成 (距离邻接 + 全局对, 供模型构造)。
消融:
  - none:    无适配 (直接喂 [B,32,750] 原始信号) -> 模型期望 [B,C,5], 必然失败;
  - naive:   线性功率 (不取 log, 不 clamp) -> 数值尺度错误;
  - official: 本 adapter (log-power + 训练集无统计, 特征无参)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso

RGNN_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
)

# 论文原版全局对 (Zhong et al. 2019, Fig. 2): 9 对左右半球对称连接
RGNN_GLOBAL_PAIRS = (
    ("Fp1", "Fp2"), ("AF3", "AF4"), ("F5", "F6"), ("FC5", "FC6"),
    ("C5", "C6"), ("CP5", "CP6"), ("P5", "P6"), ("PO5", "PO6"),
    ("O1", "O2"),
)


class RGNNAdapter(ModelAdapter):
    name = "rgnn"
    model_name = "RGNN backbone (without NodeDAT/EmotionDL)"
    official_reference = (
        "Zhong et al. 2019 (RGNN, IEEE T-AFFC); 本项目限定: 归纳式图骨干, "
        "节点特征=五频带 log-power surrogate (非官方 DE+LDS, C 类明示); "
        "邻接=距离初始化+论文全局对"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "shape": "[B, 32, 5]", "features": "五频带 log-power (delta/theta/alpha/beta/gamma)",
        "note": "特征提取无参数、无训练集统计; 邻接由数据文件提供",
    }

    def __init__(self, channel_order: Optional[Sequence[str]] = None,
                 distance_adjacency: Optional[np.ndarray] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("RGNNAdapter 需要 channel_order")
        self.channel_order = list(channel_order)
        self.n_chans = len(self.channel_order)
        self._adjacency_raw = None if distance_adjacency is None else np.asarray(distance_adjacency, dtype=np.float32)
        self.applied_pairs: tuple = ()

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            n_trials += int(X.shape[0])
            n_subjects += 1
        if self._adjacency_raw is not None:
            if self._adjacency_raw.shape != (self.n_chans, self.n_chans):
                raise RuntimeError(
                    f"RGNN 邻接形状 {self._adjacency_raw.shape} != ({self.n_chans},{self.n_chans})"
                )
            if not np.allclose(self._adjacency_raw, self._adjacency_raw.T, atol=1e-7, rtol=0):
                raise RuntimeError("RGNN 距离邻接必须对称")
        stats = AdapterStats(
            source_split="train",
            split_hash=split_hash,
            n_subjects=n_subjects,
            n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={},  # 特征无参数: 无训练集统计量
        )
        self.set_stats(stats)
        return stats

    def global_pairs(self):
        """论文全局对 (完整定义, 供审计报告)。"""
        return RGNN_GLOBAL_PAIRS

    def adjacency(self) -> np.ndarray:
        """距离邻接 + 论文全局对 (减 1), 供模型构造; 无随机、无训练集依赖。"""
        if self._adjacency_raw is None:
            raise RuntimeError("RGNNAdapter 需要 distance_adjacency (数据文件 electrode_adjacency.npy)")
        adj = self._adjacency_raw.copy()
        lookup = {name.casefold(): i for i, name in enumerate(self.channel_order)}
        if len(lookup) != self.n_chans:
            raise RuntimeError("RGNN channel_order 含重复通道名")
        applied = []
        for left, right in RGNN_GLOBAL_PAIRS:
            if left.casefold() in lookup and right.casefold() in lookup:
                i, j = lookup[left.casefold()], lookup[right.casefold()]
                adjusted = adj[i, j] - 1.0
                adj[i, j] = adjusted
                adj[j, i] = adjusted
                applied.append((self.channel_order[i], self.channel_order[j]))
        self.applied_pairs = tuple(applied)
        return adj

    def transform(self, x):
        """[B,32,750] -> [B,32,5] 五频带 log-power。确定性, 无训练集统计。"""
        if self._stats is None:
            raise RuntimeError("RGNNAdapter.transform 前必须先 fit (统一流程: 契约校验)")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        if xt.ndim != 3 or xt.shape[1:] != (self.n_chans, 750):
            raise ValueError(f"RGNN transform 期望 [B,{self.n_chans},750], 得到 {list(xt.shape)}")
        spectrum = torch.fft.rfft(xt.float(), dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        freqs = torch.fft.rfftfreq(750, d=1.0 / 250.0)
        bands = []
        for _name, low, high in RGNN_BANDS:
            mask = (freqs >= low) & (freqs < high)
            bands.append(power[..., mask].mean(-1).clamp_min(1e-8).log())
        out = torch.stack(bands, dim=-1)
        return out if is_torch else out.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {
            "none": IdentityAdapter(),
            "naive": NaiveRGNNAdapter(self.channel_order),
            "official": self,
        }


class NaiveRGNNAdapter(RGNNAdapter):
    """消融态 naive: 线性功率 (不取 log, 不 clamp) -> 数值尺度错误。"""
    name = "rgnn_naive"
    model_name = "RGNN backbone (naive linear power)"
    official_reference = "消融对照: 线性功率替代 log-power (数值尺度错误示范)"

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
        if x.ndim != 3 or x.shape[1:] != (self.n_chans, 750):
            raise ValueError(f"NaiveRGNNAdapter 期望 [B,{self.n_chans},750]")
        import torch
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
        spectrum = torch.fft.rfft(xt.float(), dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        freqs = torch.fft.rfftfreq(750, d=1.0 / 250.0)
        bands = []
        for _name, low, high in RGNN_BANDS:
            mask = (freqs >= low) & (freqs < high)
            bands.append(power[..., mask].mean(-1))  # 无 log, 无 clamp
        return torch.stack(bands, dim=-1).numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
