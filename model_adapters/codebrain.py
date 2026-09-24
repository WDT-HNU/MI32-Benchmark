"""CodeBrain 数据适配器。

官方输入条件: Ma et al. 2024 CodeBrain (jingyingma01/CodeBrain): 直接 32 通道,
200Hz, 3 个 200 点 patch, flatten-all-patches MLP (19200->600->200->3)。
MI-32 适配 (B 类): V->(µV/100) (x1e4, 对齐官方下游 loader) + 250 -> 200 Hz
抗混叠 sinc 重采样 + unflatten(3,200)。

单位修正 (2026-08-27): 原实现未做 V->µV 转换, 喂入 V 尺度数据导致 patch embedding
官方 CodeBrain 数据先读取微伏量级数组，再在下游 dataset 中除以 100。
因此冻结的伏特输入必须乘 1e6/100 = 1e4，不能只做 V->µV。
消融:
  - none:    无适配 -> shape 契约错误;
  - naive:   线性插值重采样 -> 频率混叠;
  - official: 本 adapter (V->µV/100 x1e4 + sinc 抗混叠)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ._resample import apply_sinc_resample, linear_resample, sinc_resample_kernel
from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


class CodeBrainAdapter(ModelAdapter):
    name = "codebrain"
    model_name = "CodeBrain"
    official_reference = (
        "Ma et al. 2024 (CodeBrain, jingyingma01/CodeBrain); 官方: 直接32通道/200Hz/"
        "3x200 patch/flatten-all-patches MLP; MI-32 适配: V->µV/100 x1e4 + 250->200Hz sinc 抗混叠 + unflatten (B 类)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {"channels": 32, "patches": 3, "samples_per_patch": 200, "fs": 200, "unit": "uV/100", "unit_scale_from_volts": 10_000.0}
    UNIT_SCALE = 10_000.0  # V -> µV -> official downstream /100

    def __init__(self, channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("CodeBrainAdapter 需要 channel_order")
        self.channel_order = list(channel_order)
        if len(self.channel_order) != 32:
            raise ValueError("CodeBrain MI-32 要求 32 通道")
        self._kernel, self._width, self._old_red, self._new_red = sinc_resample_kernel(250, 200)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            if X.shape[1:] != (32, 750):
                raise RuntimeError(f"CodeBrain 期望 [B,32,750], 得到 {list(X.shape)}")
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
            raise RuntimeError("CodeBrainAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        if xt.ndim != 3 or xt.shape[1:] != (32, 750):
            raise ValueError(f"CodeBrain 期望 [B,32,750], 得到 {list(xt.shape)}")
        kernel = self._kernel.to(xt.device)
        xt = xt * self.UNIT_SCALE  # V -> µV/100: official downstream loader contract
        out = apply_sinc_resample(
            xt.float(), kernel.float(), self._width,
            self._old_red, self._new_red,
        )
        if out.shape[-1] != 600:
            raise RuntimeError(f"CodeBrain 重采样必须得到 600 点, 得到 {out.shape[-1]}")
        out = out.unflatten(-1, (3, 200))
        return out if is_torch else out.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveCodeBrainAdapter(self.channel_order), "official": self}


class NaiveCodeBrainAdapter(CodeBrainAdapter):
    """消融态 naive: 仅用线性插值替代抗混叠重采样。

    单位合同必须与 official 态相同，否则消融会同时改变单位和重采样，
    无法把差异归因于重采样方法。
    """
    name = "codebrain_naive"
    model_name = "CodeBrain (naive linear resample)"
    official_reference = "消融对照: 线性重采样替代抗混叠 sinc (频率混叠错误示范)"

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("NaiveCodeBrainAdapter.transform 前必须先 fit")
        x = np.asarray(x, dtype=np.float32) * self.UNIT_SCALE
        out = linear_resample(x, 250, 200)
        return out.reshape(out.shape[0], 32, 3, 200)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
