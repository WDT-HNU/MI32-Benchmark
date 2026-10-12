"""LaBraM 数据适配器。

官方输入条件: Jiang et al. 2024 LaBraM 作者预处理要求物理 microvolts (µV),
采样率 200 Hz, 通道名 -> 官方 position embedding 映射, 每通道 3 个 200 点 patch。
MI-32 适配 (B 类):
  1. 单位 V -> µV (x1e6);
  2. 250 -> 200 Hz 抗混叠 sinc 重采样。
通道 embedding 映射/重排与 braindecode backbone 耦合, 保留在模型层。
消融:
  - none:    无适配 (V@250Hz 直接输入) -> 单位与频率契约错误;
  - naive:   线性插值重采样 (无抗混叠) -> 频率混叠;
  - official: 本 adapter (sinc 抗混叠 + µV)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ._resample import apply_sinc_resample, linear_resample, sinc_resample_kernel
from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


class LaBraMAdapter(ModelAdapter):
    name = "labram"
    model_name = "LaBraM"
    official_reference = (
        "Jiang et al. 2024 (LaBraM, ICLR); 作者预处理: 物理 µV + 200Hz + 通道名映射 + "
        "3x200 patch (braindecode 1.7.0 封印); MI-32 适配: V->µV x1e6 + 250->200Hz sinc 抗混叠 (B 类)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "channels": 32, "samples": 600, "fs": 200, "unit": "uV",
        "note": "通道重排/embedding 映射保留在模型层 (与 braindecode backbone 耦合)",
    }

    UNIT_SCALE = np.float32(1_000_000.0)

    def __init__(self, channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("LaBraMAdapter 需要 channel_order")
        self.channel_order = list(channel_order)
        if len(self.channel_order) != 32:
            raise ValueError(f"LaBraM MI-32 要求 32 通道, 得到 {len(self.channel_order)}")
        self._kernel, self._width, self._old_red, self._new_red = sinc_resample_kernel(250, 200)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            if X.shape[1:] != (32, 750):
                raise RuntimeError(f"LaBraM 期望 [B,32,750], 得到 {list(X.shape)}")
            n_trials += int(X.shape[0])
            n_subjects += 1
        stats = AdapterStats(
            source_split="train", split_hash=split_hash,
            n_subjects=n_subjects, n_trials=n_trials,
            fit_timestamp=utcnow_iso(),
            values={},  # 物理 µV: 无训练集统计 (作者预处理无 z-score)
        )
        self.set_stats(stats)
        return stats

    def transform(self, x):
        if self._stats is None:
            raise RuntimeError("LaBraMAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        if xt.ndim != 3 or xt.shape[1:] != (32, 750):
            raise ValueError(f"LaBraM 期望 [B,32,750], 得到 {list(xt.shape)}")
        xt = xt * self.UNIT_SCALE
        kernel = self._kernel.to(xt.device)
        out = apply_sinc_resample(
            xt.float(), kernel.float(), self._width,
            self._old_red, self._new_red,
        )
        if out.shape[-1] != 600:
            raise RuntimeError(f"LaBraM 重采样必须得到 600 点, 得到 {out.shape[-1]}")
        return out if is_torch else out.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveLaBraMAdapter(self.channel_order), "official": self}


class NaiveLaBraMAdapter(LaBraMAdapter):
    """消融态 naive: 线性插值重采样 (无抗混叠) + 单位转换。"""
    name = "labram_naive"
    model_name = "LaBraM (naive linear resample)"
    official_reference = "消融对照: 线性重采样替代抗混叠 sinc (频率混叠错误示范)"

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("NaiveLaBraMAdapter.transform 前必须先 fit")
        x = np.asarray(x, dtype=np.float32) * float(self.UNIT_SCALE)
        return linear_resample(x, 250, 200)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
