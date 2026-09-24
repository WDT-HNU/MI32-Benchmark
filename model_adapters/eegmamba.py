"""EEGMamba 数据适配器。

官方输入条件: Wang et al. 2024 EEGMamba 官方提交: [B,32,3,200]@200Hz
(直接 32 通道, 3 个 200 点 patch; 不做 32->60, 不补零到 800)。
MI-32 适配 (B 类): V -> µV/100 (x1e4，复现官方下游 loader) +
250 -> 200 Hz 抗混叠 sinc 重采样 + unflatten(3,200)。

单位依据 (官方 commit dbc83fa072744201e8897aeb9f65007b952ad323):
  - preprocessing_physio.py 用 epochs.get_data(units='uV') 写入样本；
  - datasets/physio_dataset.py 与 datasets/bciciv2a_dataset.py 均返回 data/100。
因此从本项目冻结的 V 输入进入官方微调 backbone 时应乘 1e6/100 = 1e4。
消融:
  - none:    无适配 (V@250Hz) -> shape 契约错误;
  - naive:   线性插值重采样 -> 频率混叠;
  - official: 本 adapter (sinc 抗混叠)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ._resample import apply_sinc_resample, linear_resample, sinc_resample_kernel
from .base import ModelAdapter, AdapterStats, IdentityAdapter, utcnow_iso


class EEGMambaAdapter(ModelAdapter):
    name = "eegmamba"
    model_name = "EEGMamba"
    official_reference = (
        "Wang et al. 2024 (EEGMamba); 官方提交: 直接 [B,32,3,200]@200Hz, "
        "all_patch_reps 头; commit dbc83fa072744201e8897aeb9f65007b952ad323 官方下游 loader: "
        "µV/100; MI-32 适配: V->µV/100 x1e4 + 250->200Hz sinc 抗混叠 + unflatten (B 类)"
    )
    input_contract = {"channels": 32, "samples": 750, "fs": 250, "unit": "V"}
    output_contract = {
        "channels": 32,
        "patches": 3,
        "samples_per_patch": 200,
        "fs": 200,
        "unit": "uV/100",
        "unit_scale_from_volts": 10_000.0,
    }
    UNIT_SCALE = 10_000.0

    def __init__(self, channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        if channel_order is None:
            raise ValueError("EEGMambaAdapter 需要 channel_order (须符合官方拓扑通道序)")
        self.channel_order = list(channel_order)
        if len(self.channel_order) != 32:
            raise ValueError("EEGMamba MI-32 要求 32 通道")
        self._kernel, self._width, self._old_red, self._new_red = sinc_resample_kernel(250, 200)

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        n_trials = 0
        n_subjects = 0
        for _sid, X, _y in train_subjects:
            if X.shape[1:] != (32, 750):
                raise RuntimeError(f"EEGMamba 期望 [B,32,750], 得到 {list(X.shape)}")
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
            raise RuntimeError("EEGMambaAdapter.transform 前必须先 fit")
        is_torch = isinstance(x, torch.Tensor)
        xt = x if is_torch else torch.as_tensor(np.asarray(x, dtype=np.float32))
        if xt.ndim != 3 or xt.shape[1:] != (32, 750):
            raise ValueError(f"EEGMamba 期望 [B,32,750], 得到 {list(xt.shape)}")
        xt = xt * self.UNIT_SCALE
        kernel = self._kernel.to(xt.device)
        out = apply_sinc_resample(
            xt.float(), kernel.float(), self._width,
            self._old_red, self._new_red,
        )
        if out.shape[-1] != 600:
            raise RuntimeError(f"EEGMamba 重采样必须得到 600 点, 得到 {out.shape[-1]}")
        out = out.unflatten(-1, (3, 200))
        return out if is_torch else out.numpy().astype(np.float32)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"none": IdentityAdapter(), "naive": NaiveEEGMambaAdapter(self.channel_order), "official": self}


class NaiveEEGMambaAdapter(EEGMambaAdapter):
    """消融态 naive: 线性插值重采样 (无抗混叠)。"""
    name = "eegmamba_naive"
    model_name = "EEGMamba (naive linear resample)"
    official_reference = "消融对照: 保留官方 V->µV/100 单位，仅用线性重采样替代抗混叠 sinc"

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("NaiveEEGMambaAdapter.transform 前必须先 fit")
        x = np.asarray(x, dtype=np.float32) * self.UNIT_SCALE
        out = linear_resample(x, 250, 200)
        return out.reshape(out.shape[0], 32, 3, 200)

    def ablation_states(self) -> Dict[str, Adapter]:
        return {"naive": self}
