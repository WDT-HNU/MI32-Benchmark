"""确定性抗混叠 sinc 重采样 (从 mi3_foundation.py 抽取, 保持逐位一致)。

Hann-windowed sinc polyphase low-pass kernel, 与 torchaudio 标准构造一致,
专门固定下来以保证封印路径不随 torchaudio 版本漂移。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sinc_resample_kernel(orig_freq=250, new_freq=200, lowpass_filter_width=6, rolloff=0.99):
    """创建确定性 Hann-windowed sinc polyphase 低通核。"""
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


def apply_sinc_resample(x, kernel, width, old_reduced=5, new_reduced=4):
    """固定 polyphase 滤波, 逐 trial/channel 独立。"""
    if not x.is_floating_point():
        raise TypeError("resampling expects floating-point EEG")
    original_shape = x.shape
    length = original_shape[-1]
    flat = x.reshape(-1, length)
    padded = F.pad(flat, (width, width + old_reduced))
    filtered = F.conv1d(padded[:, None], kernel, stride=old_reduced)
    filtered = filtered.transpose(1, 2).reshape(flat.shape[0], -1)
    target_length = math.ceil(new_reduced * length / old_reduced)
    return filtered[..., :target_length].reshape(original_shape[:-1] + (target_length,))


def linear_resample(x, orig_freq=250, new_freq=200):
    """消融态 naive: 线性插值重采样 (无抗混叠, 频率混叠错误示范)。"""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    old_len = x.shape[-1]
    new_len = math.ceil(new_freq * old_len / orig_freq)
    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, new_len)
    return np.apply_along_axis(
        lambda v: np.interp(new_idx, old_idx, v), -1, x
    ).astype(np.float32)
