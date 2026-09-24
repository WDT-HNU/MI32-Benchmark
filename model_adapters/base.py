"""Model Input Adapter 抽象基类。

本包不是 Dataset Adapter。Dataset Adapter 只把原始数据翻译为原生实验表示，
不得重采样、插值、归一化或折叠标签；本包位于公共 harmonization 之后，
只负责把已冻结的 MI-32 表示转换成各模型的官方输入合同。

设计原则 (2026-09-02 重构确认):
1. 模型本体严格复现官方; 项目贡献集中在 adapter 层。
2. adapter 只允许两类操作:
   - 格式转换 (通道/重采样/单位/patch): 参数固定, 有官方依据;
   - 标准化: 统计量只从训练受试者拟合。
3. 禁止 (gate 断言, 违反即 fail):
   - 逐 trial z-score / 全数据集统计 / 标签或样本操作 / 选择性丢弃困难样本。
4. 每个 adapter 必须能回答「有用性」: 去适配即崩或退化, 正确适配恢复官方性能。
5. 每个 adapter 的 manifest 自证清白: 依据引用 + 统计来源 + 源码 hash。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AdapterStats:
    """训练集拟合的统计量 (adapter 唯一允许的学习来源)。"""
    source_split: str          # 必须 == "train"
    split_hash: str            # 训练受试者划分的 SHA-256 (防测试集混入)
    n_subjects: int
    n_trials: int
    fit_timestamp: str
    values: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_split": self.source_split,
            "split_hash": self.split_hash,
            "n_subjects": self.n_subjects,
            "n_trials": self.n_trials,
            "fit_timestamp": self.fit_timestamp,
            "values": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.values.items()},
        }

    def validate(self) -> None:
        if self.source_split != "train":
            raise RuntimeError(f"AdapterStats.source_split 必须为 'train', 得到 {self.source_split!r}")
        if not self.split_hash or len(self.split_hash) != 64:
            raise RuntimeError("AdapterStats.split_hash 必须是训练划分的 SHA-256")
        if self.n_subjects <= 0 or self.n_trials <= 0:
            raise RuntimeError("AdapterStats 必须有正数的受试者与 trial 计数")
        for k, v in self.values.items():
            # 只校验数值统计字段; 元数据 (通道名/索引等) 自动跳过。
            try:
                arr = np.asarray(v, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"AdapterStats 含非有限值: {k}")
            # 近零检查只对明确的统计量键 (mean/std/scale) 生效, 避免误伤索引类元数据。
            if k in ("mean", "std", "unit_scale") and np.any(np.abs(arr) < 1e-12):
                raise RuntimeError(f"AdapterStats 含近零值 (数值退化): {k}")


class ModelAdapter(ABC):
    """统一的模型输入适配协议。

    子类必须声明:
      name                短名, 如 'eegnet'
      model_name          模型全名, 如 'EEGNet-8,2'
      official_reference  论文引用 + 官方仓库 + 提交 (字符串)
      input_contract      入口契约 dict (channels/samples/fs/unit)
      output_contract     出口契约 dict (模型官方输入格式描述)
    """

    name: str = ""
    model_name: str = ""
    official_reference: str = ""
    input_contract: Dict[str, Any] = {}
    output_contract: Dict[str, Any] = {}

    def __init__(self) -> None:
        if not self.name or not self.model_name or not self.official_reference:
            raise ValueError(f"Adapter {type(self).__name__} 必须声明 name/model_name/official_reference")
        self._stats: Optional[AdapterStats] = None

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        """只从训练受试者拟合统计量。train_subjects 为可迭代的 (subject_id, X, y) 流。

        禁止: 读取任何验证/测试受试者; 禁止逐 trial 统计泄漏。
        """

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """确定性转换 [B,32,750]@250Hz/V -> 模型官方输入格式。无随机性。"""

    # ------------------------------------------------------------------ #
    # 消融三态 (有用性证明)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def ablation_states(self) -> Dict[str, "ModelAdapter"]:
        """返回 {'none': 无适配, 'naive': 错误适配, 'official': 正确适配(self)}。"""

    # ------------------------------------------------------------------ #
    # 公共能力
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> Optional[AdapterStats]:
        return self._stats

    def set_stats(self, stats: AdapterStats) -> None:
        stats.validate()
        self._stats = stats

    def source_hash(self) -> str:
        """本 adapter 源码 (含依赖的模块级常量) 的 SHA-256, 用于封印。"""
        src = inspect.getsource(type(self))
        mod = inspect.getmodule(type(self))
        extra = getattr(mod, "_SEAL_EXTRA_SOURCE", "") if mod is not None else ""
        return hashlib.sha256((src + extra).encode("utf-8")).hexdigest()

    def manifest(self) -> Dict[str, Any]:
        """自证清白: 依据 + 契约 + 统计来源 + 源码 hash + 参数。"""
        m = {
            "adapter_kind": "model_input_adapter",
            "adapter": self.name,
            "model": self.model_name,
            "official_reference": self.official_reference,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "source_sha256": self.source_hash(),
            "stats": self._stats.to_json() if self._stats is not None else None,
            "ablation_states": sorted(self.ablation_states().keys()),
        }
        return m

    def save_manifest(self, out_dir) -> Path:
        out = Path(out_dir) / f"adapter_manifest_{self.name}.json"
        out.write_text(json.dumps(self.manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out


Adapter = ModelAdapter
"""Backward-compatible symbol; new code should use ``ModelAdapter``."""


class IdentityAdapter(ModelAdapter):
    """消融态 none: 完全不做适配 (直接喂 [B,32,750]V)。预期: 模型无法运行或严重退化。"""
    name = "identity"
    model_name = "identity (no adapter)"
    official_reference = "消融对照: 无适配"
    input_contract = {}
    output_contract = {}

    def fit(self, train_subjects, split_hash: str) -> AdapterStats:
        raise RuntimeError("IdentityAdapter 无统计量可拟合 (消融态 none)")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def ablation_states(self) -> Dict[str, "ModelAdapter"]:
        return {"none": self}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
