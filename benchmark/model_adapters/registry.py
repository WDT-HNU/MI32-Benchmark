"""Model Input Adapter 注册表: 模型名 -> adapter 工厂。"""
from .base import ModelAdapter, IdentityAdapter
from .eegnet import EEGNetAdapter, NaiveEEGNetAdapter
from .tsception import TSceptionAdapter, NaiveTSceptionAdapter
from .eegconformer import EEGConformerAdapter, NaiveConformerAdapter
from .rgnn import RGNNAdapter, NaiveRGNNAdapter
from .labram import LaBraMAdapter, NaiveLaBraMAdapter
from .eegmamba import EEGMambaAdapter, NaiveEEGMambaAdapter
from .codebrain import CodeBrainAdapter, NaiveCodeBrainAdapter
from .uni_ntfm import UniNTFMAdapter, NaiveUniNTFMAdapter

_ADAPTERS = {
    "eegnet": EEGNetAdapter,
    "tsception": TSceptionAdapter,
    "eegconformer": EEGConformerAdapter,
    "rgnn": RGNNAdapter,
    "labram": LaBraMAdapter,
    "eegmamba": EEGMambaAdapter,
    "codebrain": CodeBrainAdapter,
    "uni_ntfm": UniNTFMAdapter,
}


def get_adapter(name: str, **kwargs) -> ModelAdapter:
    if name not in _ADAPTERS:
        raise KeyError(f"未知 adapter: {name!r}; 可选: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name](**kwargs)


def list_adapters():
    return sorted(_ADAPTERS)
