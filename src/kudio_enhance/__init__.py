# -*- coding: utf-8 -*-
"""kudio-enhance — learned speech enhancement on top of the `kudio` toolkit.

Three stages, one config file:

    synthesize  ->  train  ->  evaluate

>>> from kudio_enhance import Config, Enhancer, pipeline
>>> cfg = Config.from_yaml('config.yaml')
>>> summary = pipeline.run(cfg, 'exp1')
>>> clean = Enhancer.load('runs/exp1').enhance(noisy)

The models are native PyTorch, and torch is only imported when a model is
actually built, trained or loaded, so `Config`, features and dataset tooling
stay usable without it. `Enhancer` is resolved on first use for the same
reason: importing this package never pulls in the framework.
"""
import logging as _logging

from kudio_enhance._version import __author__, __version__
from kudio_enhance.config import (
    AudioConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from kudio_enhance.data import Pair
from kudio_enhance.features import Standardizer
# training curves: importable without PyTorch, so a finished run can be
# inspected from anywhere
from kudio_enhance.train import load_history, save_history, summarise_history

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "__version__",
    "__author__",
    "Config",
    "AudioConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "Pair",
    "Standardizer",
    "Enhancer",
    "save_history",
    "load_history",
    "summarise_history",
]


def __getattr__(name: str):
    # PEP 562: `kudio_enhance.Enhancer` resolves without the inference path
    # being imported up front, so `import kudio_enhance` stays free of torch
    # however that module grows
    if name == "Enhancer":
        from kudio_enhance.inference import Enhancer
        return Enhancer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
