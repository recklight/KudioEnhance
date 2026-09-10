# -*- coding: utf-8 -*-
"""kudio-enhance — learned speech enhancement on top of the `kudio` toolkit.

Three stages, one config file:

    synthesize  ->  train  ->  evaluate

>>> from kudio_enhance import Config, pipeline
>>> cfg = Config.from_yaml('config.yaml')
>>> summary = pipeline.run(cfg, 'exp1')

TensorFlow is only imported when a model is actually built, so `Config`,
features and dataset tooling stay usable without it.
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
# training curves: importable without TensorFlow, so a finished run can be
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
    "save_history",
    "load_history",
    "summarise_history",
]
