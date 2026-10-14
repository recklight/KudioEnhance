# -*- coding: utf-8 -*-
"""The trained weights on disk.

A checkpoint is a plain dictionary — a format tag, a version, the architecture
and target the weights belong to, and the state dict — written with
:func:`torch.save` and read back with ``weights_only=True``.

Plain, because ``torch.load`` on anything else unpickles arbitrary objects,
which means running whatever code the file names. A run directory is the kind
of thing that gets zipped up and handed around, and opening one should not be a
way to execute somebody else's code. It also means the file holds no classes:
the architecture is rebuilt from the run's ``config.yaml``, so reorganising
the model code cannot strand a trained run the way a pickled module would.

This module imports torch at the top and is only ever imported from inside the
functions that need it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

__all__ = ["FORMAT", "VERSION", "save_checkpoint", "load_checkpoint"]

#: What a kudio-enhance checkpoint says it is, so a stray ``.pt`` from some
#: other project is refused by name instead of failing as a shape mismatch.
FORMAT = "kudio-enhance/torch"
#: Bumped when the layout of the dictionary changes.
VERSION = 1


def save_checkpoint(path, state_dict: Mapping[str, torch.Tensor], *,
                    model: str, target: str) -> Path:
    """Write *state_dict* to *path*, tagged with its architecture and target.

    The tensors are copied to the CPU first, so a run trained on a GPU loads
    on a machine without one. The file is written beside its destination and
    then moved into place, so a run interrupted mid-save keeps the previous
    best epoch rather than half of the new one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": FORMAT,
        "version": VERSION,
        "model": model,
        "target": target,
        "state_dict": {key: value.detach().cpu()
                       for key, value in state_dict.items()},
    }
    partial = path.with_name(path.name + ".partial")
    torch.save(checkpoint, partial)
    os.replace(partial, path)
    return path


def load_checkpoint(path, map_location: Union[str, torch.device] = "cpu"
                    ) -> Dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Refuses a file that is not one, and one written by a newer release whose
    layout this one cannot know.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(f"{path} is not a kudio-enhance checkpoint")
    if checkpoint.get("version", 0) > VERSION:
        raise ValueError(
            f"{path} was written by a newer kudio-enhance (checkpoint version "
            f"{checkpoint['version']}, this release reads up to {VERSION}); "
            f"upgrade to load it")
    return checkpoint
