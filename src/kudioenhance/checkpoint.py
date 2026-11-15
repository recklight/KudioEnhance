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

__all__ = ["FORMAT", "VERSION", "TRAINING_FORMAT", "save_checkpoint",
           "load_checkpoint", "save_training_state", "load_training_state"]

#: What a kudioenhance checkpoint says it is, so a stray ``.pt`` from some
#: other project is refused by name instead of failing as a shape mismatch.
FORMAT = "kudioenhance/torch"
#: The tag checkpoints carried before 1.0, when the distribution was named
#: kudio-enhance. The dictionary inside is the same, so runs trained then
#: still load.
EARLIER_FORMATS = ("kudio-enhance/torch",)
#: Bumped when the layout of the dictionary changes.
VERSION = 1
#: What a waveform-path run's ``resume.pt`` says it is. It holds more than
#: weights (the optimiser, the schedule, every random generator, the
#: curves), all of it tensors, numbers, strings and containers of them, so
#: it loads with ``weights_only=True`` like a model does.
TRAINING_FORMAT = "kudioenhance/training-state"
TRAINING_VERSION = 1


def save_checkpoint(path, state_dict: Mapping[str, torch.Tensor], *,
                    model: str, target: str) -> Path:
    """Write *state_dict* to *path*, tagged with its architecture and target.

    The tensors are copied to the CPU first, so a run trained on a GPU loads
    on a machine without one. The file is written beside its destination and
    then moved into place, so a run interrupted mid-save keeps the previous
    best epoch rather than half of the new one. The move is retried for a
    moment while another process has the file open, which on Windows refuses
    it: a run that is still training gets loaded to see how far it is.
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
    _save_beside(checkpoint, path)
    return path


def _save_beside(payload: Mapping[str, Any], path: Path) -> None:
    """``torch.save`` *payload* next to *path*, onto the disk, then move it
    over *path*."""
    from kudioenhance.progress import replace_file

    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        # on the disk before the move: after a power cut the old file is
        # better than a new name on a truncated one
        os.fsync(fh.fileno())
    replace_file(partial, path)


def load_checkpoint(path, map_location: Union[str, torch.device] = "cpu"
                    ) -> Dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Refuses a file that is not one, and one written by a newer release whose
    layout this one cannot know.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or \
            checkpoint.get("format") not in (FORMAT,) + EARLIER_FORMATS:
        raise ValueError(f"{path} is not a kudioenhance checkpoint")
    if checkpoint.get("version", 0) > VERSION:
        raise ValueError(
            f"{path} was written by a newer kudioenhance (checkpoint version "
            f"{checkpoint['version']}, this release reads up to {VERSION}); "
            f"upgrade to load it")
    return checkpoint


def save_training_state(path, state: Mapping[str, Any]) -> Path:
    """Write a run's training state to *path*, tagged as one.

    *state* must hold only what ``torch.load(weights_only=True)`` reads
    back: tensors, numbers, strings, ``None`` and lists, tuples and dicts of
    them. :mod:`kudioenhance.waveform_train` builds it. Like
    :func:`save_checkpoint`, the file is written beside its destination and
    moved into place, so a run killed while saving keeps the previous
    epoch's state whole.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tagged = {"format": TRAINING_FORMAT, "version": TRAINING_VERSION}
    tagged.update(state)
    _save_beside(tagged, path)
    return path


def load_training_state(path, map_location: Union[str, torch.device] = "cpu"
                        ) -> Dict[str, Any]:
    """Read what :func:`save_training_state` wrote, refusing anything else."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or state.get("format") != TRAINING_FORMAT:
        raise ValueError(f"{path} is not a kudioenhance training state")
    if state.get("version", 0) > TRAINING_VERSION:
        raise ValueError(
            f"{path} was written by a newer kudioenhance (training state "
            f"version {state['version']}, this release reads up to "
            f"{TRAINING_VERSION}); upgrade to resume it")
    return state
