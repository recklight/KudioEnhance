# -*- coding: utf-8 -*-
"""The files a waveform-path run keeps current while it trains.

A training run on the CPU takes hours to days, and whoever checks on it
should not have to attach to its terminal or parse its log. Every epoch the
trainer rewrites ``progress.json``, one small file that says how far the
run is, what its losses were, which epoch is best so far and how long the
rest should take, and ``history.json``, the curves.

Both are written beside their destination and moved into place, so a reader
sees the previous epoch's file or this one's, never half of one. On Windows
the move fails while another process has the destination open, so it is
retried for a moment; a progress file that still cannot be replaced is
skipped with a warning; it is not worth stopping a training run over.

Nothing here imports torch: reading a run's progress needs only the file.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

__all__ = ["PROGRESS_FILE", "RESUME_FILE", "replace_file", "write_json",
           "read_progress", "remaining_seconds", "timestamp"]

#: What the trainer writes every epoch, in the run directory.
PROGRESS_FILE = "progress.json"
#: Everything a resumed run needs to carry on where this one left off. See
#: :func:`kudioenhance.checkpoint.save_training_state`.
RESUME_FILE = "resume.pt"

#: How often, and how far apart in seconds, a move onto a file another
#: process has open is tried again before giving up.
REPLACE_TRIES = 20
REPLACE_WAIT = 0.05
#: The last this many epochs time the estimate of what is left.
RECENT_EPOCHS = 5


def replace_file(partial, path) -> None:
    """Move *partial* onto *path*, retrying while *path* is held open.

    ``os.replace`` is atomic, but on Windows it is refused while any process
    has the destination open, which a reader polling the file does for a
    millisecond at a time.

    :raises PermissionError: when the move is still refused after
        :data:`REPLACE_TRIES` tries.
    """
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(partial, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT)


def write_json(path, payload: Mapping[str, Any], *,
               required: bool = True) -> Optional[Path]:
    """Write *payload* to *path* as JSON, all at once.

    With ``required=False`` a file that cannot be written, or a destination
    that stays locked, is logged and left as it was, and ``None`` comes
    back. A ``nan`` is written as ``NaN``, as :func:`json.dump` does, which
    Python reads back and a strict JSON reader does not.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        replace_file(partial, path)
    except OSError as error:
        if required:
            raise
        log.warning("could not replace %s (%s); it still shows the previous "
                    "epoch", path, error)
        with contextlib.suppress(OSError):
            os.remove(partial)
        return None
    return path


def read_progress(path) -> Dict[str, Any]:
    """A run's ``progress.json``: the file itself or the run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / PROGRESS_FILE
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def remaining_seconds(seconds: Sequence[float], epochs_left: int) -> float:
    """How long *epochs_left* more epochs take at the pace of the latest
    :data:`RECENT_EPOCHS` of *seconds*.

    The latest rather than all of them, because the first epoch of a run
    also warms the operating system's file cache and is usually the slowest.
    """
    if epochs_left <= 0 or not seconds:
        return 0.0
    recent = list(seconds)[-RECENT_EPOCHS:]
    return epochs_left * sum(recent) / len(recent)


def timestamp() -> str:
    """Local time, to the second, with its offset from UTC."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
