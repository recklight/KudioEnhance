# -*- coding: utf-8 -*-
"""progress.json and history.json: written whole, retried while a reader
holds them open, readable without torch."""
from __future__ import annotations

import logging
import re

import pytest

from kudioenhance import Config, progress


@pytest.fixture
def locked(monkeypatch):
    """Refuse the move onto *path* the first *times* tries, as Windows does
    while another process has the file open."""
    real = progress.os.replace
    refused = []

    def lock(times=float("inf"), name=progress.PROGRESS_FILE):
        def replace(src, dst):
            if str(dst).endswith(name) and len(refused) < times:
                refused.append(dst)
                raise PermissionError(13, "in use")
            real(src, dst)
        monkeypatch.setattr(progress, "REPLACE_WAIT", 0)
        monkeypatch.setattr(progress.os, "replace", replace)
        return refused

    return lock


def test_a_file_is_written_whole_and_nothing_is_left_beside_it(tmp_path):
    path = progress.write_json(tmp_path / "run" / "progress.json",
                               {"epoch": 2, "best": None})
    assert progress.read_progress(path) == {"epoch": 2, "best": None}
    assert progress.read_progress(tmp_path / "run") == {"epoch": 2,
                                                        "best": None}
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["progress.json"]


def test_a_file_held_open_is_retried(tmp_path, locked):
    refused = locked(times=2)
    progress.write_json(tmp_path / "progress.json", {"epoch": 3})
    assert progress.read_progress(tmp_path) == {"epoch": 3}
    assert len(refused) == 2


def test_a_file_that_stays_locked_keeps_the_last_epoch(tmp_path, locked,
                                                       caplog):
    progress.write_json(tmp_path / "progress.json", {"epoch": 1})
    refused = locked()
    with caplog.at_level(logging.WARNING):
        assert progress.write_json(tmp_path / "progress.json", {"epoch": 2},
                                   required=False) is None
    assert progress.read_progress(tmp_path) == {"epoch": 1}
    assert len(refused) == progress.REPLACE_TRIES
    assert "could not replace" in caplog.text
    with pytest.raises(PermissionError):
        progress.write_json(tmp_path / "progress.json", {"epoch": 2})


def test_the_estimate_follows_the_latest_epochs():
    seconds = [100.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert progress.remaining_seconds(seconds, 10) == 20.0
    assert progress.remaining_seconds(seconds[:2], 3) == 3 * 52.0
    assert progress.remaining_seconds(seconds, 0) == 0.0
    assert progress.remaining_seconds([], 5) == 0.0


def test_the_time_carries_its_offset_from_utc():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d",
                        progress.timestamp())


def test_the_run_paths_name_the_files_the_trainer_writes():
    cfg = Config.from_dict({"train": {"out_dir": "runs"}})
    assert cfg.progress_path("exp1").parts[-2:] == ("exp1", "progress.json")
    assert cfg.resume_path("exp1").parts[-2:] == ("exp1", "resume.pt")
