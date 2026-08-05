# -*- coding: utf-8 -*-
"""CLI plumbing — argument parsing and the stages it dispatches to."""
from __future__ import annotations

import pytest

from kudio_enhance.cli import main


def test_models_lists_every_architecture(capsys):
    pytest.importorskip("tensorflow")
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for name in ("ddae", "blstm", "conv_ae"):
        assert name in out


def test_init_writes_a_loadable_config(tmp_path, capsys):
    from kudio_enhance import Config

    path = tmp_path / "config.yaml"
    assert main(["init", str(path)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert Config.from_yaml(path).model.name == "ddae"


def test_init_refuses_to_clobber(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("audio: {}", encoding="utf-8")

    assert main(["init", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_synth_reports_the_pair_count(tmp_path, config, capsys):
    config_path = config.to_yaml(tmp_path / "config.yaml")
    assert main(["synth", "-c", str(config_path), "-n", "exp1"]) == 0
    assert "pair(s) synthesized" in capsys.readouterr().out


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])
