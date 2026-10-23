# -*- coding: utf-8 -*-
"""CLI plumbing — argument parsing and the stages it dispatches to."""
from __future__ import annotations

import pytest

from kudioenhance.cli import main


def test_models_lists_every_architecture_with_its_default_widths(capsys):
    """In the form the config takes, so a line can be pasted into one."""
    from kudioenhance.models import REGISTRY, default_units

    assert main(["models"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert sorted(line.split()[0] for line in lines) == sorted(REGISTRY)
    for line in lines:
        name = line.split()[0]
        assert line.endswith(f"units: {default_units(name)}")


def test_init_writes_a_loadable_config(tmp_path, capsys):
    from kudioenhance import Config

    path = tmp_path / "config.yaml"
    assert main(["init", str(path)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert Config.from_yaml(path).model.name == "ddae"


def test_the_starter_config_leaves_the_widths_to_the_architecture(tmp_path):
    """Switching ``model.name`` in the starter file is enough to get a
    sensible network: a 1024-wide stack written into it would follow the
    switch to a 2-D model, and be far too wide there."""
    from kudioenhance import Config
    from kudioenhance.models import resolve_units

    path = tmp_path / "config.yaml"
    main(["init", str(path)])
    text = path.read_text(encoding="utf-8")
    # yaml cannot carry a comment through a dump; the header says what the
    # one unexplained value means
    assert text.startswith("#") and "kudioenhance models" in text

    cfg = Config.from_yaml(path)
    assert cfg.model.units is None
    assert resolve_units(cfg).model.units == [1024, 1024, 1024]
    cfg.model.name = "unet"
    assert resolve_units(cfg).model.units == [16, 32, 64, 128]


def test_init_refuses_to_clobber(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("audio: {}", encoding="utf-8")

    assert main(["init", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_synth_reports_the_pair_count(tmp_path, config, capsys):
    config_path = config.to_yaml(tmp_path / "config.yaml")
    assert main(["synth", "-c", str(config_path), "-n", "exp1"]) == 0
    assert "pair(s) synthesized" in capsys.readouterr().out


def test_train_names_the_checkpoint_it_wrote(tmp_path, config, capsys,
                                              torch):
    config_path = str(config.to_yaml(tmp_path / "config.yaml"))
    assert main(["synth", "-c", config_path, "-n", "exp1"]) == 0
    assert main(["train", "-c", config_path, "-n", "exp1",
                 "--epochs", "1"]) == 0

    out = capsys.readouterr().out
    assert f"model -> {config.model_path('exp1')}" in out
    assert config.model_path("exp1").is_file()


def test_denoise_enhances_a_folder_with_a_trained_run(tmp_path, config,
                                                      corpus, capsys, torch):
    config_path = str(config.to_yaml(tmp_path / "config.yaml"))
    main(["synth", "-c", config_path, "-n", "exp1"])
    main(["train", "-c", config_path, "-n", "exp1", "--epochs", "1"])

    out_dir = tmp_path / "enhanced"
    assert main(["denoise", str(config.run_dir("exp1")), str(corpus["clean"]),
                 str(out_dir), "--recursive"]) == 0
    written = sorted(path.name for path in out_dir.rglob("*.wav"))
    assert written == sorted(path.name
                             for path in corpus["clean"].glob("*.wav"))


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])
