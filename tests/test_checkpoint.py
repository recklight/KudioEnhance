# -*- coding: utf-8 -*-
"""The run directory's weights: written, read back, and refused when wrong."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from kudioenhance import inference
from kudioenhance.features import Standardizer, spectrogram
from kudioenhance.inference import Enhancer
from kudioenhance.models import build_model, is_sequence
from kudioenhance.train import fit, seeded


def _write_run(config, tone, name: str = "exp1"):
    """A run directory made by hand: a briefly trained model, the statistics
    and the config — everything :meth:`Enhancer.load` reads — plus the model
    itself, to compare against."""
    bins = config.audio.n_bins
    rng = np.random.default_rng(0)
    if is_sequence(config.model.name):
        x = rng.standard_normal((32, config.model.n_frames, bins))
    else:
        x = rng.standard_normal((256, bins * (2 * config.model.context + 1)))
    x = x.astype(np.float32)
    y = 0.5 * x[..., :bins]

    with seeded(0):
        model = build_model(config)
    fit(model, x, y, config, config.model_path(name), epochs=2, verbose=0)
    standardizer = Standardizer().fit(spectrogram(tone, config.audio))
    standardizer.save(config.stats_path(name))
    config.to_yaml(config.run_dir(name) / "config.yaml")
    return config.run_dir(name), model, standardizer


@pytest.mark.parametrize("name", ["ddae", "blstm", "conv_ae"])
def test_a_checkpoint_round_trips_to_identical_predictions(config, tone,
                                                          torch, name):
    config.model.name = name
    config.model.units = [16]
    config.model.dropout = 0.2       # so leaving training mode matters
    run_dir, model, standardizer = _write_run(config, tone)

    loaded = Enhancer.load(run_dir)
    assert loaded.model.training is False
    original, restored = model.state_dict(), loaded.model.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[key], restored[key]) for key in original)

    expected = Enhancer(model, standardizer, config).enhance(tone)
    assert np.array_equal(loaded.enhance(tone), expected)


def test_the_checkpoint_is_a_plain_dictionary(config, tone, torch):
    """Readable with ``weights_only=True``: no pickled classes, so loading a
    run someone sent you cannot run their code, and reorganising the model
    code cannot strand a trained run."""
    from kudioenhance.checkpoint import FORMAT, VERSION

    run_dir, _, _ = _write_run(config, tone)
    checkpoint = torch.load(config.model_path("exp1"), weights_only=True)

    assert config.model_path("exp1").name == "ddae.pt"
    assert set(checkpoint) == {"format", "version", "model", "target",
                               "state_dict"}
    assert (checkpoint["format"], checkpoint["version"]) == (FORMAT, VERSION)
    assert (checkpoint["model"], checkpoint["target"]) == ("ddae", "spectrum")
    assert all(value.device.type == "cpu"
               for value in checkpoint["state_dict"].values())
    assert not list(run_dir.glob("*.partial"))


def test_frame_batches_do_not_change_the_answer(config, tone, torch,
                                                monkeypatch):
    """Frame-wise models run a bounded number of rows at a time; in
    inference mode every row is independent, so the batching is invisible."""
    run_dir, _, _ = _write_run(config, tone)
    enhancer = Enhancer.load(run_dir)
    whole = enhancer.enhance(tone)

    monkeypatch.setattr(inference, "FRAME_BATCH", 7)
    assert np.allclose(enhancer.enhance(tone), whole, atol=1e-6)


def test_a_tensorflow_run_says_to_retrain(config, tmp_path):
    """A 0.1 run directory has a ``.keras`` file and no ``.pt``. Saying "no
    model" would be true and useless; what the user needs to hear is why,
    and what to do about it."""
    run_dir = config.run_dir("old")
    run_dir.mkdir(parents=True)
    # the config exactly as 0.1 wrote it, without the keys 0.2 added
    old_config = config.to_dict()
    del old_config["train"]["device"], old_config["train"]["seed"]
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(old_config, fh)
    (run_dir / "ddae.keras").write_bytes(b"PK\x03\x04")    # a Keras 3 zip

    with pytest.raises(FileNotFoundError, match="TensorFlow") as error:
        Enhancer.load(run_dir)
    message = str(error.value)
    assert "trained by version 0.1 on TensorFlow" in message
    assert "Retrain it" in message


def test_a_run_with_no_model_at_all_says_where_it_looked(config):
    run_dir = config.run_dir("empty")
    config.to_yaml(run_dir / "config.yaml")
    with pytest.raises(FileNotFoundError, match=r"no model at .*ddae\.pt"):
        Enhancer.load(run_dir)


def test_weights_for_another_target_are_refused(config, tone, torch):
    """The config decides what the prediction *means*; weights trained for
    the other target would load cleanly and produce the wrong audio."""
    from kudioenhance.checkpoint import load_checkpoint, save_checkpoint

    run_dir, _, _ = _write_run(config, tone)
    path = config.model_path("exp1")
    save_checkpoint(path, load_checkpoint(path)["state_dict"], model="ddae",
                    target="irm")

    with pytest.raises(ValueError, match="'irm' target"):
        Enhancer.load(run_dir)


def test_a_file_that_is_not_a_checkpoint_is_refused_by_name(tmp_path, torch):
    from kudioenhance.checkpoint import load_checkpoint

    path = tmp_path / "someone_else.pt"
    torch.save({"weight": torch.zeros(3)}, path)
    with pytest.raises(ValueError, match="not a kudioenhance checkpoint"):
        load_checkpoint(path)


def test_a_run_trained_before_1_0_still_loads(config, tone, torch):
    """Checkpoints written before 1.0 carry the tag under the distribution's
    earlier name; the dictionary is otherwise the same."""
    from kudioenhance.checkpoint import load_checkpoint

    run_dir, model, standardizer = _write_run(config, tone)
    path = config.model_path("exp1")
    checkpoint = load_checkpoint(path)
    checkpoint["format"] = "kudio-enhance/torch"
    torch.save(checkpoint, path)
    assert load_checkpoint(path)["format"] == "kudio-enhance/torch"

    expected = Enhancer(model, standardizer, config).enhance(tone)
    assert np.array_equal(Enhancer.load(run_dir).enhance(tone), expected)


def test_a_checkpoint_from_a_newer_release_is_refused(tmp_path, torch):
    from kudioenhance.checkpoint import FORMAT, VERSION, load_checkpoint

    path = tmp_path / "future.pt"
    torch.save({"format": FORMAT, "version": VERSION + 1, "model": "ddae",
                "target": "spectrum", "state_dict": {}}, path)
    with pytest.raises(ValueError, match="newer kudioenhance"):
        load_checkpoint(path)
