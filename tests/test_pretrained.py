# -*- coding: utf-8 -*-
"""Published weights: the entries, the cache, the download and its hash,
and loading them by name. Every URL here is a ``file://`` one."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kudioenhance import pretrained
from kudioenhance.cli import main
from kudioenhance.pretrained import PRETRAINED, Pretrained, cache_dir, fetch

SRC = Path(__file__).resolve().parent.parent / "src"
SR = 16000
#: GTCRN's STFT and the sections a run of it writes
AUDIO = {"sr": SR, "n_fft": 512, "hop_length": 256, "win_length": 512,
         "window": "sqrt_hann", "pad_mode": "reflect", "compress": None}
MODEL = {"name": "gtcrn", "target": "spectrum", "units": [16],
         "dropout": 0.0}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(url: str, sha256: str, **changes) -> Pretrained:
    fields = dict(name="gtcrn-test", model="gtcrn", url=url, sha256=sha256,
                  sr=SR, licence="CC BY 4.0",
                  data="a few seconds of noise (CC0)",
                  config={"audio": AUDIO, "model": MODEL},
                  description="for the tests")
    fields.update(changes)
    return Pretrained(**fields)


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Bytes served from a file:// URL, registered as ``gtcrn-test``, and
    the cache the tests use."""
    source = tmp_path / "server" / "weights.pt"
    source.parent.mkdir()
    source.write_bytes(b"pretend weights " * 1000)
    entry = _entry(source.as_uri(), _sha256(source.read_bytes()))
    monkeypatch.setitem(PRETRAINED, entry.name, entry)
    cache = tmp_path / "cache"
    monkeypatch.setenv("KUDIOENHANCE_CACHE", str(cache))
    return {"entry": entry, "source": source, "cache": cache}


# ------------------------------------------------------------------ entries

def test_every_published_entry_holds_together():
    """An entry is checked when it is made; this makes each one in the
    release once, and its config has to build what the entry says."""
    for name, entry in PRETRAINED.items():
        assert entry.name == name
        cfg = entry.inference_config()
        assert (cfg.model.name, cfg.audio.sr) == (entry.model, entry.sr)
        assert entry.url.startswith("https://")


def test_gtcrn_vbd16k_is_the_run_in_its_model_card():
    """The hash and the sections are those of the run in the model card,
    written out by hand from `Pretrained.from_run`; a slip in either would
    only show on someone's first download."""
    entry = PRETRAINED["gtcrn-vbd16k"]
    assert entry.sha256 == ("357f4c9da4fd09895bd28e3933aed729"
                            "dc84869104343b335d2ec990448bee99")
    assert entry.url == ("https://github.com/recklight/KudioEnhance/releases/"
                         "download/gtcrn-vbd16k/gtcrn-vbd16k.pt")
    assert (entry.model, entry.sr, entry.licence) == ("gtcrn", SR,
                                                      "CC BY 4.0")
    cfg = entry.inference_config()
    assert (cfg.audio.n_fft, cfg.audio.hop_length, cfg.audio.window,
            cfg.audio.pad_mode) == (512, 256, "sqrt_hann", "reflect")
    assert (cfg.model.target, cfg.model.units) == ("spectrum", [16])


def test_gtcrn_vbd16k_builds_the_network_its_weights_fit():
    pytest.importorskip("torch", reason="needs PyTorch", exc_type=ImportError)
    from kudioenhance.models import build_model, count_params, resolve_units

    cfg = resolve_units(PRETRAINED["gtcrn-vbd16k"].inference_config())
    assert count_params(build_model(cfg)) == 23_669


def test_every_model_card_has_its_entry_and_its_hash():
    """A card in docs/models/ is the page an entry links to: same name, the
    entry's SHA-256 and URL on it, and no placeholder left."""
    cards = sorted((SRC.parent / "docs" / "models").glob("*.md"))
    assert cards, "docs/models/ has no model cards"
    assert sorted(card.stem for card in cards) == sorted(PRETRAINED)
    for card in cards:
        entry = PRETRAINED[card.stem]
        text = card.read_text(encoding="utf-8")
        assert entry.sha256 in text, card.name
        assert entry.url in text, card.name
        assert "{URL}" not in text, card.name
        assert entry.card.endswith(f"/docs/models/{card.name}")


@pytest.mark.parametrize("changes, message", [
    ({"sha256": "abc"}, "64 lowercase hex"),
    ({"sha256": "A" * 64}, "64 lowercase hex"),
    ({"name": "GTCRN test"}, "lowercase letters"),
    ({"url": "ftp://example.org/w.pt"}, "is not a https, http, file URL"),
    ({"model": "nope"}, "no architecture called 'nope'"),
    ({"model": "crn"}, "stats.npz"),
    ({"config": {"audio": AUDIO}}, r"\['model'\]"),
    ({"sr": 8000}, "at 16000 Hz, the entry says 8000"),
    ({"config": {"audio": AUDIO, "model": dict(MODEL, name="crn")}},
     "builds a crn"),
])
def test_an_entry_that_does_not_hold_together_is_refused(changes, message):
    fields = {"url": "https://example.org/w.pt", "sha256": "0" * 64}
    fields.update(changes)
    with pytest.raises(ValueError, match=message):
        _entry(**fields)


def test_an_unknown_name_says_what_there_is(published):
    with pytest.raises(KeyError, match="gtcrn-test"):
        fetch("nothing-by-this-name")


# -------------------------------------------------------------------- cache

def test_the_cache_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("KUDIOENHANCE_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache_dir() == Path.home() / ".cache" / "kudioenhance"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "kudioenhance"
    monkeypatch.setenv("KUDIOENHANCE_CACHE", str(tmp_path / "mine"))
    assert cache_dir() == tmp_path / "mine"


def test_fetch_downloads_once_and_checks_the_hash(published):
    entry, source = published["entry"], published["source"]
    path = fetch(entry.name)
    assert path.parent == published["cache"]
    assert path.read_bytes() == source.read_bytes()
    assert entry.sha256[:12] in path.name
    # the second time it comes from the cache: the server is gone
    source.unlink()
    assert fetch(entry.name) == path
    assert sorted(p.name for p in published["cache"].iterdir()) == [path.name]


def test_a_cached_file_that_changed_is_fetched_again(published):
    entry = published["entry"]
    path = fetch(entry.name)
    path.write_bytes(b"not what was published")
    assert fetch(entry.name).read_bytes() == published["source"].read_bytes()


def test_a_download_with_another_hash_leaves_nothing_behind(published,
                                                           monkeypatch):
    entry = published["entry"]
    monkeypatch.setitem(PRETRAINED, entry.name,
                        _entry(entry.url, "1" * 64))
    with pytest.raises(ValueError, match="not the published"):
        fetch(entry.name)
    assert list(published["cache"].iterdir()) == []


def test_a_url_that_cannot_be_read_leaves_nothing_behind(published,
                                                         monkeypatch):
    entry = published["entry"]
    published["source"].unlink()
    with pytest.raises(OSError):
        fetch(entry.name)
    assert list(published["cache"].iterdir()) == []


def test_an_explicit_cache_folder_wins(published, tmp_path):
    path = fetch(published["entry"].name, tmp_path / "elsewhere")
    assert path.parent == tmp_path / "elsewhere"
    assert not published["cache"].exists()


def test_a_cache_folder_given_with_a_tilde_is_expanded(published):
    path = pretrained.cached_path(published["entry"], "~/weights")
    assert path.parent == Path.home() / "weights"


class _BreaksOff:
    """A response that hands over one block, then fails the way a body cut
    short in chunked encoding does."""

    def __init__(self, *args, **kwargs):
        self.blocks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        import http.client
        self.blocks += 1
        if self.blocks > 1:
            raise http.client.IncompleteRead(b"", 100)
        return b"the first block"


def test_a_download_that_breaks_off_is_an_oserror_and_leaves_nothing(
        published, monkeypatch):
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", _BreaksOff)
    with pytest.raises(OSError, match="broke off"):
        fetch(published["entry"].name)
    assert list(published["cache"].iterdir()) == []


def test_a_good_file_another_fetch_holds_open_is_taken(published,
                                                       monkeypatch):
    """Two processes fetching at once: the one whose move is refused finds
    the other's file in place, with the right hash, and uses it."""
    from kudioenhance import progress

    def refused(partial, path):
        Path(path).write_bytes(published["source"].read_bytes())
        raise PermissionError("held open by the other process")

    monkeypatch.setattr(progress, "replace_file", refused)
    path = fetch(published["entry"].name)
    assert path.read_bytes() == published["source"].read_bytes()
    assert [p.name for p in published["cache"].iterdir()] == [path.name]


# ---------------------------------------------------------------------- CLI

def test_the_listing_with_nothing_published(monkeypatch, capsys):
    monkeypatch.setattr(pretrained, "PRETRAINED", {})
    assert main(["pretrained"]) == 0
    assert "no pretrained weights are published" in capsys.readouterr().out


def test_the_listing_says_what_each_entry_is_and_whether_it_is_here(
        published, capsys):
    assert main(["pretrained"]) == 0
    out = capsys.readouterr().out
    assert "gtcrn-test  gtcrn, 16000 Hz  for the tests" in out
    assert "trained on a few seconds of noise (CC0); weights CC BY 4.0" in out
    assert "not downloaded" in out

    assert main(["pretrained", "gtcrn-test"]) == 0
    out = capsys.readouterr().out
    assert "gtcrn-test -> " in out and "SHA-256 checked" in out
    assert main(["pretrained"]) == 0
    assert "cached: " in capsys.readouterr().out


def test_a_download_that_fails_is_a_message_and_a_nonzero_exit(
        published, monkeypatch, capsys):
    entry = published["entry"]
    monkeypatch.setitem(PRETRAINED, entry.name, _entry(entry.url, "2" * 64))
    assert main(["pretrained", "gtcrn-test", "nothing-by-this-name"]) == 1
    err = capsys.readouterr().err
    assert "not the published" in err
    assert "no pretrained weights called 'nothing-by-this-name'" in err


def test_listing_and_downloading_do_not_import_torch(published, tmp_path):
    """Like `models`: a machine without torch can still fetch the weights
    for another one."""
    code = (
        "import sys\n"
        "from kudioenhance import pretrained\n"
        "from kudioenhance.cli import main\n"
        f"pretrained.PRETRAINED['gtcrn-test'] = pretrained.Pretrained(\n"
        f"    **{_fields(published['entry'])!r})\n"
        "assert main(['pretrained']) == 0\n"
        "assert main(['pretrained', 'gtcrn-test']) == 0\n"
        "print('torch loaded:', 'torch' in sys.modules)\n")
    env = dict(os.environ, PYTHONPATH=str(SRC),
               KUDIOENHANCE_CACHE=str(tmp_path / "fresh"))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert "torch loaded: False" in result.stdout


def _fields(entry: Pretrained) -> dict:
    from dataclasses import asdict
    return asdict(entry)


# ------------------------------------------------ loading weights by name

@pytest.fixture
def gtcrn_run(tmp_path, monkeypatch):
    """A run directory of an untrained GTCRN, published as ``gtcrn-test``
    through :meth:`Pretrained.from_run`."""
    torch = pytest.importorskip("torch", reason="needs PyTorch",
                                exc_type=ImportError)
    from kudioenhance.checkpoint import save_checkpoint
    from kudioenhance.config import Config
    from kudioenhance.models import build_model

    torch.manual_seed(5)
    cfg = Config.from_dict({"audio": AUDIO, "model": MODEL,
                            "data": {"prepared": str(tmp_path / "corpus")}})
    run = tmp_path / "runs" / "gtcrn-exp"
    run.mkdir(parents=True)
    cfg.to_yaml(run / "config.yaml")
    model = build_model(cfg)
    save_checkpoint(run / "gtcrn.pt", model.state_dict(), model="gtcrn",
                    target="spectrum")
    entry = Pretrained.from_run(run, name="gtcrn-test",
                                url=(run / "gtcrn.pt").as_uri(),
                                licence="CC BY 4.0", data="nothing")
    monkeypatch.setitem(PRETRAINED, entry.name, entry)
    monkeypatch.setenv("KUDIOENHANCE_CACHE", str(tmp_path / "cache"))
    return run


def test_an_entry_from_a_run_takes_its_sections_and_its_hash(gtcrn_run):
    entry = PRETRAINED["gtcrn-test"]
    assert entry.sha256 == pretrained.sha256_of(gtcrn_run / "gtcrn.pt")
    assert entry.config["audio"]["window"] == "sqrt_hann"
    assert entry.config["model"]["units"] == [16]
    assert set(entry.config) == {"audio", "model"}
    assert (entry.model, entry.sr) == ("gtcrn", SR)


def test_published_weights_enhance_as_the_run_does(gtcrn_run, tmp_path,
                                                   monkeypatch):
    from kudioenhance.inference import Enhancer

    noisy = np.random.default_rng(1).normal(0, 0.1, SR).astype(np.float32)
    from_run = Enhancer.load(gtcrn_run).enhance(noisy)
    by_name = Enhancer.pretrained("gtcrn-test").enhance(noisy)
    assert np.array_equal(from_run, by_name)
    # `load` takes the name too, unless a folder of that name is there
    monkeypatch.chdir(tmp_path)
    assert np.array_equal(Enhancer.load("gtcrn-test").enhance(noisy),
                          from_run)
    (tmp_path / "gtcrn-test").mkdir()
    with pytest.raises(FileNotFoundError):
        Enhancer.load("gtcrn-test")


def test_weights_that_are_not_what_the_entry_describes_are_refused(
        gtcrn_run, monkeypatch):
    from kudioenhance.checkpoint import load_checkpoint, save_checkpoint
    from kudioenhance.inference import Enhancer

    checkpoint = load_checkpoint(gtcrn_run / "gtcrn.pt")
    other = gtcrn_run.parent / "other.pt"
    save_checkpoint(other, checkpoint["state_dict"], model="gtcrn",
                    target="irm")
    entry = PRETRAINED["gtcrn-test"]
    monkeypatch.setitem(PRETRAINED, entry.name, _entry(
        other.as_uri(), pretrained.sha256_of(other), config=entry.config))
    with pytest.raises(ValueError, match="trained for the 'irm' target"):
        Enhancer.pretrained("gtcrn-test")


def test_denoise_takes_a_name(gtcrn_run, tmp_path, capsys):
    import kudio

    from kudioenhance.inference import Enhancer

    noisy = np.random.default_rng(2).normal(0, 0.1, SR // 2)
    kudio.save_wave(tmp_path / "noisy.wav", noisy.astype(np.float32), SR)
    assert main(["denoise", "gtcrn-test", str(tmp_path / "noisy.wav"),
                 str(tmp_path / "out.wav")]) == 0
    enhanced, sr = kudio.file_load(tmp_path / "out.wav")
    assert sr == SR and len(enhanced) == SR // 2
    # the run's own output, give or take writing it as 16 bits
    reference = Enhancer.load(gtcrn_run).enhance_file(tmp_path / "noisy.wav")
    assert np.max(np.abs(enhanced - reference)) < 2 / 32768


def test_the_cache_folder_given_is_where_denoise_and_load_look(
        gtcrn_run, tmp_path, monkeypatch):
    """Downloaded ahead of time into a folder of one's choosing, the
    weights load from there with the server gone."""
    import kudio

    from kudioenhance.inference import Enhancer

    elsewhere = tmp_path / "elsewhere"
    assert main(["pretrained", "gtcrn-test", "--cache", str(elsewhere)]) == 0
    noisy = np.random.default_rng(3).normal(0, 0.1, SR // 4)
    expected = Enhancer.load(gtcrn_run).enhance(noisy.astype(np.float32))
    (gtcrn_run / "gtcrn.pt").unlink()
    loaded = Enhancer.load("gtcrn-test", cache=elsewhere)
    assert np.array_equal(loaded.enhance(noisy.astype(np.float32)), expected)
    kudio.save_wave(tmp_path / "noisy.wav", noisy.astype(np.float32), SR)
    assert main(["denoise", "gtcrn-test", str(tmp_path / "noisy.wav"),
                 str(tmp_path / "out.wav"), "--cache", str(elsewhere)]) == 0
    assert not (tmp_path / "cache").exists()


def test_denoise_with_neither_a_run_nor_a_name_is_a_message(tmp_path,
                                                           capsys):
    assert main(["denoise", str(tmp_path / "missing"), "in.wav",
                 "out.wav"]) == 1
    err = capsys.readouterr().err
    assert "neither a run directory nor the name of published weights" \
        in err


def test_denoise_with_weights_that_fail_their_hash_is_a_message(
        published, monkeypatch, capsys):
    entry = published["entry"]
    monkeypatch.setitem(PRETRAINED, entry.name, _entry(entry.url, "3" * 64))
    assert main(["denoise", "gtcrn-test", "in.wav", "out.wav"]) == 1
    assert "not the published" in capsys.readouterr().err
