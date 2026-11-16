# -*- coding: utf-8 -*-
"""Published weights, and the cache they are downloaded into.

An entry names one file: a run's ``<model>.pt`` exactly as training wrote
it, the plain checkpoint dictionary of :mod:`kudioenhance.checkpoint`. It
also carries the SHA-256 the file must have, where to fetch it, and the
``audio`` and ``model`` sections of the run's config, which is all
inference needs, so the weights are the only download. Loading such a file
cannot run code, and the hash says it is the file that was published.

Only waveform-path models can be entries: a log-power model also needs the
run's normalisation statistics, a second file an entry has no place for.

The cache is ``$KUDIOENHANCE_CACHE`` when set, otherwise
``$XDG_CACHE_HOME/kudioenhance``, otherwise ``~/.cache/kudioenhance``, the
same on every platform. A cached file is hashed again every time it is
used; at a few hundred kilobytes that costs nothing, and a file that has
changed on disk is fetched again rather than loaded.

This module does not import torch.
"""
from __future__ import annotations

import hashlib
import http.client
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from kudioenhance._version import __version__
from kudioenhance.config import Config

log = logging.getLogger(__name__)

__all__ = ["Pretrained", "PRETRAINED", "cache_dir", "get", "fetch",
           "cached_path", "sha256_of", "describe"]

#: The URL schemes a download may use. ``file`` serves weights from a disk
#: or a share, and the tests.
SCHEMES = ("https", "http", "file")
#: How long a download may wait on the server, in seconds.
TIMEOUT = 60
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Pretrained:
    """One published weights file.

    :param name: what the CLI and :meth:`Enhancer.load
        <kudioenhance.inference.Enhancer.load>` call it.
    :param model: the architecture, a name from the model registry.
    :param url: where the file is fetched from.
    :param sha256: the file's SHA-256, in lowercase hex.
    :param sr: the sample rate the weights were trained at.
    :param licence: the licence the weights are published under.
    :param data: what they were trained on, and that data's licence.
    :param config: the run's ``audio`` and ``model`` sections, as
        ``config.yaml`` holds them.
    :param description: one line for the listing.
    :param card: where the model card is: the training run, its scores
        and what to cite.
    """

    name: str
    model: str
    url: str
    sha256: str
    sr: int
    licence: str
    data: str
    config: Mapping[str, Mapping[str, Any]]
    description: str = ""
    card: str = ""

    def __post_init__(self) -> None:
        from kudioenhance.models import REGISTRY

        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.name):
            raise ValueError(
                f"{self.name!r}: a name is lowercase letters, digits, '.', "
                f"'_' and '-', so that it is also a file name")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError(
                f"{self.name}: sha256 must be 64 lowercase hex digits, "
                f"got {self.sha256!r}")
        scheme = urllib.parse.urlsplit(self.url).scheme
        if scheme not in SCHEMES:
            raise ValueError(
                f"{self.name}: {self.url!r} is not a {', '.join(SCHEMES)} URL")
        spec = REGISTRY.get(self.model)
        if spec is None:
            raise ValueError(
                f"{self.name}: no architecture called {self.model!r}; "
                f"choose from {sorted(REGISTRY)}")
        if not spec.waveform:
            raise ValueError(
                f"{self.name}: {self.model} works on the standardised "
                f"log-power spectrogram, and its weights need the run's "
                f"stats.npz as well, which an entry does not carry")
        missing = {"audio", "model"} - set(self.config)
        if missing:
            raise ValueError(
                f"{self.name}: config needs the run's {sorted(missing)} "
                f"section(s)")
        cfg = self.inference_config()
        if cfg.model.name != self.model:
            raise ValueError(
                f"{self.name}: config builds a {cfg.model.name}, the entry "
                f"says {self.model}")
        if cfg.audio.sr != self.sr:
            raise ValueError(
                f"{self.name}: config is at {cfg.audio.sr} Hz, the entry "
                f"says {self.sr}")

    def inference_config(self) -> Config:
        """The run's config as far as inference reads it; the other
        sections take their defaults."""
        return Config.from_dict({"audio": dict(self.config["audio"]),
                                 "model": dict(self.config["model"])})

    @classmethod
    def from_run(cls, run_dir, *, name: str, url: str, licence: str,
                 data: str, description: str = "",
                 card: str = "") -> "Pretrained":
        """The entry for publishing a run's weights at *url*: the run's
        ``audio`` and ``model`` sections, and the SHA-256 of its
        ``<model>.pt`` as it is on disk now."""
        from kudioenhance.models import resolve_units

        run_dir = Path(run_dir)
        cfg = resolve_units(Config.from_yaml(run_dir / "config.yaml"))
        sections = cfg.to_dict()
        return cls(name=name, model=cfg.model.name, url=url,
                   sha256=sha256_of(run_dir / f"{cfg.model.name}.pt"),
                   sr=cfg.audio.sr, licence=licence, data=data,
                   config={"audio": sections["audio"],
                           "model": sections["model"]},
                   description=description, card=card)


_RELEASES = "https://github.com/recklight/KudioEnhance/releases/download"
_CARDS = "https://github.com/recklight/KudioEnhance/blob/master/docs/models"

#: The published weights by name. Each entry's URL and hash are those of the
#: release that publishes it.
PRETRAINED: Dict[str, Pretrained] = {entry.name: entry for entry in [
    # the run in the model card: 2 s crops, the epoch with the lowest
    # validation loss (51 of 61), WB-PESQ 2.77 on the VoiceBank-DEMAND test
    # set. Its own release, not a version tag's, so the file is there
    # before the version that names it.
    Pretrained(
        name="gtcrn-vbd16k",
        model="gtcrn",
        url=f"{_RELEASES}/gtcrn-vbd16k/gtcrn-vbd16k.pt",
        sha256="357f4c9da4fd09895bd28e3933aed729"
               "dc84869104343b335d2ec990448bee99",
        sr=16000,
        licence="CC BY 4.0",
        data="VoiceBank-DEMAND, 26 training speakers, 16 kHz (CC BY 4.0)",
        config={
            "audio": {"sr": 16000, "n_fft": 512, "hop_length": 256,
                      "win_length": 512, "window": "sqrt_hann",
                      "compress": None, "pad_mode": "reflect"},
            "model": {"name": "gtcrn", "target": "spectrum", "context": 2,
                      "n_frames": 64, "units": [16], "dropout": 0.0},
        },
        description="GTCRN, VoiceBank-DEMAND, trained on the CPU",
        card=f"{_CARDS}/gtcrn-vbd16k.md",
    ),
]}


def cache_dir() -> Path:
    """Where downloaded weights are kept; see the module's docstring."""
    explicit = os.environ.get("KUDIOENHANCE_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "kudioenhance"


def get(name: str) -> Pretrained:
    """The entry called *name*.

    :raises KeyError: naming the entries there are.
    """
    try:
        return PRETRAINED[name]
    except KeyError:
        known = ", ".join(sorted(PRETRAINED)) or "none in this release"
        raise KeyError(f"no pretrained weights called {name!r} "
                       f"(published: {known})") from None


def cached_path(entry: Pretrained, cache=None) -> Path:
    """Where *entry*'s file goes in the cache. The hash is part of the
    name, so a later release that publishes other weights under the same
    name downloads them beside the old file instead of over it."""
    # expanded here as well as for the environment variable: PowerShell
    # hands a command a literal ~
    root = Path(cache).expanduser() if cache is not None else cache_dir()
    return root / f"{entry.name}-{entry.sha256[:12]}.pt"


def sha256_of(path) -> str:
    """The SHA-256 of a file, in lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, cache=None) -> Path:
    """The path of *name*'s weights in the cache, downloaded first if they
    are not there, or are there with another hash.

    The download goes to a temporary file beside its destination and is
    moved into place only once its SHA-256 is the entry's, so neither an
    interrupted download nor a file that is not the published one is ever
    left where :func:`fetch` looks.

    A temporary file left by a process that was killed mid-download stays
    in the cache folder, ending in ``.partial``; it is never read, and can
    be deleted.

    :raises KeyError: for a name that is not an entry.
    :raises ValueError: when what was downloaded has another SHA-256.
    :raises OSError: when the URL cannot be read (``urllib.error.URLError``
        is one), or the server breaks off mid-response.
    """
    entry = get(name)
    path = cached_path(entry, cache)
    if path.is_file():
        if sha256_of(path) == entry.sha256:
            return path
        log.warning("%s in the cache has another SHA-256 than %s's; "
                    "downloading it again", path, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s from %s", name, entry.url)
    _download(entry.url, path, entry.sha256)
    return path


def _download(url: str, path: Path, sha256: str) -> None:
    from kudioenhance.progress import replace_file

    request = urllib.request.Request(
        url, headers={"User-Agent": f"kudioenhance/{__version__}"})
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=path.name,
                                       suffix=".partial")
    partial = Path(partial)
    digest = hashlib.sha256()
    try:
        try:
            with os.fdopen(handle, "wb") as fh, \
                    urllib.request.urlopen(request,
                                           timeout=TIMEOUT) as response:
                for block in iter(lambda: response.read(1 << 20), b""):
                    digest.update(block)
                    fh.write(block)
        except http.client.HTTPException as e:
            # a body cut short in chunked encoding, a garbled status line:
            # not OSErrors, but the same failure to whoever asked
            raise OSError(f"{url}: the download broke off ({e!r})") from e
        if digest.hexdigest() != sha256:
            raise ValueError(
                f"{url} has SHA-256 {digest.hexdigest()}, not the published "
                f"{sha256}; nothing was kept")
        try:
            replace_file(partial, path)
        except PermissionError:
            # another process fetching the same weights got there first and
            # still has the file open; what it put there is as good
            if not (path.is_file() and sha256_of(path) == sha256):
                raise
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            # a virus scanner holding the new file must not hide why the
            # download failed
            log.warning("could not remove %s: %s", partial, e)


def describe(entry: Pretrained, cache=None) -> Dict[str, Optional[str]]:
    """What the listing shows for *entry*: its fields, and the cached file
    if one with the right hash is there."""
    path = cached_path(entry, cache)
    cached = path.is_file() and sha256_of(path) == entry.sha256
    return {"name": entry.name, "model": entry.model, "sr": str(entry.sr),
            "data": entry.data, "licence": entry.licence,
            "description": entry.description, "card": entry.card,
            "cached": str(path) if cached else None}
