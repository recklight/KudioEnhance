# -*- coding: utf-8 -*-
"""Command line interface.

    kudio-enhance models                          # list architectures
    kudio-enhance init config.yaml                # write a starter config
    kudio-enhance synth    -c config.yaml -n exp1 # stage 1: mix the dataset
    kudio-enhance train    -c config.yaml -n exp1 # stage 2: fit the model
    kudio-enhance evaluate -c config.yaml -n exp1 # stage 3: score the test split
    kudio-enhance run      -c config.yaml -n exp1 # all three
    kudio-enhance denoise runs/exp1 noisy.wav out.wav
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kudio_enhance._version import __version__

log = logging.getLogger("kudio_enhance")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kudio-enhance",
        description="Train and run learned speech-enhancement models.")
    parser.add_argument("--version", action="version",
                        version=f"kudio-enhance {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list the available architectures")

    init = sub.add_parser("init", help="write a starter config file")
    init.add_argument("path", nargs="?", default="config.yaml")

    for stage, help_text in (("synth", "mix the dataset and freeze the splits"),
                             ("train", "fit the model"),
                             ("evaluate", "score the test split"),
                             ("run", "synth + train + evaluate")):
        p = sub.add_parser(stage, help=help_text)
        p.add_argument("-c", "--config", required=True, help="YAML config")
        p.add_argument("-n", "--name", required=True, help="experiment name")
        if stage in ("train", "run"):
            p.add_argument("--epochs", type=int,
                           help="override train.epochs (use 1 for a smoke run)")
        if stage in ("evaluate", "run"):
            p.add_argument("--full", action="store_true",
                           help="also compute PESQ/STOI/SDR (needs kudio[eval])")

    denoise = sub.add_parser("denoise", help="enhance a file with a trained run")
    denoise.add_argument("run_dir", help="e.g. runs/exp1")
    denoise.add_argument("input")
    denoise.add_argument("output")
    return parser


def _load(args):
    from kudio_enhance.config import Config
    return Config.from_yaml(args.config)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    if args.command == "models":
        from kudio_enhance.models import list_models
        for name, description in sorted(list_models().items()):
            print(f"{name:10s}  {description}")
        return 0

    if args.command == "init":
        from kudio_enhance.config import Config
        path = Path(args.path)
        if path.exists():
            print(f"{path} already exists", file=sys.stderr)
            return 1
        Config().to_yaml(path)
        print(f"wrote {path}")
        return 0

    if args.command == "denoise":
        from kudio_enhance.inference import Enhancer
        enhancer = Enhancer.load(args.run_dir)
        enhancer.enhance_file(args.input, args.output)
        print(f"wrote {args.output}")
        return 0

    from kudio_enhance import pipeline
    cfg = _load(args)

    if args.command == "synth":
        pairs = pipeline.synthesize(cfg, args.name)
        print(f"{len(pairs)} pair(s) synthesized")
        return 0

    if args.command == "train":
        pipeline.train(cfg, args.name, epochs=args.epochs)
        print(f"model -> {cfg.model_path(args.name)}")
        return 0

    if args.command == "evaluate":
        _, summary = pipeline.evaluate(cfg, args.name, optional=args.full)
        _print_summary(summary)
        print(f"report -> {cfg.report_path(args.name)}")
        return 0

    summary = pipeline.run(cfg, args.name, epochs=args.epochs,
                           optional=args.full)
    _print_summary(summary)
    return 0


def _print_summary(summary) -> None:
    for metric in ("si_sdr", "snr", "seg_snr", "pesq", "stoi", "sdr"):
        noisy, enhanced = summary.get(f"noisy_{metric}"), summary.get(f"enhanced_{metric}")
        if noisy is None or enhanced is None:
            continue
        print(f"{metric:8s}  {noisy:8.3f} -> {enhanced:8.3f}  "
              f"({enhanced - noisy:+.3f})")


if __name__ == "__main__":
    sys.exit(main())
