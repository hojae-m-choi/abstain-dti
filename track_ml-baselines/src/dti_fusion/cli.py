"""Command line interface: ``uv run dti <command>``."""

from __future__ import annotations

import argparse

from .config import load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="YAML config file")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config field, e.g. --set drug.encoder=chemberta (repeatable)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dti", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show torch / device information")

    p = sub.add_parser("download", help="download and clean a Hugging Face dataset")
    p.add_argument("dataset")
    p.add_argument("--data-dir", default="data")

    p = sub.add_parser("featurize", help="precompute ESM-2 (and ChemBERTa) caches for a config")
    _common(p)

    p = sub.add_parser("train", help="train + evaluate one model")
    _common(p)

    p = sub.add_parser("compare", help="ChemBERTa vs Graph Transformer grid")
    _common(p)
    p.add_argument("--encoders", default="graph,chemberta-frozen")
    p.add_argument("--splits", default="random,cold_drug,cold_target")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rerun", action="store_true", help="ignore existing metrics.json")

    args = parser.parse_args(argv)

    if args.command == "info":
        from .device import describe

        print(describe())
    elif args.command == "download":
        from .data.datasets import DATASETS, load_dataset

        df = load_dataset(args.dataset, args.data_dir)
        spec = DATASETS[args.dataset]
        print(f"{args.dataset}: {len(df)} pairs, {df['drug_key'].nunique()} drugs, "
              f"{df['target_key'].nunique()} targets | label {spec.label_name} | license {spec.license}")
        print(df["y"].describe().to_string())
    elif args.command == "featurize":
        from .device import resolve_device
        from .train import prepare

        cfg = load_config(args.config, args.set)
        prepare(cfg, resolve_device(cfg.train.device))
        print("features ready")
    elif args.command == "train":
        from .train import train

        train(load_config(args.config, args.set))
    elif args.command == "compare":
        from .compare import run_grid

        cfg = load_config(args.config, args.set)
        run_grid(cfg, args.encoders.split(","), args.splits.split(","),
                 [int(s) for s in args.seeds.split(",")], skip_existing=not args.rerun)


if __name__ == "__main__":
    main()
