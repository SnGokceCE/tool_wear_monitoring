"""Geçiş başına öznitelik tablosunu üretir ve önbelleğe yazar.

    python scripts/build_features.py            # önbellek varsa kullanır
    python scripts/build_features.py --rebuild  # yeniden üretir
    python scripts/build_features.py --limit 20 # hızlı deneme
"""

from __future__ import annotations

import argparse
import sys

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import PHM2010
from tcm.features.build import load_or_build


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true", help="önbelleği yok say")
    parser.add_argument("--limit", type=int, default=None, help="kesici başına geçiş sınırı")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    try:
        dataset = PHM2010(
            config.path("phm2010.root"),
            wear_aggregation=config.get("phm2010.wear_aggregation", "max"),
        )
    except FileNotFoundError as error:
        print(error)
        return 1

    cutters = dataset.labelled_cutters()
    cache = config.path("paths.data_processed") / "phm_cut_features.csv"

    features = load_or_build(
        cache,
        dataset,
        cutters,
        sampling_rate_hz=float(config.get("phm2010.sampling_rate_hz")),
        rpm=float(config.get("phm2010.spindle_rpm")),
        max_order=int(config.get("transfer.max_order", 8)),
        rebuild=args.rebuild,
        limit=args.limit,
    )

    print(f"\nSatır: {len(features)}  Sütun: {features.shape[1]}")
    print(features.groupby("cutter").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
