"""Veri kümelerini indirir ve yerleşimi doğrular.

    python scripts/download_data.py --dataset nasa
    python scripts/download_data.py --dataset phm2010
    python scripts/download_data.py --verify
"""

from __future__ import annotations

import argparse
import sys

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import download


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["nasa", "phm2010", "all"],
        help="indirilecek veri kümesi",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="indirmeden yalnızca yerleşimi doğrula",
    )
    parser.add_argument("--force", action="store_true", help="mevcut olsa da yeniden indir")
    parser.add_argument("--config", default=None, help="yapılandırma dosyası yolu")
    args = parser.parse_args(argv)

    if not args.dataset and not args.verify:
        parser.error("--dataset veya --verify vermelisiniz")

    config = load_config(args.config)
    phm_root = config.path("phm2010.root")
    nasa_root = config.path("nasa.root")

    if args.dataset in {"nasa", "all"}:
        download.download_nasa(nasa_root, force=args.force)

    if args.dataset in {"phm2010", "all"}:
        print(download.phm2010_instructions(phm_root))

    if args.verify:
        print()
        ok = download.verify(phm_root, nasa_root)
        print("\nSonuç:", "hazır" if ok else "eksik var")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
