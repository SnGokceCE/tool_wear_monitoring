"""Naif tabanı leave-one-cutter-out protokolüyle çalıştırır.

Bu betik sinyal dosyalarına ihtiyaç duymaz - yalnızca aşınma etiketlerini
okur. Dolayısıyla PHM 2010 arşivinin tamamı inmeden, sadece ``c*_wear.csv``
dosyaları yerindeyse bile çalışır.

Ürettiği sayı, bundan sonraki her modelin geçmek zorunda olduğu çizgidir.

    python scripts/run_naive_baseline.py
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import PHM2010
from tcm.evaluation import describe, leave_one_cutter_out, summarise
from tcm.models import NaiveWearBaseline


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="yapılandırma dosyası yolu")
    parser.add_argument("--save", action="store_true", help="sonucu reports/ altına yaz")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    wear_limit = float(config.get("evaluation.wear_limit_um", 150))

    try:
        dataset = PHM2010(
            config.path("phm2010.root"),
            wear_aggregation=config.get("phm2010.wear_aggregation", "max"),
        )
    except FileNotFoundError as error:
        print(error)
        return 1

    cutters = dataset.labelled_cutters()
    if len(cutters) < 2:
        print(
            f"Etiketli kesici sayısı yetersiz: {cutters}\n"
            "Yerleşimi doğrulamak için: python scripts/download_data.py --verify"
        )
        return 1

    wear = {cutter: dataset.wear(cutter) for cutter in cutters}

    splits = leave_one_cutter_out(cutters)
    print(f"Etiketli kesiciler: {', '.join(cutters)}")
    print(f"Aşınma sınırı     : {wear_limit:.0f} um")
    print(f"Bölme             : leave-one-cutter-out\n{describe(splits)}\n")

    rows = []
    for split in splits:
        train = pd.concat([wear[c] for c in split.train], ignore_index=True)
        test = wear[split.test[0]]

        predictions = NaiveWearBaseline().fit_predict(
            train["cut"], train["vb_um"], test["cut"]
        )

        scores = summarise(test["vb_um"], predictions, wear_limit)
        scores["fold"] = split.name
        scores["n_test_cuts"] = len(test)
        scores["flute_spread_um"] = float(test["flute_spread_um"].mean())
        rows.append(scores)

    results = pd.DataFrame(rows)[
        ["fold", "n_test_cuts", "mae_um", "rmse_um", "crossing_delay_cuts", "flute_spread_um"]
    ]

    print("Katlama sonuçları")
    print(results.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    mean_mae = results["mae_um"].mean()
    mean_delay = results["crossing_delay_cuts"].mean()
    mean_spread = results["flute_spread_um"].mean()

    print(f"\nOrtalama MAE          : {mean_mae:.2f} um")
    print(f"Ortalama gecikme      : {mean_delay:+.2f} geçiş")
    print(f"Ağızlar arası saçılım : {mean_spread:.2f} um  (ölçüm taban gürültüsü)")

    target_mae = float(config.get("acceptance.max_mae_um", 15.0))
    print(
        f"\nHedef MAE < {target_mae:.0f} um. Naif taban {mean_mae:.2f} um -> "
        f"modelin geçmesi gereken çizgi bu."
    )
    if mean_mae < mean_spread:
        print(
            "[uyarı] Naif tabanın hatası ölçüm gürültüsünün altında. "
            "Bu veri setinde iyileştirme payı dar; sonuçları buna göre yorumlayın."
        )

    if args.save:
        target = config.path("paths.reports") / "naive_baseline.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(target, index=False)
        print(f"\nKaydedildi: {target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
