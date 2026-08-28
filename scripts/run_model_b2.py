"""Model B-2 - NASA + ağırlıklandırılmış PHM, ortak öznitelik uzayında.

Sorulan soru tek: PHM'i eğitime eklemek NASA'daki performansı iyileştiriyor mu?

Deney kurgusu:
  - Sınav NASA'nın kendi grupları üzerinde yapılır (B-1 ile birebir aynı).
  - PHM satırları HER katlamada eğitime eklenir, hiçbir zaman test olmaz.
  - Böylece B-1 ile B-2 doğrudan karşılaştırılabilir; tek değişen PHM'in
    varlığı ve ağırlığı.

Ağırlık: PHM 945, NASA 145 satır. Ağırlıksız bırakılırsa model neredeyse
tamamen PHM'i öğrenir ve parametre etkisini göremez - çünkü parametreler
sadece NASA'da değişiyor. Birkaç ağırlık denenir ve HEPSİ raporlanır;
en iyisini seçip onu sunmak test kümesine göre seçim yapmak olurdu.

    python scripts/run_model_b2.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.evaluation.protocol import run_grouped_cv, summarise_folds
from tcm.features.shared import (
    PARAMETER_COLUMNS,
    TIME_COLUMN,
    build_shared_table,
    sensor_columns_of,
)
from tcm.models.gbm import make_gbm_small

PROTOCOLS = {
    "vaka-dışı": "tool",
    "koşul-dışı": "condition",
    "malzeme-dışı": "material",
}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))

    processed = config.path("paths.data_processed")
    phm = pd.read_csv(processed / "phm_cut_features.csv")
    nasa = pd.read_csv(processed / "nasa_run_features.csv")

    table = build_shared_table(phm, nasa, normalise=True, n_baseline=5)
    sensors = sensor_columns_of(table)
    inputs = sensors + PARAMETER_COLUMNS + [TIME_COLUMN]

    nasa_rows = table[table["source"] == "nasa"].reset_index(drop=True)
    phm_rows = table[table["source"] == "phm"].reset_index(drop=True)

    print("ORTAK ÖZNİTELİK UZAYI")
    print("=" * 78)
    print(f"Ortak sensör özniteliği : {len(sensors)}")
    print(f"  (PHM'de 168, NASA'da 144 vardı - kuvvet ve motor akımı düştü)")
    print(f"Kesme parametreleri     : {', '.join(PARAMETER_COLUMNS)} + {TIME_COLUMN}")
    print(f"NASA satırı             : {len(nasa_rows)}")
    print(f"PHM satırı              : {len(phm_rows)}")
    print(f"Normalizasyon           : takım bazında, ilk 5 geçiş taban")

    # Ağırlık seçenekleri. 0.153 = 145/945, yani iki veri setinin toplam
    # ağırlığını eşitler. 1.0 = ağırlıksız (PHM baskın). 0.05 = PHM zayıf.
    weights = {
        "PHM yok (= B-1)": None,
        "PHM w=0.05": 0.05,
        "PHM w=0.15 (eşit toplam)": len(nasa_rows) / len(phm_rows),
        "PHM w=1.0 (ağırlıksız)": 1.0,
    }

    all_rows = []
    for label, group_column in PROTOCOLS.items():
        print("\n" + "=" * 78)
        print(f"SINAV: {label}  (gruplama: {group_column})")
        print("=" * 78)

        rows = []
        for weight_label, weight in weights.items():
            scores = _evaluate(
                nasa_rows, phm_rows, group_column, inputs, weight, limit, seed
            )
            rows.append({"model": weight_label, **scores})

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        summary["sinav"] = label
        all_rows.append(summary)

    print("\n" + "=" * 78)
    print("SONUÇ")
    print("=" * 78)
    _verdict(pd.concat(all_rows, ignore_index=True))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        pd.concat(all_rows, ignore_index=True).to_csv(
            target / "model_b2_summary.csv", index=False)
        print(f"\nKaydedildi: {target}")

    return 0


def _evaluate(nasa_rows, phm_rows, group_column, inputs, weight, limit, seed):
    """Tek bir ağırlık ayarını üç katlamalı protokolle ölçer."""
    if weight is None:
        extra = None
        data = nasa_rows.copy()
        data["w"] = 1.0
    else:
        extra = phm_rows.copy()
        extra["w"] = weight
        data = nasa_rows.copy()
        data["w"] = 1.0

    table, _ = run_grouped_cv(
        data, group_column, inputs, "vb_um",
        model_factory=lambda: make_gbm_small(random_state=seed),
        wear_limit_um=limit, sort_column="step",
        sample_weight_column="w",
        extra_train=extra,
    )
    return summarise_folds(table)


def _verdict(combined: pd.DataFrame) -> None:
    """PHM eklemek yardım etti mi?

    DİKKAT - burada kolay bir hataya düşmemek gerekiyor: her sınav için en iyi
    ağırlığı seçip "yardım etti" demek, test kümesine BAKARAK seçim yapmaktır.
    Sonuç ne çıkarsa çıksın olumlu görünür.

    Dürüst ölçüt: TEK BİR ağırlık, üç sınavın hepsinde birden taban çizgisini
    geçiyor mu? Geçmiyorsa, kazanç seçimden geliyordur - gerçek değil.
    """
    exams = list(combined["sinav"].unique())
    weights = [m for m in combined["model"].unique() if m != "PHM yok (= B-1)"]

    baseline = {
        exam: float(combined[(combined["sinav"] == exam)
                             & (combined["model"] == "PHM yok (= B-1)")]["mae_um"].iloc[0])
        for exam in exams
    }

    print(f"{'ağırlık':26s}" + "".join(f"{e:>16s}" for e in exams) + "   sonuç")
    print("-" * (26 + 16 * len(exams) + 10))

    consistent = []
    for weight in weights:
        deltas = []
        cells = []
        for exam in exams:
            value = float(combined[(combined["sinav"] == exam)
                                   & (combined["model"] == weight)]["mae_um"].iloc[0])
            delta = 100 * (value - baseline[exam]) / baseline[exam]
            deltas.append(delta)
            cells.append(f"{delta:+14.1f}%")

        wins = sum(1 for d in deltas if d < 0)
        if wins == len(exams):
            consistent.append(weight)
        print(f"{weight:26s}" + "".join(cells) + f"   {wins}/{len(exams)} sınavda iyi")

    print()
    if consistent:
        print(f"Tutarlı kazanç: {', '.join(consistent)} - üç sınavda da taban çizgisini geçti.")
    else:
        print(
            "SONUÇ: Hiçbir ağırlık üç sınavda birden taban çizgisini geçmedi.\n"
            "PHM'i eğitime eklemek NASA'daki performansı GÜVENİLİR biçimde\n"
            "iyileştirmiyor. Tek tek bakıldığında bazı hücreler iyi görünüyor,\n"
            "ama ağırlığı sonuca bakarak seçmek gerekiyor - yani gerçek bir\n"
            "kazanç değil, seçim yanlılığı.\n"
        )

    # Alarm davranışı ayrı bakılmalı: MAE iyileşse bile overshoot bozulabilir.
    print("En kötü overshoot (µm) - güvenlik açısından kritik sütun:")
    pivot = combined.pivot(index="model", columns="sinav", values="worst_overshoot_um")
    print(pivot.to_string(float_format=lambda v: f"{v:9.1f}"))
    print(
        "\n'Yardım etmedi' de geçerli bir bulgudur ve raporlanacaktır - "
        "literatürde\nçok az çalışmanın dürüstçe bildirdiği bir sonuç."
    )


if __name__ == "__main__":
    sys.exit(main())
