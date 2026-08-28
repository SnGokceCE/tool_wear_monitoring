"""Model B-1 - sadece NASA, kesme parametreleri girdi olarak (Faz 04b).

Model A'dan iki farkı var:

  1. GİRDİ: sensör özniteliklerine ek olarak malzeme, ilerleme ve kesme
     derinliği. Bu değişkenler ancak NASA'da değiştiği için model burada
     eğitiliyor - PHM'de hepsi sabit, sıfır bilgi taşırlar.

  2. BÖLME: üç ayrı sınav.
       vaka-dışı    : bir takımı dışarıda bırak (Model A'nın karşılığı)
       koşul-dışı   : bir kesme koşulunu tamamen dışarıda bırak
       malzeme-dışı : dökme demirde eğit, çelikte test et

     Üçüncüsü en sert olanı ve Tomtaş bağlamında en değerlisi: "hiç görmediği
     malzemede ne yapıyor" sorusu alüminyum/titanyum meselesine en yakın kanıt.

    python scripts/run_model_b1.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import NASAMilling
from tcm.evaluation.protocol import run_grouped_cv, summarise_folds
from tcm.features.build import load_or_build_nasa
from tcm.models import NaiveWearBaseline, enforce_monotonic
from tcm.models.gbm import feature_importance, make_gbm_small

# Etiket ve kimlik sütunları - öznitelik değiller.
META_COLUMNS = {"case", "run", "vb_um", "condition", "run_time"}
# Kesme parametreleri - Model B'nin yeni girdileri.
PARAMETER_COLUMNS = ["material", "feed", "doc", "rpm"]
# Kümülatif kesme süresi. Parametreler aşınma HIZINI belirler; miktarı
# bilmek için süreyi de bilmek gerekir. Sahada bilinen bir değer.
TIME_COLUMN = "cum_time"


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))
    bad_cases = tuple(config.get("nasa.known_bad_cases", []) or [])

    dataset = NASAMilling(config.path("nasa.root"))
    data = load_or_build_nasa(
        config.path("paths.data_processed") / "nasa_run_features.csv",
        dataset,
        sampling_rate_hz=float(config.get("nasa.sampling_rate_hz")),
        rpm=float(config.get("nasa.spindle_rpm")),
        max_order=int(config.get("transfer.max_order", 8)),
        drop_cases=bad_cases,
        rebuild=args.rebuild,
    )

    sensor_columns = [
        c for c in data.columns
        if c not in META_COLUMNS
        and c not in PARAMETER_COLUMNS
        and c != TIME_COLUMN
    ]
    # Tamamen NaN olan sütunlar: Nyquist üstündeki mertebeler. Atılır.
    sensor_columns = [c for c in sensor_columns if data[c].notna().any()]
    param_time = PARAMETER_COLUMNS + [TIME_COLUMN]
    all_columns = sensor_columns + param_time

    _describe_data(data, sensor_columns, limit, bad_cases)

    # rpm sabit (826) - bilgi taşımıyor ama istenen girdi tanımında var.
    # Modele veriyoruz; önem tablosunda sıfır çıkması beklenen davranış.

    protocols = {
        "vaka-dışı": "case",
        "koşul-dışı": "condition",
        "malzeme-dışı": "material",
    }

    all_rows = []
    for label, group_column in protocols.items():
        print("\n" + "=" * 78)
        print(f"SINAV: {label}  (gruplama: {group_column})")
        print("=" * 78)

        n_groups = data[group_column].nunique()
        sizes = data.groupby(group_column).size()
        print(f"{n_groups} grup | test kümesi boyutu: "
              f"min {sizes.min()}, ortanca {int(sizes.median())}, maks {sizes.max()}")

        rows = []
        rows.append({"model": "0 · naif taban",
                     **summarise_folds(_run_naive(data, group_column, limit))})

        table_sensor, _ = run_grouped_cv(
            data, group_column, sensor_columns, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        rows.append({"model": "1 · sadece sensör", **summarise_folds(table_sensor)})

        table_pt, _ = run_grouped_cv(
            data, group_column, param_time, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        rows.append({"model": "2 · parametre + süre", **summarise_folds(table_pt)})

        table_sp, _ = run_grouped_cv(
            data, group_column, sensor_columns + PARAMETER_COLUMNS, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        rows.append({"model": "3 · sensör + parametre", **summarise_folds(table_sp)})

        table_all, _ = run_grouped_cv(
            data, group_column, all_columns, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        rows.append({"model": "4 · sensör + parametre + süre", **summarise_folds(table_all)})

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        if group_column == "material":
            print("\nKatlama ayrıntısı (sensör + parametre):")
            print(table_all.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        summary["sinav"] = label
        all_rows.append(summary)

    # ---------------------------------------------------------- önemler
    print("\n" + "=" * 78)
    print("ÖZNİTELİK ÖNEMLERİ - kesme parametreleri işe yarıyor mu?")
    print("=" * 78)
    final = make_gbm_small(random_state=seed)
    final.fit(data[all_columns], data["vb_um"])
    importance = feature_importance(final, all_columns)

    print("\nİlk 12 öznitelik:")
    print(importance.head(12).to_string(index=False))

    params = importance[importance["oznitelik"].isin(param_time)]
    total = importance["onem"].sum()
    print("\nKesme parametrelerinin sıralamadaki yeri:")
    for row in params.itertuples(index=False):
        rank = int(importance.index[importance["oznitelik"] == row.oznitelik][0]) + 1
        print(f"  {row.oznitelik:10s} önem {row.onem:5d}  "
              f"({100 * row.onem / total:4.1f}%)  sıra {rank}/{len(importance)}")

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        pd.concat(all_rows, ignore_index=True).to_csv(
            target / "model_b1_summary.csv", index=False)
        importance.to_csv(target / "model_b1_importance.csv", index=False)
        print(f"\nKaydedildi: {target}")

    return 0


def _describe_data(data, sensor_columns, limit, bad_cases) -> None:
    print(f"Veri      : {len(data)} koşu, {len(sensor_columns)} sensör özniteliği")
    print(f"Vaka      : {data['case'].nunique()}  (dışlanan: {list(bad_cases) or 'yok'})")
    print(f"Koşul     : {data['condition'].nunique()}")
    print(f"Malzeme   : {sorted(data['material'].unique())}  (1=dökme demir, 2=çelik)")
    print(f"Eşik      : {limit:.0f} µm")
    print(f"VB aralığı: {data['vb_um'].min():.0f} - {data['vb_um'].max():.0f} µm "
          f"| eşik üstü {int((data['vb_um'] >= limit).sum())}/{len(data)}")


def _run_naive(data: pd.DataFrame, group_column: str, limit: float) -> pd.DataFrame:
    """Naif taban: koşu numarasından izotonik tahmin."""
    class _NaiveAdapter:
        def __init__(self):
            self._model = NaiveWearBaseline()

        def fit(self, X, y, **_):
            self._model.fit(X["run"], y)
            return self

        def predict(self, X):
            return self._model.predict(X["run"])

    table, _ = run_grouped_cv(
        data, group_column, ["run"], "vb_um",
        model_factory=_NaiveAdapter,
        wear_limit_um=limit, sort_column="run",
    )
    return table


if __name__ == "__main__":
    sys.exit(main())
