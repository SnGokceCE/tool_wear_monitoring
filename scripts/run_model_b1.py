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
from lightgbm import LGBMClassifier

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import NASAMilling
from tcm.evaluation.protocol import run_grouped_cv, summarise_folds
from tcm.features.build import load_or_build_nasa
from tcm.models import NaiveWearBaseline, enforce_monotonic
from tcm.models.gbm import SMALL_DATA_PARAMS, feature_importance, make_gbm_small
from tcm.provenance import format_stamp, run_stamp

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

        # Süresiz parametre modeli. 2. satırla karşılaştırıldığında kümülatif
        # sürenin tek başına ne kadar iş yaptığını gösterir: parametreler
        # aşınma HIZINI belirler, MİKTARINI değil.
        table_param, _ = run_grouped_cv(
            data, group_column, PARAMETER_COLUMNS, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        rows.append({"model": "5 · sadece parametre (süresiz)",
                     **summarise_folds(table_param)})

        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        if group_column == "material":
            print("\nKatlama ayrıntısı (sensör + parametre):")
            print(table_all.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        summary["sinav"] = label
        all_rows.append(summary)

    # ------------------------------------------------------- ek analizler
    extras = _extra_analyses(data, sensor_columns, protocols, limit, seed)

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

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 78)
    print("ÇALIŞTIRMA KÜNYESİ")
    print("-" * 78)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        pd.concat(all_rows, ignore_index=True).assign(
            git_hash=stamp["git_hash"]
        ).to_csv(target / "model_b1_summary.csv", index=False)
        importance.to_csv(target / "model_b1_importance.csv", index=False)
        extras.assign(git_hash=stamp["git_hash"]).to_csv(
            target / "model_b1_extras.csv", index=False)
        print(f"\nKaydedildi: {target}")

    return 0


def _extra_analyses(data, sensor_columns, protocols, limit, seed) -> pd.DataFrame:
    """İki yan analiz. Önceden elle yapılıyorlardı; betiğe alındılar.

    Elle yapılan analizin sorunu, sayısının kaydedilmemesi ve protokolünün
    yazılı olmaması: "sensörden malzeme tahmini %79,5" değeri raporda üç gün
    durdu ve hangi bölmeyle üretildiği bilinmediği için yeniden üretilemedi.
    """
    rows = []

    # --- 1) Aşırı aşınmış koşuların MAE'ye etkisi -------------------------
    #
    # NASA'da VB 1530 µm'ye kadar çıkıyor; ISO sınırının beş katı. Bu koşular
    # aşınma tahmini için anlamlı bir çalışma bölgesi değil ve hatayı tek
    # başlarına şişiriyorlar. Sınırlandırılmış alt küme, modelin KULLANILABİLİR
    # aralıktaki gerçek doğruluğunu gösterir.
    print("\n" + "=" * 78)
    print("EK ANALİZ 1 - aşırı aşınmış koşuların MAE'ye etkisi")
    print("=" * 78)
    capped = data[data["vb_um"] <= 600.0]
    print(f"VB ≤ 600 µm: {len(capped)}/{len(data)} koşu\n")

    for label, group_column in protocols.items():
        if capped[group_column].nunique() < 2:
            continue
        full, _ = run_grouped_cv(
            data, group_column, sensor_columns, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        sub, _ = run_grouped_cv(
            capped, group_column, sensor_columns, "vb_um",
            model_factory=lambda: make_gbm_small(random_state=seed),
            wear_limit_um=limit, sort_column="run",
        )
        mae_full = summarise_folds(full)["mae_um"]
        mae_sub = summarise_folds(sub)["mae_um"]
        print(f"  {label:14s} sensör MAE  tam veri {mae_full:7.2f}  ->  "
              f"VB≤600 {mae_sub:7.2f}   ({100 * (mae_sub - mae_full) / mae_full:+.1f}%)")
        rows.append({"analiz": "vb_kapali_600", "sinav": label,
                     "deger": mae_sub, "referans": mae_full})

    # --- 2) Sensörler malzemeyi ele veriyor mu? --------------------------
    #
    # Bölme protokolü sonucu BELİRLİYOR, o yüzden ikisi de raporlanıyor:
    #
    #   takım bazında  - iyimser. `condition` malzemeyi İÇERİYOR (malzeme +
    #                    doc + feed), dolayısıyla dışarıda bırakılan takımın
    #                    kardeşleri aynı koşulla eğitimde kalır ve model
    #                    "bu imza = bu koşul = bu malzeme" ezberleyebilir.
    #   koşul bazında  - dürüst sınav: hiç görülmemiş bir kesme koşulunda
    #                    malzeme sinyalden okunabiliyor mu?
    print("\n" + "=" * 78)
    print("EK ANALİZ 2 - sensörler malzemeyi ele veriyor mu?")
    print("=" * 78)
    baseline = float(data["material"].value_counts().max() / len(data))
    print(f"Çoğunluk sınıfı tabanı: {baseline:.3f}\n")

    for label, group_column in [("takım bazında", "case"), ("koşul bazında", "condition")]:
        truths, preds = [], []
        for held_out in sorted(data[group_column].unique()):
            train = data[data[group_column] != held_out]
            test = data[data[group_column] == held_out]
            if train["material"].nunique() < 2:
                continue
            model = LGBMClassifier(**SMALL_DATA_PARAMS, random_state=seed)
            model.fit(train[sensor_columns], train["material"])
            truths.append(test["material"].to_numpy())
            preds.append(model.predict(test[sensor_columns]))

        if not truths:
            continue
        accuracy = float((np.concatenate(truths) == np.concatenate(preds)).mean())
        print(f"  {label:14s} doğruluk {accuracy:.3f}   "
              f"(tabanın {accuracy - baseline:+.3f} üstünde)")
        rows.append({"analiz": "malzeme_tahmini", "sinav": label,
                     "deger": accuracy, "referans": baseline})

    print("\n  Koşul bazındaki sayı asıl olandır: takım bazında bölme, kardeş")
    print("  takımlar aynı kesme koşulunu paylaştığı için iyimserdir.")

    return pd.DataFrame(rows)


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
