"""worn / unworn sınıflandırma çıktısı (Faz 04d).

Mentor güncellemesi: girdiler serbest, ama çıktılar arasında worn/unworn
bulunmak zorunda.

İki üretim yolu karşılaştırılıyor:

  A) REGRESYON + EŞİK
     VB tahmin edilir, eşiği geçtiyse "worn". Tüm aşınma eğrisini doğru
     tahmin etmeye çalışır; eşik sonradan uygulanır.

  B) DOĞRUDAN SINIFLANDIRICI
     Modele VB hiç öğretilmez, doğrudan "aşınmış mı" sorulur. Sadece eşiğin
     etrafındaki karar sınırını öğrenir.

Hangisinin iyi olduğu deneysel bir soru: (A) daha fazla bilgi kullanır ama
dolaylıdır; (B) doğrudan hedefe gider ama eğrinin geri kalanını görmez.

    python scripts/run_classification.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from tcm import load_config
from tcm.cli import setup_console
from tcm.evaluation.classification import classification_scores, majority_baseline
from tcm.models.gbm import SMALL_DATA_PARAMS, make_gbm_small
from tcm.models import NaiveWearBaseline

META_COLUMNS = {"case", "run", "vb_um", "condition", "run_time"}
PARAMETER_COLUMNS = ["material", "feed", "doc", "rpm"]
TIME_COLUMN = "cum_time"

PROTOCOLS = {"vaka-dışı": "case", "koşul-dışı": "condition", "malzeme-dışı": "material"}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))

    data = pd.read_csv(config.path("paths.data_processed") / "nasa_run_features.csv")
    sensors = [
        c for c in data.columns
        if c not in META_COLUMNS and c not in PARAMETER_COLUMNS
        and c != TIME_COLUMN and data[c].notna().any()
    ]
    param_time = PARAMETER_COLUMNS + [TIME_COLUMN]

    data = data.copy()
    data["worn"] = data["vb_um"] >= limit

    print(f"Veri  : {len(data)} koşu | eşik {limit:.0f} µm")
    print(f"worn  : {int(data['worn'].sum())}  |  unworn: {int((~data['worn']).sum())}")

    base = majority_baseline(data["worn"])
    print(f"\n'Hep çoğunluk sınıfını söyle' tabanı:")
    print(f"  doğruluk {base['accuracy']:.3f} ama worn_recall {base['worn_recall']:.3f} "
          f"- {int(base['missed_worn'])} aşınmış takımın hepsi kaçırılıyor.")
    print("  Doğruluğun neden tek başına yeterli olmadığının kanıtı.")

    input_sets = {
        "sensör": sensors,
        "parametre + süre": param_time,
        "hepsi": sensors + param_time,
    }

    all_rows = []
    for exam, group_column in PROTOCOLS.items():
        print("\n" + "=" * 92)
        print(f"SINAV: {exam}  (gruplama: {group_column})")
        print("=" * 92)

        rows = [{"yöntem": "0 · naif (koşu no + eşik)", "girdi": "1",
                 **_evaluate_naive(data, group_column, limit)}]

        for label, columns in input_sets.items():
            rows.append({
                "yöntem": f"A · regresyon + eşik",
                "girdi": label,
                **_evaluate_regression(data, group_column, columns, limit, seed),
            })
            rows.append({
                "yöntem": f"B · doğrudan sınıflandırıcı",
                "girdi": label,
                **_evaluate_classifier(data, group_column, columns, seed),
            })

        table = pd.DataFrame(rows)[
            ["yöntem", "girdi", "balanced_acc", "worn_recall", "unworn_recall",
             "worn_precision", "missed_worn", "false_alarms"]
        ]
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        table["sinav"] = exam
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True)
    _verdict(combined)

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        combined.to_csv(target / "classification_summary.csv", index=False)
        print(f"\nKaydedildi: {target / 'classification_summary.csv'}")

    return 0


# ------------------------------------------------------------------ yöntemler

def _evaluate_naive(data, group_column, limit) -> dict[str, float]:
    """Naif taban: koşu numarasından VB tahmini, sonra eşik."""
    truths, preds = [], []
    for held_out in sorted(data[group_column].unique()):
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out].sort_values("run")

        model = NaiveWearBaseline().fit(train["run"], train["vb_um"])
        predicted = model.predict(test["run"])

        truths.append(test["worn"].to_numpy())
        preds.append(predicted >= limit)

    return classification_scores(np.concatenate(truths), np.concatenate(preds))


def _evaluate_regression(data, group_column, columns, limit, seed) -> dict[str, float]:
    """A yolu: VB regresyonu eğit, tahmini eşikle."""
    truths, preds = [], []
    for held_out in sorted(data[group_column].unique()):
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out].sort_values("run")

        model = make_gbm_small(random_state=seed)
        model.fit(train[columns], train["vb_um"])
        predicted = np.asarray(model.predict(test[columns]), dtype=float)

        truths.append(test["worn"].to_numpy())
        preds.append(predicted >= limit)

    return classification_scores(np.concatenate(truths), np.concatenate(preds))


def _evaluate_classifier(data, group_column, columns, seed) -> dict[str, float]:
    """B yolu: doğrudan worn/unworn sınıflandırıcı."""
    params = {k: v for k, v in SMALL_DATA_PARAMS.items()}
    truths, preds = [], []

    for held_out in sorted(data[group_column].unique()):
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out].sort_values("run")

        # Tek sınıflı eğitim kümesi olursa sınıflandırıcı eğitilemez.
        if train["worn"].nunique() < 2:
            truths.append(test["worn"].to_numpy())
            preds.append(np.full(len(test), bool(train["worn"].iloc[0])))
            continue

        model = LGBMClassifier(**params, random_state=seed)
        model.fit(train[columns], train["worn"])
        predicted = model.predict(test[columns]).astype(bool)

        truths.append(test["worn"].to_numpy())
        preds.append(predicted)

    return classification_scores(np.concatenate(truths), np.concatenate(preds))


def _verdict(combined: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("KARAR")
    print("=" * 92)

    print("\nDengeli doğruluk (balanced_acc), yöntem x sınav:")
    pivot = combined.pivot_table(
        index=["yöntem", "girdi"], columns="sinav", values="balanced_acc"
    )
    print(pivot.to_string(float_format=lambda v: f"{v:8.3f}"))

    print("\nKaçırılan aşınmış takım sayısı (asıl maliyet):")
    missed = combined.pivot_table(
        index=["yöntem", "girdi"], columns="sinav", values="missed_worn"
    )
    print(missed.to_string(float_format=lambda v: f"{v:6.0f}"))

    # Üç sınavda da en yüksek ortalama dengeli doğruluk - ama tek sınavda
    # parlayanı değil, hepsinde tutarlı olanı arıyoruz.
    mean_acc = pivot.mean(axis=1).sort_values(ascending=False)
    print("\nÜç sınavın ortalaması (yüksekten düşüğe):")
    print(mean_acc.to_string(float_format=lambda v: f"{v:8.3f}"))


if __name__ == "__main__":
    sys.exit(main())
