"""Faz 09 - karar mantığı: alarm eşiğinin veriden seçilmesi.

Model bir VB sayısı üretiyor; "takımı değiştir" demek ayrı bir karar.
Varsayılan eşik (aşınma sınırının kendisi) optimal değil, çünkü modelin hatası
var ve hata türlerinin maliyeti simetrik değil.

SIZINTIYI ÖNLEYEN KURGU
-----------------------
Eşik, her dış katlamanın İÇİNDE ikinci bir çapraz doğrulama ile seçilir:

    dış katlama: vaka 7 testte, geri kalanı eğitimde
        iç çapraz doğrulama: eğitim vakalarını kendi aralarında böl,
                             katlama dışı tahminler üret
        eşik: bu tahminlerden maliyeti en aza indiren değer
    seçilen eşik vaka 7'ye uygulanır

Test verisi eşik seçimine hiçbir noktada karışmaz.

    python scripts/run_decision_rule.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.decision import (
    alarm_cost,
    alarm_flags,
    choose_consecutive,
    choose_threshold,
    oof_predictions,
)
from tcm.evaluation.classification import classification_scores
from tcm.models.gbm import make_gbm_small

META_COLUMNS = {"case", "run", "vb_um", "condition", "run_time"}
PARAMETER_COLUMNS = ["material", "feed", "doc", "rpm"]
TIME_COLUMN = "cum_time"
PROTOCOLS = {"vaka-dışı": "case", "koşul-dışı": "condition", "malzeme-dışı": "material"}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cost-missed", type=float, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))
    cost_missed = args.cost_missed or float(config.get("decision.cost_missed", 5.0))
    cost_false = float(config.get("decision.cost_false_alarm", 1.0))

    data = pd.read_csv(config.path("paths.data_processed") / "nasa_run_features.csv").copy()
    data["worn"] = data["vb_um"] >= limit
    columns = PARAMETER_COLUMNS + [TIME_COLUMN]

    print("KARAR MANTIĞI")
    print("=" * 84)
    print(f"Model     : gradyan artırma, girdi = parametre + süre ({len(columns)} sütun)")
    print(f"Eşik      : aşınma sınırı {limit:.0f} µm")
    print(f"Maliyet   : 1 kaçırılan aşınma = {cost_missed / cost_false:.0f} yanlış alarm")
    print(f"Eşik seçimi: her katlamanın içinde, YALNIZCA eğitim verisiyle")

    all_rows = []
    for exam, group_column in PROTOCOLS.items():
        print("\n" + "=" * 84)
        print(f"SINAV: {exam}")
        print("=" * 84)

        rows = []
        rows.append({"kural": "sabit eşik (= sınır)",
                     **_run(data, group_column, columns, limit, seed,
                            cost_missed, cost_false, tune=False, consecutive=1)})
        rows.append({"kural": "ayarlı eşik",
                     **_run(data, group_column, columns, limit, seed,
                            cost_missed, cost_false, tune=True, consecutive=1)})
        rows.append({"kural": "ayarlı eşik + ardışık onay",
                     **_run(data, group_column, columns, limit, seed,
                            cost_missed, cost_false, tune=True, consecutive=None)})

        table = pd.DataFrame(rows)[
            ["kural", "missed_worn", "false_alarms", "worn_recall",
             "balanced_acc", "maliyet", "secilen_esik", "secilen_k"]
        ]
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

        table["sinav"] = exam
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True)
    _verdict(combined, cost_missed, cost_false)

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        combined.to_csv(target / "decision_rule_summary.csv", index=False)
        print(f"\nKaydedildi: {target / 'decision_rule_summary.csv'}")

    return 0


def _run(data, group_column, columns, limit, seed, cost_missed, cost_false,
         tune: bool, consecutive) -> dict[str, float]:
    """Dış çapraz doğrulama; eşik iç çapraz doğrulamadan seçilir."""
    truths, flags, thresholds, ks = [], [], [], []

    for held_out in sorted(data[group_column].unique()):
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out].sort_values("run")

        model = make_gbm_small(random_state=seed)
        model.fit(train[columns], train["vb_um"])
        predicted = np.asarray(model.predict(test[columns]), dtype=float)

        if tune:
            inner_true, inner_pred, inner_groups = _inner_predictions(
                train, group_column, columns, seed
            )
            k = 1 if consecutive is not None else choose_consecutive(
                inner_true, inner_pred, limit, limit,
                cost_missed, cost_false, groups=inner_groups,
            )
            threshold = choose_threshold(
                inner_true, inner_pred, limit,
                cost_missed=cost_missed, cost_false_alarm=cost_false,
                consecutive=k, groups=inner_groups,
            )
        else:
            threshold, k = limit, 1

        thresholds.append(threshold)
        ks.append(k)
        truths.append(test["worn"].to_numpy())
        # Kilit TAKIM bazında. Katlamanın tamamına uygulamak (eski hali) bir
        # takımdaki erken alarmı sonraki bütün takımlara taşıyordu ve
        # kaçırılan aşınmayı üçte bir gösteriyordu. Bkz. README, "yol boyunca
        # bulunan hatalar".
        flags.append(alarm_flags(predicted, threshold, k, test["case"].to_numpy()))

    truth = np.concatenate(truths)
    flag = np.concatenate(flags)
    scores = classification_scores(truth, flag)
    scores["maliyet"] = alarm_cost(truth, flag, cost_missed, cost_false)
    scores["secilen_esik"] = float(np.mean(thresholds))
    scores["secilen_k"] = float(np.mean(ks))
    return scores


def _inner_predictions(train, group_column, columns, seed):
    """Eğitim kümesi içinde katlama dışı tahminler üretir.

    Eşik bu tahminlerden seçilir; böylece dış test kümesi hiç kullanılmaz.

    İç gruplama: dış gruplama sütununda en az 3 grup varsa o kullanılır.
    Aksi halde (malzeme-dışı sınavında eğitim tarafında tek malzeme kalır)
    takım (case) bazında bölünür. Bu, iç tahminleri dış sınavdan biraz
    iyimser yapar - eşik seçimi için kabul edilebilir, ama raporlanmalı.

    Uygulama ``tcm.decision.oof_predictions`` içinde: Faz 06'daki nihai model
    de aynı fonksiyonu çağırıyor. İkisinin aynı kodu kullanması, "eşik Faz
    09'daki mantıkla seçildi" ifadesinin denetlenebilir olmasının koşulu.
    """
    oof = oof_predictions(
        train, group_column, columns,
        lambda: make_gbm_small(random_state=seed),
        target_column="vb_um", sort_column="run",
        latch_column="case", fallback_column="case", min_groups=3,
    )
    return oof.y_true, oof.y_pred, oof.groups


def _verdict(combined, cost_missed, cost_false) -> None:
    print("\n" + "=" * 84)
    print("KARAR")
    print("=" * 84)

    print("\nKaçırılan aşınma sayısı:")
    print(combined.pivot(index="kural", columns="sinav", values="missed_worn")
          .to_string(float_format=lambda v: f"{v:6.0f}"))

    print("\nYanlış alarm sayısı:")
    print(combined.pivot(index="kural", columns="sinav", values="false_alarms")
          .to_string(float_format=lambda v: f"{v:6.0f}"))

    print(f"\nToplam maliyet (1 kaçırma = {cost_missed / cost_false:.0f} yanlış alarm):")
    pivot = combined.pivot(index="kural", columns="sinav", values="maliyet")
    pivot["TOPLAM"] = pivot.sum(axis=1)
    print(pivot.to_string(float_format=lambda v: f"{v:8.0f}"))

    baseline = pivot.loc["sabit eşik (= sınır)", "TOPLAM"]
    for rule in pivot.index:
        if rule == "sabit eşik (= sınır)":
            continue
        change = 100 * (pivot.loc[rule, "TOPLAM"] - baseline) / baseline
        print(f"  {rule:30s} {change:+6.1f}% maliyet")


if __name__ == "__main__":
    sys.exit(main())
