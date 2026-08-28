"""Model A - PHM 2010 çekirdek sensör modeli (Faz 04).

Görevi: sensör sinyalinden aşınma ne kadar iyi tahmin edilebilir? Bu, teslim
edilecek sistem değil; bir ÖLÇÜM ALETİ ve literatürle karşılaştırma zemini.

Dört şey ölçülür:
  1. Naif taban (referans)
  2. Gradyan artırma, ham tahmin
  3. Gradyan artırma + monoton düzleştirme
  4. Kanal alt kümeleri - kuvvet sensörünü kaybetmenin bedeli

    python scripts/run_model_a.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import PHM2010
from tcm.evaluation.protocol import run_grouped_cv, summarise_folds
from tcm.features.normalise import baseline_normalise
from tcm.models import NaiveWearBaseline, enforce_monotonic
from tcm.models.gbm import feature_importance, make_gbm, select_channels

META_COLUMNS = {"cutter", "cut", "vb_um", "flute_spread_um"}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save", action="store_true", help="sonuçları reports/ altına yaz")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    wear_limit = float(config.get("evaluation.wear_limit_um"))
    seed = int(config.get("random_seed", 42))
    channels = list(config.get("phm2010.channels"))

    cache = config.path("paths.data_processed") / "phm_cut_features.csv"
    if not cache.exists():
        print(f"Öznitelik tablosu yok: {cache}\n"
              "Önce çalıştırın: python scripts/build_features.py")
        return 1

    data = pd.read_csv(cache)
    feature_columns = [c for c in data.columns if c not in META_COLUMNS]

    print(f"Veri   : {len(data)} geçiş, {len(feature_columns)} öznitelik")
    print(f"Kesici : {', '.join(sorted(data['cutter'].unique()))}")
    print(f"Bölme  : leave-one-cutter-out (3 katlama)")
    print(f"Hedef  : vb_um  |  eşik {wear_limit:.0f} µm\n")

    rows = []

    # 1 -------------------------------------------------- naif taban
    naive_table = _run_naive(data, wear_limit)
    rows.append({"model": "0 · naif taban", **summarise_folds(naive_table)})

    # 2 -------------------------------------------------- GBM, ham
    raw_table, _ = run_grouped_cv(
        data, "cutter", feature_columns, "vb_um",
        model_factory=lambda: make_gbm(random_state=seed),
        wear_limit_um=wear_limit, sort_column="cut",
    )
    rows.append({"model": "1 · GBM (ham)", **summarise_folds(raw_table)})

    # 3 -------------------------------------------------- GBM + monoton
    mono_table, _ = run_grouped_cv(
        data, "cutter", feature_columns, "vb_um",
        model_factory=lambda: make_gbm(random_state=seed),
        wear_limit_um=wear_limit, sort_column="cut",
        postprocess=enforce_monotonic,
    )
    rows.append({"model": "2 · GBM + monoton", **summarise_folds(mono_table)})

    # 4 -------------------------------------------------- taban normalizasyonu
    # Teşhis: 168 özniteliğin 82'si kesiciler arasında daha ilk geçişlerde
    # 2 kattan fazla farklı. Model mutlak seviyeyi öğrenirse, sinyali baştan
    # yüksek olan bir kesicide aşınmayı olduğundan fazla tahmin eder.
    normalised = baseline_normalise(
        data, feature_columns, "cutter", "cut",
        n_baseline=int(config.get("phm2010.baseline_cuts", 10)),
    )
    norm_table, _ = run_grouped_cv(
        normalised, "cutter", feature_columns, "vb_um",
        model_factory=lambda: make_gbm(random_state=seed),
        wear_limit_um=wear_limit, sort_column="cut",
        postprocess=enforce_monotonic,
    )
    rows.append({"model": "3 · GBM + normalize + monoton", **summarise_folds(norm_table)})

    print("=" * 78)
    print("ANA SONUÇ")
    print("=" * 78)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    print("\nKatlama ayrıntısı (GBM ham):")
    print(raw_table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    _check_acceptance(summary, config)

    # 4 -------------------------------------------------- kanal alt kümeleri
    print("\n" + "=" * 78)
    print("KANAL ALT KÜMELERİ - kuvvet sensörünü kaybetmenin bedeli")
    print("=" * 78)
    print("Model B'de kuvvet kanalı olmayacak (NASA'da dinamometre yok).")
    print("Bu tablo, o kaybın ne kadara mal olduğunu şimdiden gösterir.\n")

    subsets = {
        "hepsi (7 kanal)": channels,
        "sadece kuvvet": [c for c in channels if c.startswith("force")],
        "titreşim + AE": [c for c in channels if c.startswith(("vib", "ae"))],
        "sadece titreşim": [c for c in channels if c.startswith("vib")],
        "sadece AE": [c for c in channels if c.startswith("ae")],
    }

    subset_rows = []
    for label, subset in subsets.items():
        columns = select_channels(feature_columns, subset)
        table, _ = run_grouped_cv(
            data, "cutter", columns, "vb_um",
            model_factory=lambda: make_gbm(random_state=seed),
            wear_limit_um=wear_limit, sort_column="cut",
            postprocess=enforce_monotonic,
        )
        subset_rows.append({
            "kanal kümesi": label,
            "oznitelik": len(columns),
            **summarise_folds(table),
        })

    subset_table = pd.DataFrame(subset_rows)
    print(subset_table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    # 5 -------------------------------------------------- öznitelik önemleri
    print("\n" + "=" * 78)
    print("ÖZNİTELİK ÖNEMLERİ (tüm veriyle eğitilmiş model)")
    print("=" * 78)
    final = make_gbm(random_state=seed)
    final.fit(data[feature_columns], data["vb_um"])
    importance = feature_importance(final, feature_columns)
    print(importance.head(15).to_string(index=False))

    importance["kanal"] = importance["oznitelik"].apply(
        lambda name: next((c for c in channels if name.startswith(f"{c}_")), "?")
    )
    print("\nKanal bazında toplam önem:")
    by_channel = (importance.groupby("kanal")["onem"].sum()
                  .sort_values(ascending=False).reset_index())
    by_channel["pay_%"] = (100 * by_channel["onem"] / by_channel["onem"].sum()).round(1)
    print(by_channel.to_string(index=False))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target / "model_a_summary.csv", index=False)
        mono_table.to_csv(target / "model_a_folds.csv", index=False)
        subset_table.to_csv(target / "model_a_channels.csv", index=False)
        importance.to_csv(target / "model_a_importance.csv", index=False)
        print(f"\nKaydedildi: {target}")

    return 0


def _run_naive(data: pd.DataFrame, wear_limit: float) -> pd.DataFrame:
    """Naif tabanı aynı protokolle çalıştırır - karşılaştırma adil olsun diye."""
    class _NaiveAdapter:
        """run_grouped_cv'nin beklediği fit/predict arayüzüne sarar."""

        def __init__(self):
            self._model = NaiveWearBaseline()

        def fit(self, X, y, **_):
            self._model.fit(X["cut"], y)
            return self

        def predict(self, X):
            return self._model.predict(X["cut"])

    table, _ = run_grouped_cv(
        data, "cutter", ["cut"], "vb_um",
        model_factory=_NaiveAdapter,
        wear_limit_um=wear_limit, sort_column="cut",
    )
    return table


def _check_acceptance(summary: pd.DataFrame, config) -> None:
    """Kabul kriterlerini kontrol eder ve açıkça söyler."""
    target_mae = float(config.get("acceptance.max_mae_um", 15.0))
    target_overshoot = float(config.get("acceptance.max_overshoot_um", 25.0))

    naive = summary.iloc[0]
    models = summary.iloc[1:]
    best_mae = models.iloc[models["mae_um"].to_numpy().argmin()]
    best_over = models.iloc[models["abs_overshoot_um"].to_numpy().argmin()]

    def verdict(ok: bool) -> str:
        return "GEÇTİ" if ok else "GEÇMEDİ"

    print("\n--- Kabul kriterleri ---")
    print(f"MAE < {target_mae:.0f} µm            | en iyi {best_mae['mae_um']:.2f} µm "
          f"({best_mae['model']})  {verdict(best_mae['mae_um'] < target_mae)}")
    print(f"|overshoot| < {target_overshoot:.0f} µm    | en iyi "
          f"{best_over['abs_overshoot_um']:.2f} µm ({best_over['model']})  "
          f"{verdict(best_over['abs_overshoot_um'] < target_overshoot)}")
    print(f"Naif tabanı geçmeli    | naif {naive['mae_um']:.2f} µm -> "
          f"model {best_mae['mae_um']:.2f} µm  "
          f"{verdict(best_mae['mae_um'] < naive['mae_um'])}")
    print(
        "\nNot: 'worst_overshoot_um' sütunu güvenlik açısından en kritik sayıdır -\n"
        "pozitif ve büyükse, model en kötü katlamada takımın eşiği bu kadar\n"
        "aşmasına izin vermiş demektir."
    )


if __name__ == "__main__":
    sys.exit(main())
