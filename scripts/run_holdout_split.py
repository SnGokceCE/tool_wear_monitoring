"""Sabit eğitim / doğrulama / test bölmesi - 100 / 20 / 25 (Faz 12).

Projenin geri kalanı çapraz doğrulama kullanıyor; bu betik klasik üçlü bölmeyi
uygular. İki bölme kuralını da çalıştırır, çünkü aradaki fark bu veri setinde
öğreticidir:

  takım bazlı   Bir takımın bütün koşuları aynı kümeye gider. 145 koşu tam
                olarak 100 / 20 / 25'e ayrılıyor (takım 11+8 test, 3+5
                doğrulama, kalan 11 takım eğitim).

  rastgele      Satırlar takım gözetilmeden karıştırılır. AYNI takımın
                koşuları hem eğitime hem teste düşer.

Rastgele bölme neden sorunlu: bir takımın ardışık koşuları neredeyse aynı
sinyali taşır (aşınma kademeli ilerler). 40. koşu eğitimde, 41. koşu testteyse
model ezberlediğini hatırlar, genellemeyi değil. Test hatası gerçek dışı düşer.

DOĞRULAMA KÜMESİ NE İŞE YARIYOR
-------------------------------
İki iş yapıyor, ikisi de test kümesine dokunmadan:
  1. Erken durdurma - kaç ağaç kurulacağını doğrulama hatası belirler.
  2. Alarm eşiği kalibrasyonu - eşik doğrulama tahminlerinden seçilir.

Test kümesi yalnızca en sonda, bir kez kullanılır.

    python scripts/run_holdout_split.py
    python scripts/run_holdout_split.py --split tool --save
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from tcm import load_config
from tcm.cli import setup_console
from tcm.decision import alarm_cost, alarm_flags, choose_threshold
from tcm.evaluation.classification import classification_scores
from tcm.evaluation.metrics import summarise
from tcm.models.gbm import SMALL_DATA_PARAMS
from tcm.provenance import format_stamp, run_stamp
from tcm.serving import resolve_feature_columns

# Takım bazlı bölme: bu üçlü 100/20/25'i tam tutturuyor.
TEST_CASES = (11, 8)
VALIDATION_CASES = (3, 5)

TARGET_SIZES = {"eğitim": 100, "doğrulama": 20, "test": 25}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--feature-set", default="sensor+param+time",
                        choices=["sensor+param+time", "param+time"])
    parser.add_argument("--split", default="both",
                        choices=["tool", "random", "both"])
    parser.add_argument("--detail", action="store_true",
                        help="test kümesinin satır satır tahminlerini ve "
                             "karışıklık matrisini bas")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))
    cost_missed = float(config.get("decision.cost_missed", 5.0))
    cost_false = float(config.get("decision.cost_false_alarm", 1.0))

    data = pd.read_csv(
        config.path("paths.data_processed") / "nasa_run_features.csv"
    ).sort_values(["case", "run"]).reset_index(drop=True)
    columns = resolve_feature_columns(data, args.feature_set)

    print("=" * 84)
    print("SABİT BÖLME - 100 eğitim / 20 doğrulama / 25 test")
    print("=" * 84)
    print(f"Veri     : {len(data)} koşu, {data['case'].nunique()} takım")
    print(f"Girdi    : {args.feature_set} ({len(columns)} öznitelik)")
    print(f"Aşınma s.: {limit:.0f} µm  |  maliyet {cost_missed:.0f}:{cost_false:.0f}")

    modes = ["tool", "random"] if args.split == "both" else [args.split]
    rows = []
    for mode in modes:
        rows.append(_run_split(data, columns, mode, limit, seed,
                               cost_missed, cost_false, detail=args.detail))

    table = pd.DataFrame(rows)
    _print_comparison(table, modes)

    sweep = None
    if "tool" in modes:
        sweep = _tree_sweep(data, columns, limit, seed)

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 84)
    print("ÇALIŞTIRMA KÜNYESİ")
    print("-" * 84)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        out = target / "holdout_split_summary.csv"
        table.assign(git_hash=stamp["git_hash"],
                     feature_set=args.feature_set).to_csv(out, index=False)
        print(f"\nKaydedildi: {out}")
        if sweep is not None:
            sweep_out = target / "holdout_tree_sweep.csv"
            sweep.assign(git_hash=stamp["git_hash"]).to_csv(sweep_out, index=False)
            print(f"Kaydedildi: {sweep_out}")

    return 0


# ------------------------------------------------------------------ bölme

def _split_by_tool(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bir takımın bütün koşuları aynı kümeye gider."""
    test = data[data["case"].isin(TEST_CASES)]
    validation = data[data["case"].isin(VALIDATION_CASES)]
    train = data[~data["case"].isin(TEST_CASES + VALIDATION_CASES)]
    return {"eğitim": train, "doğrulama": validation, "test": test}


def _split_random(data: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Satır bazlı rastgele bölme - takım sınırı gözetilmez."""
    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train, n_validation = TARGET_SIZES["eğitim"], TARGET_SIZES["doğrulama"]
    return {
        "eğitim": shuffled.iloc[:n_train],
        "doğrulama": shuffled.iloc[n_train:n_train + n_validation],
        "test": shuffled.iloc[n_train + n_validation:],
    }


def _describe(parts: dict[str, pd.DataFrame], mode: str) -> None:
    print("\n" + "=" * 84)
    label = "TAKIM BAZLI" if mode == "tool" else "RASTGELE (satır bazlı)"
    print(f"BÖLME: {label}")
    print("=" * 84)

    for name, part in parts.items():
        cases = sorted(int(c) for c in part["case"].unique())
        worn = int((part["vb_um"] >= 300).sum())
        print(f"  {name:10s} {len(part):3d} koşu | {len(cases):2d} takım | "
              f"aşınmış {worn:2d} | takımlar {cases}")

    # Sızıntı ölçüsü: eğitim ve test kümesinde ORTAK takım var mı?
    shared = set(parts["eğitim"]["case"]) & set(parts["test"]["case"])
    if shared:
        print(f"\n  ⚠ SIZINTI: {len(shared)} takım hem eğitimde hem testte "
              f"-> {sorted(int(c) for c in shared)}")
        print("    Aynı takımın komşu koşuları neredeyse aynı sinyali taşır;")
        print("    model ezberlediğini hatırlayabilir. Test hatası iyimser çıkar.")
    else:
        print("\n  ✓ Eğitim ve test kümeleri takım düzeyinde ayrık.")


# ------------------------------------------------------------- çalıştırma

def _run_split(data, columns, mode, limit, seed, cost_missed, cost_false,
               detail: bool = False) -> dict:
    parts = _split_by_tool(data) if mode == "tool" else _split_random(data, seed)
    _describe(parts, mode)

    train, validation, test = parts["eğitim"], parts["doğrulama"], parts["test"]

    # --- 1) Eğitim; ağaç sayısını DOĞRULAMA kümesi belirliyor --------------
    model = LGBMRegressor(
        **{**SMALL_DATA_PARAMS, "n_estimators": 2000},
        random_state=seed,
    )
    model.fit(
        train[columns], train["vb_um"],
        eval_set=[(validation[columns], validation["vb_um"])],
        eval_metric="l1",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    n_trees = int(model.best_iteration_ or model.n_estimators)

    # --- 2) Alarm eşiği; yine DOĞRULAMA kümesinden ------------------------
    val_pred = np.asarray(model.predict(validation[columns]), dtype=float)
    threshold = choose_threshold(
        validation["vb_um"].to_numpy(dtype=float), val_pred, limit,
        cost_missed=cost_missed, cost_false_alarm=cost_false,
        groups=validation["case"].to_numpy(),
    )

    print(f"\n  Erken durdurma  : {n_trees} ağaç (doğrulama hatası ile seçildi)")
    print(f"  Doğrulama MAE   : {np.mean(np.abs(val_pred - validation['vb_um'])):.2f} µm")
    print(f"  Seçilen eşik    : {threshold:.1f} µm")

    # --- 3) TEST - burada bir kez kullanılıyor ----------------------------
    test_pred = np.asarray(model.predict(test[columns]), dtype=float)
    truth = test["vb_um"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(test_pred - truth)))
    rmse = float(np.sqrt(np.mean((test_pred - truth) ** 2)))

    # Overshoot takım bazında anlamlı; takım başına hesaplanıp ortalanır.
    overshoots = []
    for _, tool in test.groupby("case"):
        tool = tool.sort_values("run")
        scores = summarise(
            tool["vb_um"].to_numpy(dtype=float),
            np.asarray(model.predict(tool[columns]), dtype=float),
            limit,
        )
        overshoots.append(scores["overshoot_um"])

    flags = alarm_flags(test_pred, threshold, 1, test["case"].to_numpy())
    worn = truth >= limit
    scores = classification_scores(worn, flags)

    print(f"\n  --- TEST ({len(test)} koşu) ---")
    print(f"  MAE             : {mae:.2f} µm")
    print(f"  RMSE            : {rmse:.2f} µm")
    print(f"  |overshoot|     : {np.mean(np.abs(overshoots)):.2f} µm")
    print(f"  Yakalama oranı  : {scores['worn_recall'] * 100:.1f}%  "
          f"(kaçırılan {int(scores['missed_worn'])}/{int(worn.sum())})")
    print(f"  Yanlış alarm    : {int(scores['false_alarms'])}")
    print(f"  Dengeli doğruluk: {scores['balanced_acc']:.3f}")

    if detail:
        _print_detail(test, truth, test_pred, flags, worn, threshold, scores)

    return {
        "bolme": "takım bazlı" if mode == "tool" else "rastgele",
        "n_egitim": len(train), "n_dogrulama": len(validation), "n_test": len(test),
        "agac": n_trees,
        "esik_um": threshold,
        "test_mae_um": mae,
        "test_rmse_um": rmse,
        "test_abs_overshoot_um": float(np.mean(np.abs(overshoots))),
        "worn_recall": scores["worn_recall"],
        "missed_worn": scores["missed_worn"],
        "false_alarms": scores["false_alarms"],
        "balanced_acc": scores["balanced_acc"],
        "maliyet": alarm_cost(worn, flags, cost_missed, cost_false),
    }



def _tree_sweep(data, columns, limit, seed) -> pd.DataFrame:
    """Doğrulama kümesi model seçimi için yeterli mi? Ağaç sayısını tarar.

    Bulgu: bu bölmede doğrulama ve test ZIT yönleri gösteriyor. Doğrulama
    hatası ağaç sayısıyla artarken test hatası düşüyor. Erken durdurma
    doğrulamaya baktığı için en az ağacı seçiyor - ve bu test için en kötü
    seçim oluyor.

    Sebep: doğrulama kümesi yalnızca 2 takımdan (20 koşu) oluşuyor ve o iki
    takımın aşınma davranışı test takımlarınınkine benzemiyor. 20 satır,
    model seçimi için yeterli bir örneklem değil.
    """
    parts = _split_by_tool(data)
    train, validation, test = parts["eğitim"], parts["doğrulama"], parts["test"]

    print("\n" + "=" * 84)
    print("TEŞHİS - doğrulama kümesi model seçimi için yeterli mi?")
    print("=" * 84)

    rows = []
    for n_trees in (10, 50, 100, 300, 600):
        model = LGBMRegressor(
            **{**SMALL_DATA_PARAMS, "n_estimators": n_trees}, random_state=seed
        )
        model.fit(train[columns], train["vb_um"])
        val_mae = float(np.mean(
            np.abs(model.predict(validation[columns]) - validation["vb_um"])))
        test_mae = float(np.mean(
            np.abs(model.predict(test[columns]) - test["vb_um"])))
        rows.append({"agac": n_trees, "dogrulama_mae_um": val_mae,
                     "test_mae_um": test_mae})

    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    best_val = int(sweep.loc[sweep["dogrulama_mae_um"].idxmin(), "agac"])
    best_test = int(sweep.loc[sweep["test_mae_um"].idxmin(), "agac"])
    print(f"\n  Doğrulamaya göre en iyi : {best_val:3d} ağaç")
    print(f"  Teste göre en iyi       : {best_test:3d} ağaç")

    if best_val != best_test:
        print(
            "\n  Doğrulama ve test ZIT yönü gösteriyor. Doğrulama kümesi 2\n"
            "  takımdan (20 koşu) oluşuyor ve test takımlarını temsil etmiyor;\n"
            "  model seçimini aktif olarak YANLIŞ yöne çekiyor. Bu veri\n"
            "  ölçeğinde sabit bölme, çapraz doğrulamanın yerini tutmuyor."
        )
    return sweep


def _print_detail(test, truth, pred, flags, worn, threshold, scores) -> None:
    """Test kümesinin satır satır dökümü ve karışıklık matrisi.

    Tek bir MAE sayısı modelin nerede yanıldığını göstermez; bu tablo
    gösterir. Sütunlar:

      gerçek/tahmin VB : mikrometre
      hata             : tahmin - gerçek (pozitif = fazla tahmin)
      aşınmış?         : gerçek VB >= 300 (ISO sınırı)
      alarm?           : tahmin >= kalibre eşik, takım içinde kilitli
      durum            : dördünün kesişimi
    """
    print("\n  " + "-" * 74)
    print("  TEST KÜMESİ - satır satır")
    print("  " + "-" * 74)

    def label(is_worn: bool, has_alarm: bool) -> str:
        if is_worn and has_alarm:
            return "TP  doğru yakalandı"
        if is_worn and not has_alarm:
            return "FN  KAÇIRILDI"
        if not is_worn and has_alarm:
            return "FP  yanlış alarm"
        return "TN  doğru"

    frame = pd.DataFrame({
        "takım": test["case"].astype(int).to_numpy(),
        "koşu": test["run"].astype(int).to_numpy(),
        "süre": test["cum_time"].to_numpy(),
        "gerçek": truth,
        "tahmin": pred,
        "hata": pred - truth,
        "aşınmış": np.where(worn, "evet", "hayır"),
        "alarm": np.where(flags, "VAR", "-"),
        "durum": [label(bool(w), bool(f)) for w, f in zip(worn, flags)],
    })
    print(frame.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))

    tp = int(np.sum(worn & flags))
    fn = int(np.sum(worn & ~flags))
    fp = int(np.sum(~worn & flags))
    tn = int(np.sum(~worn & ~flags))

    print("\n  " + "-" * 74)
    print(f"  KARIŞIKLIK MATRİSİ  (aşınma sınırı 300 µm, alarm eşiği {threshold:.0f} µm)")
    print("  " + "-" * 74)
    print("                    | alarm VAR | alarm YOK |")
    print(f"    gerçekte aşınmış |    TP {tp:3d} |    FN {fn:3d} |")
    print(f"    gerçekte sağlam  |    FP {fp:3d} |    TN {tn:3d} |")

    precision = scores["worn_precision"]
    recall = scores["worn_recall"]
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))

    print(f"""
  Precision (kesinlik)   : {precision:.3f}   "alarm" dediklerimizin kaçı gerçekten aşınmış
  Recall   (duyarlılık)  : {recall:.3f}   aşınmışların kaçını yakaladık   <- ASIL METRİK
  F1                     : {f1:.3f}   ikisinin harmonik ortalaması
  Specificity (seçicilik): {scores['unworn_recall']:.3f}   sağlamların kaçına doğru "sağlam" dedik
  Accuracy (doğruluk)    : {scores['accuracy']:.3f}   TEK BAŞINA YANILTICI
  Balanced accuracy      : {scores['balanced_acc']:.3f}   recall ve specificity ortalaması""")

    print("\n  Not: üretimde precision ile recall eşit önemde değildir. Kaçırılan")
    print("  aşınma (FN) parçayı hurdaya çıkarır; yanlış alarm (FP) yalnızca takım")
    print("  ömrü israf eder. Bu yüzden asıl bakılan recall'dır.")


def _print_comparison(table: pd.DataFrame, modes: list[str]) -> None:
    print("\n" + "=" * 84)
    print("KARŞILAŞTIRMA")
    print("=" * 84)
    view = table[["bolme", "n_egitim", "n_dogrulama", "n_test", "agac",
                  "esik_um", "test_mae_um", "test_rmse_um", "worn_recall"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    if len(modes) < 2:
        return

    tool = table[table["bolme"] == "takım bazlı"].iloc[0]
    rand = table[table["bolme"] == "rastgele"].iloc[0]
    change = 100 * (rand["test_mae_um"] - tool["test_mae_um"]) / tool["test_mae_um"]

    print(f"\nRastgele bölme, takım bazlı bölmeye göre {change:+.1f}% MAE.")
    if change < 0:
        print(
            "\nRastgele bölme daha İYİ görünüyor - ve sorun tam olarak budur.\n"
            "Aynı takımın koşuları hem eğitime hem teste düştüğü için model,\n"
            "test satırlarına çok benzeyen satırları eğitimde görmüş oluyor.\n"
            "Bu sayı sistemin genelleme yeteneğini DEĞİL, ezberini ölçüyor;\n"
            "yeni bir takımda bu performans elde edilmez."
        )


if __name__ == "__main__":
    sys.exit(main())
