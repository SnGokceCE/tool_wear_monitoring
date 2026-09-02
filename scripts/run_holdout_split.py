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
from tcm.models.deep import (
    CNNGRUWearModel,
    SignalStandardiser,
    predict,
    require_torch,
    train_model,
)
from tcm.models.gbm import SMALL_DATA_PARAMS
from tcm.provenance import format_stamp, run_stamp
from tcm.serving import resolve_feature_columns

# Takım bazlı bölme: bu üçlü 100/20/25'i tam tutturuyor.
TEST_CASES = (11, 8)
VALIDATION_CASES = (3, 5)

TARGET_SIZES = {"eğitim": 100, "doğrulama": 20, "test": 25}

# Derin modele giren kesme parametreleri (evrişimden geçmez, GRU
# çıktısına eklenir). Sensör öznitelikleri yok - ağ ham sinyalden
# kendi çıkarımını yapıyor.
DEEP_PARAMETERS = ["material", "feed", "doc", "rpm", "cum_time"]


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--feature-set", default="sensor+param+time",
                        choices=["sensor+param+time", "param+time"])
    parser.add_argument("--split", default="both",
                        choices=["tool", "random", "both"])
    parser.add_argument("--model", default="gbm",
                        choices=["gbm", "deep", "both"],
                        help="gbm = LightGBM, deep = 1B-CNN+GRU")
    parser.add_argument("--seeds", type=int, default=3,
                        help="derin model kaç tohumla tekrarlanacak")
    parser.add_argument("--epochs", type=int, default=200)
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

    if args.model in ("gbm", "both"):
        for mode in modes:
            rows.append(_run_split(data, columns, mode, limit, seed,
                                   cost_missed, cost_false, detail=args.detail))

    if args.model in ("deep", "both"):
        rows.extend(_run_deep_split(
            config, data, limit, seed, cost_missed, cost_false,
            n_seeds=args.seeds, epochs=args.epochs, modes=modes,
            detail=args.detail,
        ))

    # İki model de çalıştıysa, aynı test satırlarını yan yana bas.
    joint = {}
    if args.model == "both" and args.detail:
        for mode in modes:
            frame = _print_joint_detail(data, rows, mode, limit)
            if frame is not None:
                joint[mode] = frame

    table = pd.DataFrame([
        {k: v for k, v in row.items() if not k.startswith("_")} for row in rows
    ])
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
        if joint:
            detail_out = target / "holdout_detail.csv"
            pd.concat(joint.values(), ignore_index=True).assign(
                git_hash=stamp["git_hash"]).to_csv(detail_out, index=False)
            print(f"Kaydedildi: {detail_out}")

    return 0


# ------------------------------------------------------------------ bölme

def _split_by_tool(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bir takımın bütün koşuları aynı kümeye gider."""
    test = data[data["case"].isin(TEST_CASES)]
    validation = data[data["case"].isin(VALIDATION_CASES)]
    train = data[~data["case"].isin(TEST_CASES + VALIDATION_CASES)]
    return {"eğitim": train, "doğrulama": validation, "test": test}


def _split_random(data: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Satır bazlı rastgele bölme - takım sınırı gözetilmez.

    DİKKAT: indeks SIFIRLANMAZ. Derin model kolu, ham sinyal dizisine
    (``nasa_signals.npy``) satır indeksiyle erişiyor; indeks sıfırlanırsa o
    erişim yanlış satırları çeker ve sinyaller etiketlerle eşleşmez.

    Bu gerçekten oldu: ``reset_index(drop=True)`` yüzünden derin model
    rastgele bölmede 308 µm MAE verdi. Sebep modelin başarısızlığı değil,
    bölmenin sessizce başka bir bölmeye dönüşmesiydi - eğitim ilk 100
    satırı, test son 25 satırı alıyordu (tablo takıma göre sıralı olduğu
    için bu aslında bir takım bölmesidir).
    """
    shuffled = data.sample(frac=1.0, random_state=seed)
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
        # Alt çizgiyle başlayan anahtarlar ortak dökümde kullanılıyor;
        # CSV'ye yazılmadan önce ayıklanıyorlar.
        "_model": "LightGBM",
        "_mode": mode,
        "_pred": test_pred,
        "_threshold": threshold,
        "_index": test.index.to_numpy(),
    }



def _run_deep_split(config, data, limit, seed, cost_missed, cost_false,
                    n_seeds: int, epochs: int, modes: list[str],
                    detail: bool = False) -> list[dict]:
    """Aynı bölmede 1B-CNN + GRU.

    LightGBM kolundan iki farkı var:

      1. Girdi ham sinyal (6 kanal x 4500 örnek) artı kesme parametreleri.
         Öznitelikleri biz tanımlamıyoruz.
      2. Tek koşuya güvenilmiyor. Sinir ağında ağırlık başlangıcı rastgele;
         bu veri ölçeğinde tohumlar arası saçılım model farkından büyük
         olabiliyor (Bölüm 5.6'da ±10-18 µm ölçüldü). Bu yüzden deney
         ``n_seeds`` tohumla tekrarlanıp ortalama VE saçılım raporlanıyor.

    Doğrulama kümesi burada da iki iş yapıyor: erken durdurma (en iyi epoch)
    ve alarm eşiği kalibrasyonu.
    """
    require_torch()

    signals = _load_signals(config, data)
    parameters = data[DEEP_PARAMETERS].to_numpy(dtype=np.float32)
    targets = data["vb_um"].to_numpy(dtype=np.float32)

    results = []
    for mode in modes:
        parts = _split_by_tool(data) if mode == "tool" else _split_random(data, seed)
        idx = {name: part.index.to_numpy() for name, part in parts.items()}
        test = parts["test"]
        validation = parts["doğrulama"]

        print("\n" + "=" * 84)
        label = "TAKIM BAZLI" if mode == "tool" else "RASTGELE (satır bazlı)"
        print(f"CNN + GRU · BÖLME: {label}  ({n_seeds} tohum, {epochs} epoch)")
        print("=" * 84)

        # Sinyal ve parametre standardizasyonu YALNIZCA eğitim kümesinden.
        standardiser = SignalStandardiser()
        x = {k: standardiser.fit_transform(signals[v]) if k == "eğitim"
             else standardiser.transform(signals[v]) for k, v in idx.items()}

        mean = parameters[idx["eğitim"]].mean(axis=0)
        std = parameters[idx["eğitim"]].std(axis=0)
        std[std < 1e-9] = 1.0
        p = {k: (parameters[v] - mean) / std for k, v in idx.items()}
        y = {k: targets[v] for k, v in idx.items()}

        per_seed, epochs_used, thresholds = [], [], []
        test_predictions = []

        for offset in range(n_seeds):
            current = seed + offset
            model = CNNGRUWearModel(
                n_channels=signals.shape[1], n_parameters=len(DEEP_PARAMETERS)
            )
            model = train_model(
                model, x["eğitim"], p["eğitim"], y["eğitim"],
                epochs=epochs, seed=current,
                validation=(x["doğrulama"], p["doğrulama"], y["doğrulama"]),
                patience=25,
            )
            epochs_used.append(getattr(model, "best_epoch_", epochs))

            val_pred = predict(model, x["doğrulama"], p["doğrulama"])
            threshold = choose_threshold(
                y["doğrulama"].astype(float), val_pred.astype(float), limit,
                cost_missed=cost_missed, cost_false_alarm=cost_false,
                groups=validation["case"].to_numpy(),
            )
            thresholds.append(threshold)

            prediction = predict(model, x["test"], p["test"]).astype(float)
            test_predictions.append(prediction)
            per_seed.append(float(np.mean(np.abs(prediction - y["test"]))))

            print(f"  tohum {current}: en iyi epoch {epochs_used[-1]:3d}  "
                  f"eşik {threshold:6.1f}  test MAE {per_seed[-1]:7.2f}")

        mean_pred = np.mean(test_predictions, axis=0)
        threshold = float(np.mean(thresholds))
        truth = y["test"].astype(float)

        flags = alarm_flags(mean_pred, threshold, 1, test["case"].to_numpy())
        worn = truth >= limit
        scores = classification_scores(worn, flags)

        overshoots = []
        for case in sorted(test["case"].unique()):
            mask = (test["case"] == case).to_numpy()
            order = np.argsort(test.loc[mask, "run"].to_numpy())
            overshoots.append(summarise(
                truth[mask][order], mean_pred[mask][order], limit)["overshoot_um"])

        mae = float(np.mean(np.abs(mean_pred - truth)))
        spread = float(np.std(per_seed))

        print(f"\n  Tohum başına test MAE : "
              f"{'  '.join(f'{v:.1f}' for v in per_seed)}")
        print(f"  Ortalama ± saçılım    : {np.mean(per_seed):.2f} ± {spread:.2f} µm")
        print(f"  Ortalama tahminle MAE : {mae:.2f} µm")
        print(f"  Yakalama oranı        : {scores['worn_recall'] * 100:.1f}%  "
              f"(kaçırılan {int(scores['missed_worn'])}/{int(worn.sum())})")
        print(f"  Yanlış alarm          : {int(scores['false_alarms'])}")

        if detail:
            # Tahmin olarak tohumların ORTALAMASI kullanılıyor; tek bir
            # tohumun dökümü rastgele başlangıca bağlı olurdu.
            _print_detail(test, truth, mean_pred, flags, worn, threshold, scores)

        results.append({
            "bolme": ("takım bazlı" if mode == "tool" else "rastgele") + " · CNN+GRU",
            "n_egitim": len(parts["eğitim"]), "n_dogrulama": len(validation),
            "n_test": len(test),
            "agac": float(np.mean(epochs_used)),      # burada epoch sayısı
            "esik_um": threshold,
            "test_mae_um": mae,
            "test_rmse_um": float(np.sqrt(np.mean((mean_pred - truth) ** 2))),
            "test_abs_overshoot_um": float(np.mean(np.abs(overshoots))),
            "worn_recall": scores["worn_recall"],
            "missed_worn": scores["missed_worn"],
            "false_alarms": scores["false_alarms"],
            "balanced_acc": scores["balanced_acc"],
            "maliyet": alarm_cost(worn, flags, cost_missed, cost_false),
            "tohum_sayisi": n_seeds,
            "tohum_maeleri": ";".join(f"{v:.4f}" for v in per_seed),
            "mae_std": spread,
            "_model": "CNN+GRU",
            "_mode": mode,
            "_pred": mean_pred,
            "_threshold": threshold,
            "_index": test.index.to_numpy(),
        })

    return results


def _load_signals(config, data) -> np.ndarray:
    """Ham sinyal önbelleğini okur ve tablo sırasına hizalar."""
    cache = config.path("paths.data_processed") / "nasa_signals.npy"
    if not cache.exists():
        raise SystemExit(
            f"Sinyal önbelleği yok: {cache}\n"
            "Önce çalıştırın: python scripts/run_model_deep.py --protocols "
            '"malzeme-dışı" --seeds 1'
        )
    signals = np.load(cache)
    if len(signals) != len(data):
        raise SystemExit(
            f"Önbellek {len(signals)} örnek içeriyor ama tablo {len(data)} satır. "
            "Önbellek bayat; --rebuild ile yenileyin."
        )
    return signals


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


def _print_joint_detail(data, rows, mode, limit) -> None:
    """İki modelin aynı test satırlarındaki tahminlerini yan yana basar.

    Ayrı ayrı basılan dökümler karşılaştırmayı zorlaştırıyordu: hangi koşuda
    hangi modelin yanıldığını görmek için iki tabloyu göz kararı hizalamak
    gerekiyordu. Bu tablo ikisini tek satırda birleştiriyor.

    Her modelin kendi alarm eşiği var (doğrulama kümesinden ayrı ayrı
    kalibre edildi), o yüzden eşikler başlıkta ayrı ayrı yazılıyor.
    """
    selected = [r for r in rows if r.get("_mode") == mode]
    if len(selected) < 2:
        return None

    by_model = {r["_model"]: r for r in selected}
    if not {"LightGBM", "CNN+GRU"} <= set(by_model):
        return None

    gbm, cnn = by_model["LightGBM"], by_model["CNN+GRU"]
    index = gbm["_index"]
    test = data.loc[index]
    truth = test["vb_um"].to_numpy(dtype=float)
    worn = truth >= limit

    groups = test["case"].to_numpy()
    gbm_flags = alarm_flags(gbm["_pred"], gbm["_threshold"], 1, groups)
    cnn_flags = alarm_flags(cnn["_pred"], cnn["_threshold"], 1, groups)

    def status(is_worn, has_alarm):
        if is_worn:
            return "TP" if has_alarm else "FN"
        return "FP" if has_alarm else "TN"

    label = "TAKIM BAZLI" if mode == "tool" else "RASTGELE"
    print("\n" + "=" * 100)
    print(f"ORTAK DÖKÜM - {label}  |  test {len(test)} koşu")
    print(f"aşınma sınırı {limit:.0f} µm  ·  "
          f"LightGBM eşiği {gbm['_threshold']:.0f} µm  ·  "
          f"CNN+GRU eşiği {cnn['_threshold']:.0f} µm")
    print("=" * 100)

    frame = pd.DataFrame({
        "takım": test["case"].astype(int).to_numpy(),
        "koşu": test["run"].astype(int).to_numpy(),
        "süre": test["cum_time"].to_numpy(),
        "gerçek": truth,
        "GBM tahmin": gbm["_pred"],
        "GBM hata": gbm["_pred"] - truth,
        "GBM": [status(bool(w), bool(f)) for w, f in zip(worn, gbm_flags)],
        "CNN tahmin": cnn["_pred"],
        "CNN hata": cnn["_pred"] - truth,
        "CNN": [status(bool(w), bool(f)) for w, f in zip(worn, cnn_flags)],
        "aşınmış": np.where(worn, "evet", "hayır"),
    })
    print(frame.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))

    def counts(flags):
        return (int(np.sum(worn & flags)), int(np.sum(worn & ~flags)),
                int(np.sum(~worn & flags)), int(np.sum(~worn & ~flags)))

    g_tp, g_fn, g_fp, g_tn = counts(gbm_flags)
    c_tp, c_fn, c_fp, c_tn = counts(cnn_flags)

    print(f"""
  Model      TP   FN   FP   TN   recall     MAE
  LightGBM  {g_tp:3d}  {g_fn:3d}  {g_fp:3d}  {g_tn:3d}    {g_tp / max(g_tp + g_fn, 1):.3f}  {np.mean(np.abs(gbm['_pred'] - truth)):6.2f}
  CNN+GRU   {c_tp:3d}  {c_fn:3d}  {c_fp:3d}  {c_tn:3d}    {c_tp / max(c_tp + c_fn, 1):.3f}  {np.mean(np.abs(cnn['_pred'] - truth)):6.2f}""")

    differing = [i for i in range(len(truth))
                 if status(bool(worn[i]), bool(gbm_flags[i]))
                 != status(bool(worn[i]), bool(cnn_flags[i]))]
    if differing:
        print(f"\n  İki modelin AYRIŞTIĞI {len(differing)} koşu:")
        for i in differing:
            print(f"    takım {int(test['case'].iloc[i])} koşu {int(test['run'].iloc[i]):2d}  "
                  f"gerçek {truth[i]:6.1f}  |  GBM {gbm['_pred'][i]:6.1f} "
                  f"{status(bool(worn[i]), bool(gbm_flags[i]))}  |  "
                  f"CNN {cnn['_pred'][i]:6.1f} "
                  f"{status(bool(worn[i]), bool(cnn_flags[i]))}")

    return frame.assign(
        bolme="takım bazlı" if mode == "tool" else "rastgele",
        gbm_esik=gbm["_threshold"],
        cnn_esik=cnn["_threshold"],
    )


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
