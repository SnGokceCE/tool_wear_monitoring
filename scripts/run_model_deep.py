"""Faz 05 - derin öğrenme (1B-CNN + GRU), NASA.

Gradyan artırmadan farkı: öznitelikleri biz tanımlamıyoruz, ağ ham sinyalden
kendisi öğreniyor.

Aynı üç sınav, aynı değerlendirme protokolü, aynı metrikler kullanılıyor -
yoksa gradyan artırmayla karşılaştırmak anlamsız olurdu.

    python scripts/run_model_deep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import NASAMilling
from tcm.evaluation.metrics import summarise
from tcm.evaluation.protocol import summarise_folds
from tcm.evaluation.verdict import (
    count_decisive_wins,
    describe_wins,
    seed_stability_verdict,
)
from tcm.features.timedomain import stable_region
from tcm.provenance import format_stamp, run_stamp
from tcm.models.deep import (
    CNNGRUWearModel,
    SignalStandardiser,
    predict,
    require_torch,
    train_model,
)
from tcm.models.gbm import make_gbm_small

PARAMETER_COLUMNS = ["material", "feed", "doc", "rpm", "cum_time"]
PROTOCOLS = {"vaka-dışı": "case", "koşul-dışı": "condition", "malzeme-dışı": "material"}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=1,
                        help="kaç farklı rastgele tohumla tekrarlanacak")
    parser.add_argument("--rebuild", action="store_true", help="sinyal önbelleğini yenile")
    parser.add_argument("--protocols", default=None,
                        help="virgülle ayrılmış sınav adları; boşsa hepsi")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    require_torch()

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))

    table = pd.read_csv(config.path("paths.data_processed") / "nasa_run_features.csv")
    signals = _load_signals(config, table, args.rebuild)

    print("DERİN ÖĞRENME - 1B-CNN + GRU")
    print("=" * 84)
    print(f"Sinyal   : {signals.shape[0]} örnek × {signals.shape[1]} kanal "
          f"× {signals.shape[2]} zaman adımı")
    print(f"Parametre: {', '.join(PARAMETER_COLUMNS)}")
    print(f"Epoch    : {args.epochs}")
    print(f"Uyarı    : {len(table)} örnekle derin ağ eğitiyoruz. Bu ölçek "
          "derin öğrenme için çok küçük;\n           model kasıtlı olarak "
          "küçük tutuldu ve düzenlileştirme yüksek.\n")

    protocols = PROTOCOLS
    if args.protocols:
        wanted = {name.strip() for name in args.protocols.split(",")}
        protocols = {k: v for k, v in PROTOCOLS.items() if k in wanted}
        if not protocols:
            raise SystemExit(f"Bilinmeyen sınav adı: {args.protocols}")

    all_rows = []
    for exam, group_column in protocols.items():
        print("=" * 84)
        print(f"SINAV: {exam}")
        print("=" * 84)

        started = time.time()
        # Küçük veride tek bir tohumla alınan sonuç güvenilir değil: ağırlık
        # başlangıcı ve yığın sırası şansa bağlı. Birden çok tohumla tekrarlayıp
        # ortalama VE saçılım raporlanıyor - saçılım farktan büyükse "kazandı"
        # demek anlamsızdır.
        seeds = [seed + offset for offset in range(args.seeds)]
        runs = [
            _run_deep(table, signals, group_column, limit, s, args.epochs)
            for s in seeds
        ]
        deep_scores = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        deep_std = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
        deep_time = time.time() - started

        started = time.time()
        gbm_scores = _run_gbm(table, group_column, limit, seed)
        gbm_time = time.time() - started

        rows = [
            {"model": "gradyan artırma", "süre_s": round(gbm_time, 1), **gbm_scores},
            {"model": "CNN + GRU", "süre_s": round(deep_time, 1), **deep_scores},
        ]
        summary = pd.DataFrame(rows)[
            ["model", "mae_um", "rmse_um", "abs_overshoot_um", "süre_s"]
        ].copy()
        print(summary.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

        change = 100 * (deep_scores["mae_um"] - gbm_scores["mae_um"]) / gbm_scores["mae_um"]
        gap = abs(deep_scores["mae_um"] - gbm_scores["mae_um"])
        spread = deep_std["mae_um"]

        # Hüküm ölçütü tek yerde: tcm.evaluation.verdict. Saçılım farktan
        # büyükse "kazandı" demek anlamsızdır - sonuç modelden değil, rastgele
        # başlangıçtan geliyor olabilir.
        verdict = seed_stability_verdict(deep_scores["mae_um"], gbm_scores["mae_um"], spread)

        if len(seeds) > 1:
            per_seed = "  ".join(f"{run['mae_um']:.1f}" for run in runs)
            print(f"\nTohum başına CNN+GRU MAE : {per_seed}")
            print(f"Ortalama ± saçılım       : "
                  f"{deep_scores['mae_um']:.2f} ± {spread:.2f} µm")
            print(f"Gradyan artırma ile fark : {gap:.2f} µm")
        else:
            # Tek tohumda saçılım ÖLÇÜLMEDİ, sıfır varsayıldı. Hüküm bu yüzden
            # olduğundan kesin görünür; raporda böyle işaretlenmeli.
            print("\nUYARI: tek tohum - saçılım ölçülmedi, hüküm kesin sayılamaz.")

        print(f"CNN+GRU, gradyan artırmaya göre {change:+.1f}% MAE  ->  {verdict}\n")

        summary["sinav"] = exam
        summary["karar"] = ""
        summary.loc[summary["model"] == "CNN + GRU", "karar"] = verdict
        summary["mae_std"] = 0.0
        summary.loc[summary["model"] == "CNN + GRU", "mae_std"] = spread
        summary["tohum_sayisi"] = len(seeds)
        summary["tohumlar"] = ";".join(str(s) for s in seeds)
        all_rows.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    _verdict(combined)

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 84)
    print("ÇALIŞTIRMA KÜNYESİ")
    print("-" * 84)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)

        combined["git_hash"] = stamp["git_hash"]
        summary_path = target / "model_deep_summary.csv"
        combined = _merge_with_existing(combined, summary_path, set(protocols))
        combined.to_csv(summary_path, index=False)

        # Künye ayrı dosyada: csv'yi okuyan kod sütun beklentisiyle kırılmasın.
        stamp_path = target / "model_deep_summary.provenance.json"
        with stamp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {**stamp, "epochs": args.epochs, "seeds": args.seeds},
                handle, ensure_ascii=False, indent=2,
            )

        print(f"\nKaydedildi: {target / 'model_deep_summary.csv'}")
        print(f"Kaydedildi: {stamp_path}")

    return 0


def _load_signals(config, table, rebuild: bool) -> np.ndarray:
    """Ham sinyalleri (örnek, kanal, zaman) dizisi olarak yükler; önbellekli."""
    cache = config.path("paths.data_processed") / "nasa_signals.npy"
    if cache.exists() and not rebuild:
        print(f"Sinyal önbelleğinden okunuyor: {cache}")
        return np.load(cache)

    print("Ham sinyaller okunuyor...")
    dataset = NASAMilling(config.path("nasa.root"))
    metadata = dataset.metadata()

    frames = []
    for row in table.itertuples(index=False):
        match = metadata[
            (metadata["case"] == row.case) & (metadata["run"] == row.run)
        ]
        frame = stable_region(dataset.signals(int(match["entry"].iloc[0])), keep=0.5)
        frames.append(frame.to_numpy(dtype=np.float32).T)  # (kanal, zaman)

    length = min(frame.shape[1] for frame in frames)
    signals = np.stack([frame[:, :length] for frame in frames])

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, signals)
    print(f"Kaydedildi: {cache}  {signals.shape}")
    return signals


def _run_deep(table, signals, group_column, limit, seed, epochs) -> dict[str, float]:
    fold_rows = []

    for held_out in sorted(table[group_column].unique()):
        train_mask = (table[group_column] != held_out).to_numpy()
        test_mask = ~train_mask

        standardiser = SignalStandardiser()
        x_train = standardiser.fit_transform(signals[train_mask])
        x_test = standardiser.transform(signals[test_mask])

        params = table[PARAMETER_COLUMNS].to_numpy(dtype=np.float32)
        mean = params[train_mask].mean(axis=0)
        std = params[train_mask].std(axis=0)
        std[std < 1e-9] = 1.0
        p_train = (params[train_mask] - mean) / std
        p_test = (params[test_mask] - mean) / std

        y_train = table.loc[train_mask, "vb_um"].to_numpy(dtype=np.float32)
        y_test = table.loc[test_mask, "vb_um"].to_numpy(dtype=np.float32)

        model = CNNGRUWearModel(
            n_channels=signals.shape[1], n_parameters=len(PARAMETER_COLUMNS)
        )
        model = train_model(
            model, x_train, p_train, y_train, epochs=epochs, seed=seed
        )

        # Metrikler KATLAMA BAŞINA hesaplanır, sonra ortalanır.
        # Tüm katlamaları birleştirip tek seferde hesaplamak yanlış olurdu:
        # overshoot ve eşik geçişi metrikleri tek bir takımın sırasını
        # varsayar; birleştirilmiş dizide "ilk geçiş" anlamını kaybeder.
        fold_rows.append(summarise(y_test, predict(model, x_test, p_test), limit))

    return summarise_folds(pd.DataFrame(fold_rows))


def _run_gbm(table, group_column, limit, seed) -> dict[str, float]:
    """Aynı girdilerle gradyan artırma - adil karşılaştırma için."""
    meta = {"case", "run", "vb_um", "condition", "run_time"}
    columns = [c for c in table.columns if c not in meta and table[c].notna().any()]

    fold_rows = []
    for held_out in sorted(table[group_column].unique()):
        train = table[table[group_column] != held_out]
        test = table[table[group_column] == held_out].sort_values("run")

        model = make_gbm_small(random_state=seed)
        model.fit(train[columns], train["vb_um"])

        fold_rows.append(summarise(
            test["vb_um"].to_numpy(),
            np.asarray(model.predict(test[columns]), dtype=float),
            limit,
        ))

    return summarise_folds(pd.DataFrame(fold_rows))


def _merge_with_existing(
    fresh: pd.DataFrame,
    path,
    ran_protocols: set[str],
) -> pd.DataFrame:
    """Çalıştırılmayan sınavların önceki sonuçlarını korur.

    ``--protocols`` ile bir alt küme çalıştırıldığında, kaydetmek diğer
    sınavların satırlarını silmemelidir: dosya "üç sınavın sonucu" gibi
    görünüp aslında birini içermez hale gelirdi.

    Devralınan satırlar KENDİ ``git_hash``'lerini korur. Bu bilinçli: o sayılar
    gerçekten başka bir kodla üretildi ve künyeleri öyle kalmalı. Künyesi
    olmayan eski satırlara "önceki çalıştırma" işareti konur.
    """
    if not path.exists():
        return fresh

    previous = pd.read_csv(path)
    carried = previous[~previous["sinav"].isin(ran_protocols)].copy()
    if carried.empty:
        return fresh

    for column in fresh.columns:
        if column not in carried.columns:
            carried[column] = np.nan

    # Devralınan satırlar açıkça etiketlenir. Boş bırakılırsa okuyan kişi
    # hükmün ne olduğunu bilemez; "GEÇTİ" yazmak ise uydurma olurdu - o
    # çalıştırmada saçılım hiç ölçülmemişti.
    if "git_hash" in carried.columns:
        carried["git_hash"] = carried["git_hash"].fillna("önceki çalıştırma (künyesiz)")
    if "karar" in carried.columns:
        carried["karar"] = carried["karar"].fillna("saçılım ölçülmedi")

    print(f"\nDevralınan sınavlar (bu çalıştırmada yeniden ölçülmedi): "
          f"{sorted(carried['sinav'].unique())}")

    return pd.concat([fresh, carried[fresh.columns]], ignore_index=True)


def _verdict(combined: pd.DataFrame) -> None:
    print("=" * 84)
    print("KARAR")
    print("=" * 84)
    pivot = combined.pivot(index="model", columns="sinav", values="mae_um")
    print("\nMAE (µm):")
    print(pivot.to_string(float_format=lambda v: f"{v:9.2f}"))

    deep = combined[combined["model"] == "CNN + GRU"]
    print("\nHüküm (saçılım ölçütüyle):")
    for row in deep.itertuples(index=False):
        note = (
            f"± {row.mae_std:.2f} µm ({row.tohum_sayisi} tohum)"
            if row.tohum_sayisi > 1
            else "saçılım ÖLÇÜLMEDİ (tek tohum)"
        )
        print(f"  {row.sinav:<14} {row.karar:<10} {note}")

    # KRİTİK: kazanç sayımı ortalama MAE karşılaştırmasıyla değil, saçılım
    # ölçütüyle yapılır. Aksi halde KARARSIZ çıkan sınavlar da geçmiş sayılır -
    # bu hata bu betikte gerçekten vardı.
    counts = count_decisive_wins(deep["karar"])
    print("\n" + describe_wins(counts, "CNN+GRU", "gradyan artırma"))

    if counts["passed"] < counts["total"]:
        print(
            "\nBeklenen sonuç. 145 örnek derin öğrenme için çok küçük bir veri\n"
            "kümesi; ağ öznitelik öğrenmek yerine ezberlemeye yöneliyor.\n"
            "Bu bir bulgudur ve raporlanacaktır."
        )

    if counts["undecided"]:
        print(
            "\nKARARSIZ sınavlar için doğru okuma: 'derin ağ daha iyi' DEĞİL,\n"
            "'bu veri ölçeğinde iki modeli ayırt edecek kanıt yok'. Ayırt etmek\n"
            "için ya daha çok tohum ya daha çok veri gerekir."
        )


if __name__ == "__main__":
    sys.exit(main())
