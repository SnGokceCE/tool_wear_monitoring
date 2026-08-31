"""Faz 06 - teslim edilecek modeli eğit ve paketle.

Bu betik araştırma betiği DEĞİL. ``run_model_b1.py`` "hangi girdi kümesi daha
iyi" sorusunu ölçüyordu; burada o soru cevaplanmış kabul edilir ve tek bir
model üretilir.

ÜÇ TASARIM KARARI
-----------------

1. NİHAİ MODEL 145 SATIRIN TAMAMIYLA EĞİTİLİR
   Çapraz doğrulama katlamaları performansı ÖLÇMEK içindi. Ölçüm bittikten
   sonra veriyi eğitim dışında tutmanın faydası yok - sadece model zayıflar.
   Ama bunun bir bedeli var: bu modelin doğruluğu artık ölçülemez. Raporlanan
   sayılar Faz 04b'nin çapraz doğrulama sayılarıdır ve öyle kalmalıdır.

2. EŞİK NİHAİ MODELDEN DEĞİL, KATLAMA DIŞI TAHMİNLERDEN SEÇİLİR
   Nihai model kendi eğitim verisinin cevabını biliyor; ondan alınan hata
   gerçek dışı küçük çıkar, eşik de yanlış yere oturur. Bu yüzden eşik ayrı
   bir iç çapraz doğrulamayla, Faz 09'un aynı koduyla seçilir
   (``tcm.decision.calibrate_threshold``).

3. İKİ EŞİK KALİBRE EDİLİR, İKİSİ DE PAKETE YAZILIR - AKTİF OLAN ``case``
   case     : takım bazında bölme, 15 katlama. AKTİF.
   material : malzeme bazında bölme, 2 katlama. Pakette durur, aktif değil.

   material kalibrasyonu 156 µm veriyor ve ölçüldüğünde operasyonel olarak
   savunulamaz çıktı: 15/15 takımda alarm ömrün ilk yarısında, 9/15'inde ilk
   geçişte çalıyor. Sebebi hem 2 katlamanın gürültüsü hem de maliyet
   fonksiyonunun takım bazında erken değiştirmeyi hiç ölçmemesi.
   (bkz. scripts/threshold_sweep.py ve README, Faz 06)

    python scripts/train_model.py
    python scripts/train_model.py --feature-set param+time
    python scripts/train_model.py --calibration material   # karşılaştırma için
    python scripts/train_model.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import NASAMilling
from tcm.decision import calibrate_threshold
from tcm.features.build import load_or_build_nasa
from tcm.models.gbm import make_gbm_small
from tcm.provenance import format_stamp, relative_path, run_stamp
from tcm.serving import (
    FEATURE_SET_NAMES,
    FeatureBaselines,
    ModelPackage,
    TrainingCoverage,
    resolve_feature_columns,
)

# Kalibrasyon kurguları: (ad, bölme sütunu, o sütunda gereken en az grup sayısı).
#
# material için min_groups=2 KASITLI. NASA'da iki malzeme var; varsayılan 3 ile
# çalışsaydı koşul sağlanmaz ve bölme takım (case) bazına düşerdi - yani sınav
# "görülmemiş malzeme"den "görülmemiş takım"a dönüşür, kolaylaşır ve eşik
# olduğundan gevşek seçilirdi. Teslim senaryosu tam olarak görülmemiş malzeme.
CALIBRATIONS = (
    ("case", "case", 3),
    ("material", "material", 2),
)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--feature-set", default=None, choices=FEATURE_SET_NAMES)
    parser.add_argument("--calibration", default=None, choices=["case", "material"])
    parser.add_argument("--rebuild", action="store_true",
                        help="öznitelik önbelleğini yenile")
    parser.add_argument("--dry-run", action="store_true",
                        help="eğit ve raporla ama diske yazma")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))
    cost_missed = float(config.get("decision.cost_missed", 5.0))
    cost_false = float(config.get("decision.cost_false_alarm", 1.0))
    search_span = float(config.get("decision.search_span", 0.5))
    consecutive_candidates = (
        tuple(config.get("decision.consecutive_candidates", [1, 2, 3]) or [1])
        if config.get("serving.tune_consecutive", True) else None
    )

    feature_set = args.feature_set or config.get("serving.feature_set", "sensor+param+time")
    calibration = args.calibration or config.get("serving.calibration", "case")

    extraction = {
        "sampling_rate_hz": float(config.get("nasa.sampling_rate_hz")),
        "rpm": float(config.get("nasa.spindle_rpm")),
        "max_order": int(config.get("transfer.max_order", 8)),
        "keep": 0.5,
        "drop_cases": list(config.get("nasa.known_bad_cases", []) or []),
    }

    # ------------------------------------------------------------------ veri
    dataset = NASAMilling(config.path("nasa.root"))
    data = load_or_build_nasa(
        config.path("paths.data_processed") / "nasa_run_features.csv",
        dataset,
        sampling_rate_hz=extraction["sampling_rate_hz"],
        rpm=extraction["rpm"],
        max_order=extraction["max_order"],
        drop_cases=tuple(extraction["drop_cases"]),
        rebuild=args.rebuild,
    )

    feature_columns = resolve_feature_columns(data, feature_set)

    print("\n" + "=" * 84)
    print("TESLİM MODELİ - Model B-1")
    print("=" * 84)
    print(f"Veri       : {len(data)} koşu, {data['case'].nunique()} takım, "
          f"{data['condition'].nunique()} kesme koşulu")
    print(f"Girdi      : {feature_set} ({len(feature_columns)} öznitelik)")
    print(f"Aşınma sın.: {limit:.0f} µm  |  maliyet {cost_missed:.0f}:{cost_false:.0f}")
    print(f"Tohum      : {seed} (sabit)")

    # -------------------------------------------------------------- kalibrasyon
    print("\n" + "-" * 84)
    print("ALARM EŞİĞİ KALİBRASYONU (Faz 09 mantığı, yalnızca eğitim verisiyle)")
    print("-" * 84)

    calibrations: dict[str, object] = {}
    for name, group_column, min_groups in CALIBRATIONS:
        result = calibrate_threshold(
            data, feature_columns,
            lambda: make_gbm_small(random_state=seed),
            group_column=group_column,
            wear_limit_um=limit,
            cost_missed=cost_missed,
            cost_false_alarm=cost_false,
            search_span=search_span,
            consecutive_candidates=consecutive_candidates,
            min_groups=min_groups,
        )
        calibrations[name] = result

        fallback = (
            f"  UYARI: '{result.requested_column}' yerine "
            f"'{result.split_column}' ile bölündü"
            if result.fell_back else ""
        )
        print(f"  {name:<9} eşik {result.threshold:7.1f} µm   k={result.consecutive}   "
              f"{result.n_folds:2d} katlama   iç maliyet {result.cost:5.0f}{fallback}")

    thresholds = {name: c.threshold for name, c in calibrations.items()}
    active = _choose_active(thresholds, calibration)
    consecutive_k = int(calibrations[active].consecutive)

    print(f"\n  Seçim kuralı: {calibration} (açıkça adlandırıldı, otomatik seçim yok)")
    print(f"  ETKİN EŞİK  : {thresholds[active]:.1f} µm  ({active} kalibrasyonu)")
    inactive = [n for n in thresholds if n != active]
    for name in inactive:
        print(f"  (aktif değil : {name} = {thresholds[name]:.1f} µm - "
              "pakette ve künyede duruyor)")

    _explain_threshold(thresholds[active], limit)

    # ------------------------------------------------------------- nihai model
    print("\n" + "-" * 84)
    print("NİHAİ MODEL")
    print("-" * 84)
    print(f"{len(data)} satırın TAMAMIYLA eğitiliyor (katlama ayrılmıyor)...")

    model = make_gbm_small(random_state=seed)
    model.fit(data[feature_columns], data["vb_um"])

    package = ModelPackage(
        model=model,
        feature_set=feature_set,
        feature_columns=feature_columns,
        baselines=FeatureBaselines.from_frame(data, feature_columns),
        coverage=TrainingCoverage.from_frame(data),
        thresholds=thresholds,
        threshold_details={n: c.to_dict() for n, c in calibrations.items()},
        active_threshold=active,
        consecutive_k=consecutive_k,
        wear_limit_um=limit,
        cost_missed=cost_missed,
        cost_false_alarm=cost_false,
        extraction=extraction,
        provenance={
            **(run_stamp(args.config) if args.config else run_stamp()),
            "random_seed": seed,
            "calibration_rule": calibration,
            "feature_cache": relative_path(
                config.path("paths.data_processed") / "nasa_run_features.csv"
            ),
        },
        n_train=len(data),
        training_keys=[
            (float(c), float(r)) for c, r in zip(data["case"], data["run"])
        ],
    )

    print("\n" + package.describe())

    print("\n" + "-" * 84)
    print("ÇALIŞTIRMA KÜNYESİ")
    print("-" * 84)
    print(format_stamp(package.provenance))

    _warn_if_in_sample(package, data, feature_columns)

    # -------------------------------------------------------------------- yaz
    if args.dry_run:
        print("\n--dry-run: diske yazılmadı.")
        return 0

    written = package.save(
        config.path("serving.package_dir"),
        manifest_path=config.path("serving.manifest"),
        baselines_path=config.path("serving.baselines"),
    )

    print("\n" + "-" * 84)
    for label, path in written.items():
        print(f"Kaydedildi ({label:9s}): {path}")
    print("\nÇıkarım için: python scripts/predict.py --from-nasa")
    return 0


def _choose_active(thresholds: dict[str, float], rule: str) -> str:
    """Etkin eşiği seçer - kural açıkça bir kalibrasyonu adlandırmalıdır.

    KALDIRILAN SEÇENEK: ``conservative``, iki kalibrasyonun DÜŞÜĞÜNÜ otomatik
    seçiyordu. Gerekçesi makuldü (kaçırılan aşınma 5 kat pahalı, belirsizlikte
    güvenli taraf düşük eşik) ama sonucu değildi: kural her zaman 2 katlamayla
    hesaplanan gürültülü ``material`` kalibrasyonunu seçiyordu ve o kalibrasyon
    ölçüldüğünde 15/15 takımda erken, 9/15'inde ilk geçişte alarm üretiyordu.

    Ders: "belirsizlikte güvenli tarafı seç" kuralı, güvenli tarafın bedeli
    maliyet fonksiyonunda temsil edilmiyorsa otomatikleştirilemez. Burada
    yanlış alarm geçiş başına sayılıyor, oysa gerçek bedel takım başına.
    """
    if rule not in thresholds:
        raise ValueError(
            f"Bilinmeyen kalibrasyon kuralı: {rule!r}. "
            f"Seçenekler: {', '.join(sorted(thresholds))}"
        )
    return rule


def _explain_threshold(threshold: float, limit: float) -> None:
    """Seçilen eşiğin ne anlama geldiğini açıkça yazar."""
    delta = threshold - limit
    if delta < 0:
        print(f"\n  Yorum: eşik aşınma sınırının {abs(delta):.1f} µm ALTINDA. "
              "Model sistematik olarak\n         eksik tahmin ediyor; karar "
              "kuralı bunu telafi edip güvenli tarafa kayıyor.\n"
              "         Fizikten beklenen davranış.")
    elif delta > 0:
        print(f"\n  DİKKAT: eşik aşınma sınırının {delta:.1f} µm ÜSTÜNDE. "
              "Bu, alarmın sınır\n          aşıldıktan SONRA çalması demek. "
              "Faz 09'da bu belirti bir hataya\n          işaret etmişti "
              "(alarm kilidinin takımlar arasına taşması) - kontrol edin.")


def _warn_if_in_sample(package, data: pd.DataFrame, feature_columns) -> None:
    """Nihai modelin kendi eğitim verisindeki hatasını gösterir ve uyarır.

    Bu sayı raporlanacak bir performans DEĞİL. Amacı tam tersi: örneklem içi
    hatanın çapraz doğrulama hatasından ne kadar düşük çıktığını göstererek
    ikisinin karıştırılmasını engellemek.
    """
    predicted = package.model.predict(data[feature_columns])
    mae = float((data["vb_um"] - predicted).abs().mean())
    print(f"\nÖrneklem içi MAE: {mae:.2f} µm")
    print("  BU BİR PERFORMANS SAYISI DEĞİLDİR - model bu satırların cevabını")
    print("  eğitimde gördü. Raporlanacak sayılar Faz 04b'nin çapraz doğrulama")
    print("  sonuçlarıdır: reports/model_b1_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
