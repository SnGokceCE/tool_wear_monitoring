"""Faz 06 - çıkarım hattı: kaydedilmiş paketten tahmin al.

Bu betiğin asıl işlevi tahmin basmak değil, ÇIKARIM YOLUNU GÖSTERMEK. Sahada
Siemens 840D'den gelecek sinyal de aynı yoldan geçecek:

    ham sinyal + kesme parametreleri
      -> tcm.features.extract.assemble_feature_table   (EĞİTİMLE AYNI KOD)
      -> ModelPackage.predict                          (kaydedilmiş öznitelik
                                                        listesi ve eşikle)
      -> VB tahmini + worn/unworn + kapsam uyarısı

``--from-nasa`` seçeneği öznitelikleri önbellekten OKUMAZ, ham ``mill.mat``
sinyallerinden YENİDEN ÜRETİR. Kasıtlı: eğitim ve çıkarımın gerçekten aynı
kodu kullandığı ancak böyle gösterilebilir. Önbelleği okumak bu kanıtı
atlardı.

    python scripts/predict.py --from-nasa
    python scripts/predict.py --from-nasa --only-alarms
    python scripts/predict.py --from-nasa --simulate-unseen-material
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import NASAMilling, nasa_run_table
from tcm.features.extract import add_derived_columns, assemble_feature_table
from tcm.serving import ModelPackage


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--package", default=None,
                        help="paket klasörü (öntanımlı: config serving.package_dir)")
    parser.add_argument("--from-nasa", action="store_true",
                        help="NASA ham sinyallerinden öznitelik üretip tahmin al")
    parser.add_argument("--limit", type=int, default=None,
                        help="yalnızca ilk N koşu (hızlı deneme)")
    parser.add_argument("--only-alarms", action="store_true",
                        help="yalnızca alarm veren satırları bas")
    parser.add_argument("--simulate-unseen-material", action="store_true",
                        help="malzemeyi eğitimde olmayan bir koda çevirip "
                             "kapsam dışı uyarısını gösterir (alüminyum senaryosu)")
    parser.add_argument("--save", default=None, help="sonuçları bu csv'ye yaz")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    package_dir = args.package or config.path("serving.package_dir")
    package = ModelPackage.load(package_dir)

    print("=" * 100)
    print("ÇIKARIM")
    print("=" * 100)
    print(package.describe())
    print(f"Paket      : {package_dir}")
    print(f"Üretim     : {package.created_at}  |  git {package.provenance['git_hash'][:10]}")

    if not args.from_nasa:
        raise SystemExit(
            "Bir veri kaynağı seçin. Şu an yalnızca --from-nasa destekleniyor.\n"
            "Saha kaynağı (Siemens 840D) yazılmadı - tezgâh erişimi yok; "
            "yalnızca giriş arayüzü sözleşmesi tanımlı."
        )

    features = _extract_from_nasa(config, package, args.limit)

    if args.simulate_unseen_material:
        # Alüminyum senaryosunun benzetimi: eğitimde 1 (dökme demir) ve
        # 2 (çelik) var. 4 kodunu vererek "hiç görülmemiş malzeme" durumunu
        # üretiyoruz. Sinyaller gerçek, etiket sahte - amaç kapsam
        # kontrolünün çalıştığını göstermek.
        features = features.copy()
        features["material"] = 4.0
        # Koşul kimliği elle kurulmaz - eğitimdeki tanımın aynısı kullanılır.
        features = add_derived_columns(features)
        print("\nBENZETİM: malzeme kodu 4 (eğitimde yok) yapıldı - "
              "alüminyum senaryosunun karşılığı.")

    result = package.predict(features)

    _report(result, args.only_alarms)

    if args.save:
        result.to_csv(args.save, index=False)
        print(f"\nKaydedildi: {args.save}")

    return 0


def _extract_from_nasa(config, package, limit: int | None) -> pd.DataFrame:
    """NASA ham sinyallerinden öznitelik tablosu - EĞİTİMDEKİ KODLA.

    Öznitelik çıkarım ayarları (örnekleme hızı, devir, mertebe sayısı)
    config'den değil PAKETTEN okunur. Sebebi önemli: config değişebilir, ama
    model belirli ayarlarla üretilmiş özniteliklerle eğitildi. Çıkarımda başka
    bir ayar kullanmak modeli sessizce bozar.
    """
    dataset = NASAMilling(config.path("nasa.root"))

    # Ham metadata -> ortak koşu şeması. Eğitimin kullandığı fonksiyonun
    # AYNISI (tcm.datasets.nasa.run_table): VB'nin mikrometreye çevrilmesi
    # gibi dönüşümler de sözleşmenin parçası, çıkarımda yeniden yazılamaz.
    runs = nasa_run_table(
        dataset.metadata(),
        drop_cases=tuple(package.extraction.get("drop_cases", [])),
    )
    if limit:
        runs = runs.head(limit)

    print(f"\n{len(runs)} koşunun öznitelikleri ham sinyalden üretiliyor "
          "(eğitimle aynı kod)...")

    return assemble_feature_table(
        runs,
        lambda entry: dataset.signals(int(entry.entry)),
        sampling_rate_hz=package.extraction["sampling_rate_hz"],
        rpm=package.extraction["rpm"],
        max_order=package.extraction["max_order"],
        keep=package.extraction["keep"],
        desc="çıkarım",
    )


def _report(result: pd.DataFrame, only_alarms: bool) -> None:
    in_sample = int(result["in_sample"].sum())
    out_of_scope = int(result["out_of_scope"].sum())
    advisory = int((result["advisory"] != "").sum())

    print("\n" + "=" * 100)
    print(f"SONUÇ: {len(result)} koşu  |  alarm {int(result['worn'].sum())}  |  "
          f"kapsam dışı {out_of_scope}  |  aralık uyarısı {advisory}")
    print("=" * 100)

    if in_sample:
        print(
            f"\n!! ÖRNEKLEM İÇİ UYARISI: {in_sample}/{len(result)} satır nihai\n"
            "   modelin EĞİTİM verisindeydi. Model bu satırların cevabını gördü;\n"
            "   buradaki hata gerçek saha hatasını TEMSİL ETMEZ, iyimserdir.\n"
            "   Bu çalıştırma hattın çalıştığını gösterir, performansını ÖLÇMEZ.\n"
            "   Performans sayıları için Faz 04b çapraz doğrulama sonuçlarına bakın."
        )

    shown = result[result["worn"]] if only_alarms else result
    columns = [c for c in ("case", "run", "material", "cum_time", "vb_true_um",
                           "vb_pred_um", "worn", "out_of_scope") if c in shown.columns]

    print("\nİlk 15 satır:")
    print(shown[columns].head(15).to_string(index=False,
                                            float_format=lambda v: f"{v:9.1f}"))

    if out_of_scope:
        reasons = result.loc[result["out_of_scope"], "out_of_scope_reason"]
        print(f"\nKAPSAM DIŞI - {out_of_scope} satır. Gerekçeler:")
        for reason, count in reasons.value_counts().items():
            print(f"  {count:4d}x  {reason}")
        print(
            "\n  Bu satırlarda tahmin BASILDI ama güvenilmez. Faz 04b'de ölçüldü:\n"
            "  görülmemiş malzemede parametre tabanlı model naif tabanın bile\n"
            "  altına düşüyor. Operatör bu uyarıyı gördüğünde tahmine değil,\n"
            "  takım ömrü sayacına ve kendi tecrübesine güvenmelidir."
        )

    if advisory:
        notes = result.loc[result["advisory"] != "", "advisory"]
        print(f"\nARALIK UYARISI - {advisory} satır (kapsam dışı değil, "
              "ama ekstrapolasyon):")
        for note, count in notes.value_counts().head(5).items():
            print(f"  {count:4d}x  {note}")


if __name__ == "__main__":
    sys.exit(main())
