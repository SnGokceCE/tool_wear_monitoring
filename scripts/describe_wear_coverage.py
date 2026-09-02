"""Eğitim verisi aşınma aralığını ne kadar kapsıyor? (Faz 12 teşhisi)

NEDEN VAR
---------
Faz 12'de derin model, kümülatif süresi SIFIR olan yepyeni bir takıma
188 µm aşınma dedi. "Süre en önemli parametre" diye ölçmüştük; o hâlde
süre sıfırken tahmin neden sıfıra yakın değil?

Cevap modelde değil, VERİDE. Bu betik onu gösteriyor: eğitim kümesinde
"yeni takım" bölgesi neredeyse temsil edilmiyor. Model az gördüğü bölgede
risk almıyor, ortalamaya doğru kaçıyor.

Bu, bir model kusuru değil örneklem kusurudur ve düzeltmesi de veri
tarafındadır - hedef tezgâhtan veri toplanırken takım ömrünün BAŞINDAN
sonuna dengeli örnekleme yapmak gerekir.

    python scripts/describe_wear_coverage.py
    python scripts/describe_wear_coverage.py --save
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
import pandas as pd

from tcm import PROJECT_ROOT, load_config
from tcm.cli import setup_console
from tcm.provenance import format_stamp, run_stamp

# Aşınma bantları. Sınır (300 µm) ve onun altındaki çalışma bölgesi ayrı
# ayrı görünsün diye eşit aralıklı değil.
BANDS = [(0, 0), (1, 50), (51, 100), (101, 200), (201, 300),
         (301, 500), (501, 800), (801, 2000)]


def _load_split_module():
    path = PROJECT_ROOT / "scripts" / "run_holdout_split.py"
    spec = importlib.util.spec_from_file_location("_holdout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    limit = float(config.get("nasa.wear_limit_um", 300))
    data = pd.read_csv(
        config.path("paths.data_processed") / "nasa_run_features.csv"
    ).sort_values(["case", "run"]).reset_index(drop=True)

    split_module = _load_split_module()
    parts = split_module._split_by_tool(data)

    print("=" * 78)
    print("AŞINMA ARALIĞI KAPSAMI")
    print("=" * 78)
    print(f"Aşınma sınırı: {limit:.0f} µm  |  toplam {len(data)} koşu\n")

    rows = _band_table(data, parts)
    table = pd.DataFrame(rows)
    _print_bands(table)
    _print_new_tool_region(data, parts)
    _print_time(data, parts)
    _print_reading(data, parts)

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 78)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        out = target / "wear_coverage.csv"
        table.assign(git_hash=stamp["git_hash"]).to_csv(out, index=False)
        print(f"\nKaydedildi: {out}")

    return 0


def _band_table(data, parts) -> list[dict]:
    rows = []
    for low, high in BANDS:
        mask = (data["vb_um"] >= low) & (data["vb_um"] <= high)
        row = {
            "bant": "VB = 0" if high == 0 else f"{low}-{high} µm",
            "tumu": int(mask.sum()),
        }
        for name, part in parts.items():
            part_mask = (part["vb_um"] >= low) & (part["vb_um"] <= high)
            row[name] = int(part_mask.sum())
        rows.append(row)
    return rows


def _print_bands(table: pd.DataFrame) -> None:
    view = table.copy()
    total_train = view["eğitim"].sum()
    view["eğitim %"] = (100 * view["eğitim"] / total_train).round(1)
    print("Aşınma bandına göre koşu sayısı:")
    print(view.to_string(index=False))


def _print_new_tool_region(data, parts) -> None:
    train = parts["eğitim"]
    print("\n" + "-" * 78)
    print("YENİ TAKIM BÖLGESİ - eğitimde ne kadar temsil ediliyor?")
    print("-" * 78)
    for threshold in (0, 50, 100, 200):
        n = int((train["vb_um"] <= threshold).sum())
        bar = "█" * max(1, round(n / 2))
        print(f"  VB ≤ {threshold:4d} µm : {n:3d}/{len(train)} koşu  "
              f"(%{100 * n / len(train):4.1f})  {bar}")

    zero_time = int((train["cum_time"] == 0).sum())
    print(f"\n  Kümülatif süre = 0 olan koşu: {zero_time}/{len(train)}")
    if zero_time <= 2:
        print("  -> Model 'yepyeni takım' durumunu neredeyse hiç görmedi.")


def _print_time(data, parts) -> None:
    train = parts["eğitim"]
    print("\n" + "-" * 78)
    print("KÜMÜLATİF SÜRE DAĞILIMI (eğitim)")
    print("-" * 78)
    print(f"  ortalama {train['cum_time'].mean():7.1f}  |  "
          f"ortanca {train['cum_time'].median():7.1f}  |  "
          f"maks {train['cum_time'].max():7.1f}")


def _print_reading(data, parts) -> None:
    train = parts["eğitim"]
    mean_wear = float(train["vb_um"].mean())
    median_wear = float(train["vb_um"].median())
    low = int((train["vb_um"] <= 50).sum())

    print("\n" + "=" * 78)
    print("OKUMA")
    print("=" * 78)
    print(f"""
Eğitim kümesinin aşınma ortalaması {mean_wear:.0f} µm, ortancası {median_wear:.0f} µm.
Yani veri, takım ömrünün İLERİ bölgesine yığılmış durumda - deneyi yapanlar
takımın ömrünü ölçmek istemiş, ilgilendikleri yer aşınmanın ilerlediği bölge.

Yeni takım bölgesi (VB ≤ 50 µm) eğitimde yalnızca {low}/{len(train)} koşu.
Model bu bölgeyi neredeyse hiç görmediği için orada risk almıyor ve
tahminini ortalamaya doğru çekiyor.

Bu bir MODEL kusuru değil, ÖRNEKLEM kusurudur. Kararı da bozmuyor: yeni
takıma 188 µm demek yanlış ama alarm eşiğinin altında kaldığı için "sağlam"
kararı doğru çıkıyor.

Düzeltmesi veri tarafındadır: hedef tezgâhtan veri toplanırken takım
ömrünün başından sonuna DENGELİ örnekleme yapılmalı. Rapordaki gelecek
çalışma maddesinin gerekçelerinden biri budur.""")


if __name__ == "__main__":
    sys.exit(main())
