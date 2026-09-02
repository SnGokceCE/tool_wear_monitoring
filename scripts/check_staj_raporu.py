"""Staj raporundaki sayıları reports/ altındaki kayıtlı çıktılarla karşılaştırır.

``check_report_numbers.py`` README'yi denetler; bu betik aynı işi teslim edilen
rapor için yapar. İki belgenin tablo başlıkları ve satır adları farklı olduğu
için ayrı tutulmuşlardır; ayrıştırma yardımcıları ortaktır.

    python scripts/check_staj_raporu.py

Çıkış kodu 0 = hepsi tutuyor, 1 = en az bir uyuşmazlık var.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from tcm import PROJECT_ROOT
from tcm.cli import setup_console

REPORTS = PROJECT_ROOT / "reports"
DEFAULT_DOCUMENT = PROJECT_ROOT / "staj_raporu.md"


def _load_helpers():
    """Ayrıştırma yardımcılarını kardeş betikten alır - kopyalamamak için."""
    path = PROJECT_ROOT / "scripts" / "check_report_numbers.py"
    spec = importlib.util.spec_from_file_location("_check_helpers", path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass, sınıfın modülünü sys.modules üzerinden arar; kaydetmeden
    # exec edilirse çözümleme başarısız olur.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    helpers = _load_helpers()
    document = Path(args.file) if args.file else DEFAULT_DOCUMENT
    if not document.exists():
        raise SystemExit(f"Belge yok: {document}")

    tables = helpers.read_tables(document)
    load = lambda name: pd.read_csv(REPORTS / f"{name}.csv")

    checks: list[tuple[str, object, str, float, object]] = []

    def add(label, row, column, expected, parser=None):
        checks.append((label, row, column, float(expected), parser))

    # ------------------------------------------------------------ naif taban
    naive = load("naive_baseline")
    add("naif taban MAE (ortalama)", None, None, naive["mae_um"].mean())

    # -------------------------------------------------------------- Model A
    a = load("model_a_summary").set_index("model")
    rows_a = {
        "naif taban": "0 · naif taban",
        "GBM (ham)": "1 · GBM (ham)",
        "GBM + monoton düzleştirme": "2 · GBM + monoton",
        "GBM + normalize + monoton": "3 · GBM + normalize + monoton",
    }
    for report_row, csv_row in rows_a.items():
        add(f"Model A · {report_row} · MAE", report_row, "MAE (µm)",
            a.loc[csv_row, "mae_um"])
        add(f"Model A · {report_row} · RMSE", report_row, "RMSE (µm)",
            a.loc[csv_row, "rmse_um"])
        add(f"Model A · {report_row} · |overshoot|", report_row, "|overshoot|",
            a.loc[csv_row, "abs_overshoot_um"])
        add(f"Model A · {report_row} · en kötü", report_row, "en kötü overshoot",
            a.loc[csv_row, "worst_overshoot_um"])

    ch = load("model_a_channels").set_index("kanal kümesi")
    rows_ch = {
        "tümü (7 kanal)": "hepsi (7 kanal)",
        "titreşim + AE": "titreşim + AE",
        "yalnızca kuvvet": "sadece kuvvet",
        "yalnızca titreşim": "sadece titreşim",
        "yalnızca AE": "sadece AE",
    }
    for report_row, csv_row in rows_ch.items():
        add(f"kanal · {report_row} · öznitelik", report_row, "Öznitelik",
            ch.loc[csv_row, "oznitelik"])
        add(f"kanal · {report_row} · MAE", report_row, "MAE (µm)",
            ch.loc[csv_row, "mae_um"])

    # ------------------------------------------------------------- Model B-1
    b1 = load("model_b1_summary")
    rows_b1 = {
        "naif taban": "0 · naif taban",
        "yalnızca sensör": "1 · sadece sensör",
        "yalnızca parametre + süre": "2 · parametre + süre",
        "yalnızca parametre": "5 · sadece parametre (süresiz)",
        "sensör + parametre + süre": "4 · sensör + parametre + süre",
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = b1[b1["sinav"] == exam].set_index("model")
        for report_row, csv_row in rows_b1.items():
            add(f"B-1 · {report_row} · {exam}", report_row, exam,
                subset.loc[csv_row, "mae_um"])

    # ------------------------------------------------------------ Model B-2
    b2 = load("model_b2_summary")
    rows_b2 = {
        "w = 0,05": "PHM w=0.05",
        "w = 0,15": "PHM w=0.15 (eşit toplam)",
        "w = 1,00": "PHM w=1.0 (ağırlıksız)",
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = b2[b2["sinav"] == exam].set_index("model")
        base = float(subset.loc["PHM yok (= B-1)", "mae_um"])
        for report_row, csv_row in rows_b2.items():
            mae = float(subset.loc[csv_row, "mae_um"])
            add(f"B-2 · {report_row} · {exam}", report_row, exam,
                100.0 * (mae - base) / base)

    # -------------------------------------------------------- sınıflandırma
    cls = load("classification_summary")
    rows_cls = {
        ("naif (geçiş no + eşik)", "1"): ("0 · naif (koşu no + eşik)", "1"),
        ("regresyon + eşik", "parametre + süre"): ("A · regresyon + eşik", "parametre + süre"),
        ("regresyon + eşik", "sensör"): ("A · regresyon + eşik", "sensör"),
        ("regresyon + eşik", "tümü"): ("A · regresyon + eşik", "hepsi"),
        ("doğrudan sınıflandırıcı", "sensör"): ("B · doğrudan sınıflandırıcı", "sensör"),
        ("doğrudan sınıflandırıcı", "tümü"): ("B · doğrudan sınıflandırıcı", "hepsi"),
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = cls[cls["sinav"] == exam]
        for report_key, (csv_method, csv_input) in rows_cls.items():
            match = subset[(subset["yöntem"] == csv_method) & (subset["girdi"] == csv_input)]
            add(f"sınıflandırma · {report_key[0]}/{report_key[1]} · {exam}",
                list(report_key), exam, float(match["balanced_acc"].iloc[0]))

    # --------------------------------------------------------- karar kuralı
    dec = load("decision_rule_summary")
    # Rapor sabit eşik satırını vaka-dışında "sabit (300)", diğerlerinde
    # "sabit" diye yazıyor; belgede hangisi varsa o denetlenir.
    rows_dec = {"sabit (300)": "sabit eşik (= sınır)", "sabit": "sabit eşik (= sınır)",
                "ayarlı": "ayarlı eşik"}
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = dec[dec["sinav"] == exam].set_index("kural")
        fixed_label = next(
            (name for name in ("sabit (300)", "sabit")
             if any(t.has([exam, name], "Kaçırılan") for t in tables)),
            "sabit",
        )
        for report_rule in [fixed_label, "ayarlı"]:
            csv_rule = rows_dec[report_rule]
            key = [exam, report_rule]
            for column, report_column in [
                ("missed_worn", "Kaçırılan"), ("false_alarms", "Yanlış alarm"),
                ("maliyet", "Maliyet"), ("secilen_esik", "Seçilen eşik"),
            ]:
                add(f"karar · {exam}/{report_rule} · {report_column}",
                    key, report_column, subset.loc[csv_rule, column])

    # ------------------------------------------------------- derin öğrenme
    deep = load("model_deep_summary")
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = deep[deep["sinav"] == exam].set_index("model")
        add(f"derin · {exam} · GBM", exam, "GBM", subset.loc["gradyan artırma", "mae_um"])
        add(f"derin · {exam} · CNN", exam, "CNN + GRU", subset.loc["CNN + GRU", "mae_um"])
        add(f"derin · {exam} · saçılım", exam, "Saçılım", subset.loc["CNN + GRU", "mae_std"])
        add(f"derin · {exam} · fark", exam, "Fark",
            abs(subset.loc["gradyan artırma", "mae_um"] - subset.loc["CNN + GRU", "mae_um"]))

    seeds = deep[(deep["model"] == "CNN + GRU") & deep["tohum_maeleri"].notna()]
    for row in seeds.itertuples(index=False):
        values = [float(v) for v in str(row.tohum_maeleri).split(";")]
        for index, (value, column) in enumerate(
                zip(values, ["Tohum 42", "Tohum 43", "Tohum 44"])):
            add(f"derin tohum · {row.sinav} · {column}", row.sinav, column, value)

    # ---------------------------------------------------------- eşik taraması
    sweep = load("threshold_sweep")
    for row in sweep.itertuples(index=False):
        label = f"{row.esik_um:.0f}"
        add(f"eşik · {label} · kaçırılan", label, "Kaçırılan", row.missed_worn)
        add(f"eşik · {label} · yanlış alarm", label, "Yanlış alarm", row.false_alarms)
        add(f"eşik · {label} · ömür oranı", label, "Ort. alarm konumu",
            row.omur_orani_ort * 100.0, helpers.parse_leading_number)
        add(f"eşik · {label} · ilk geçişte", label, "İlk geçişte alarm",
            row.ilk_geciste_takim, helpers.parse_leading_number)

    # ------------------------------------------------------- Faz 12 bölme
    holdout = load("holdout_split_summary")
    # Rapordaki tablo iki anahtarlı: (bölme, model). CSV'de tek sütunda
    # birleşik ("takım bazlı · CNN+GRU"), o yüzden ayrıştırılıyor.
    for column, report_column in [
        ("agac", "Ağaç/epoch"), ("esik_um", "Eşik (µm)"),
        ("test_mae_um", "Test MAE"), ("test_rmse_um", "Test RMSE"),
        ("worn_recall", "Yakalama"), ("missed_worn", "Kaçırılan"),
        ("false_alarms", "Yanlış alarm"),
    ]:
        for row in holdout.itertuples(index=False):
            if " · " in row.bolme:
                key = list(row.bolme.split(" · ", 1))
            else:
                key = [row.bolme, "LightGBM"]
            add(f"Faz 12 · {row.bolme} · {report_column}",
                key, report_column, getattr(row, column))

    tree = load("holdout_tree_sweep")
    for row in tree.itertuples(index=False):
        label = f"{row.agac:.0f}"
        add(f"Faz 12 ağaç · {label} · doğrulama", label, "Doğrulama MAE",
            row.dogrulama_mae_um)
        add(f"Faz 12 ağaç · {label} · test", label, "Test MAE", row.test_mae_um)

    # ------------------------------------------------------------ korelasyon
    corr = load("correlation_summary").set_index("olcum")
    for report_row, csv_row in [
        ("En güçlü öznitelik", "en güçlü öznitelik (ort. |rho|)"),
        ("Tutarlı yönlü öznitelik", "tutarlı yönlü öznitelik sayısı"),
    ]:
        for column, report_column in [("ham", "Ham"), ("egilim_cikarilmis", "Eğilim çıkarılmış")]:
            add(f"korelasyon · {report_row} · {report_column}", report_row, report_column,
                corr.loc[csv_row, column], helpers.parse_leading_number)

    # ------------------------------------------------------------ Faz 06 eşik
    package = json.loads((REPORTS / "model_b1_package.json").read_text(encoding="utf-8"))
    active = float(package["decision"]["threshold_um"])

    # ------------------------------------------------------ metin içi sayılar
    #
    # Bazı sayılar tabloda değil, cümlenin içinde geçiyor (özet, sonuç bölümü).
    # Tablo denetimi bunları görmez. Beklenen değer CSV'den türetilip metinde
    # aranır: değer değişirse dizge bulunamaz ve uyuşmazlık olarak raporlanır.
    text = document.read_text(encoding="utf-8")
    prose: list[tuple[str, str, str]] = []

    def section_text(heading: str) -> str:
        """Belgenin yalnızca o bölümünü döndürür.

        KRİTİK: arama belge geneline yapılırsa denetim yer belirtemez. Aynı
        sayı birden çok bölümde geçiyorsa (örneğin "en iyi model 0,692" hem
        5.4'te hem 9'da), bir kopyanın bozulması diğeri sayesinde fark
        edilmez. Bu kusur denetimin kendi negatif sınamasında yakalanmıştır.
        """
        start = text.find(heading)
        if start < 0:
            return ""
        level = len(heading) - len(heading.lstrip("#"))
        rest = text[start + len(heading):]
        end = len(rest)
        for candidate in range(level, 0, -1):
            marker = "\n" + "#" * candidate + " "
            position = rest.find(marker)
            if position >= 0:
                end = min(end, position)
        return rest[:end]

    def add_prose(label: str, needle: str, heading: str) -> None:
        prose.append((label, needle, heading))

    b1_mat = b1[b1["sinav"] == "malzeme-dışı"].set_index("model")
    SONUC = "## 9. Sonuç"
    b1_vaka = b1[b1["sinav"] == "vaka-dışı"].set_index("model")

    # Sensör içeren modellerin malzeme-dışı aralığı. Bir taslakta bu aralığın
    # üst ucu ("en iyi model 271,5") yanlışlıkla EN İYİ diye sunulmuştu; en iyi
    # sonuç aslında alt uçtaki sensör kümesidir. Denetim artık iki ucu da
    # ayrı ayrı doğruluyor.
    sensor_rows = ["1 · sadece sensör", "3 · sensör + parametre",
                   "4 · sensör + parametre + süre"]
    best = min(float(b1_mat.loc[r, "mae_um"]) for r in sensor_rows)
    worst = max(float(b1_mat.loc[r, "mae_um"]) for r in sensor_rows)
    add_prose("sonuç · sensörlü en iyi (malzeme-dışı)",
              f"{best:.1f}".replace(".", ","), SONUC)
    add_prose("sonuç · sensörlü en kötü (malzeme-dışı)",
              f"{worst:.1f}".replace(".", ","), SONUC)
    add_prose("sonuç · naif taban (malzeme-dışı)",
              f"{b1_mat.loc['0 · naif taban', 'mae_um']:.2f}".replace(".", ","), SONUC)
    add_prose("sonuç · parametre modeli çöküşü",
              f"{b1_mat.loc['2 · parametre + süre', 'mae_um']:.1f}".replace(".", ","), SONUC)

    # Bozulma oranları: malzeme-dışı / vaka-dışı.
    for row, name in [("1 · sadece sensör", "sensör"),
                      ("4 · sensör + parametre + süre", "sensör+parametre+süre")]:
        ratio = float(b1_mat.loc[row, "mae_um"]) / float(b1_vaka.loc[row, "mae_um"])
        add_prose(f"sonuç · bozulma oranı ({name})",
                  f"{ratio:.2f}×".replace(".", ","), SONUC)

    cls_mat = cls[cls["sinav"] == "malzeme-dışı"]
    naive_acc = float(cls_mat[cls_mat["yöntem"] == "0 · naif (koşu no + eşik)"]["balanced_acc"].iloc[0])
    best_acc = float(cls_mat[cls_mat["yöntem"] != "0 · naif (koşu no + eşik)"]["balanced_acc"].max())
    add_prose("sonuç · sınıflandırma naif", f"naif {naive_acc:.3f}".replace(".", ","), SONUC)
    add_prose("sonuç · sınıflandırma en iyi",
              f"en iyi model {best_acc:.3f}".replace(".", ","), SONUC)

    dec_mat = dec[dec["sinav"] == "malzeme-dışı"].set_index("kural")
    cost_fixed = dec_mat.loc["sabit eşik (= sınır)", "maliyet"]
    cost_tuned = dec_mat.loc["ayarlı eşik", "maliyet"]
    add_prose("sonuç · karar kuralı başabaş",
              f"maliyet {cost_fixed:.0f} → {cost_tuned:.0f}", SONUC)

    material = load("model_b1_extras")
    material = material[material["analiz"] == "malzeme_tahmini"]
    honest = float(material[material["sinav"] == "koşul bazında"]["deger"].iloc[0])
    baseline = float(material[material["sinav"] == "koşul bazında"]["referans"].iloc[0])
    add_prose("kapsam · malzeme tahmini", f"%{100 * honest:.1f}".replace(".", ","),
              "### 6.2 Kapsam-dışı uyarısı")
    add_prose("kapsam · çoğunluk tabanı", f"%{100 * baseline:.1f}".replace(".", ","),
              "### 6.2 Kapsam-dışı uyarısı")

    add_prose("naif taban MAE", f"MAE {naive['mae_um'].mean():.2f} µm".replace(".", ","),
              "### 4.4 Naif taban çizgisi")

    deep_vaka = deep[(deep["sinav"] == "vaka-dışı") & (deep["model"] == "CNN + GRU")]
    add_prose("derin · vaka-dışı ortalama ± saçılım",
              f"{float(deep_vaka['mae_um'].iloc[0]):.2f} ± "
              f"{float(deep_vaka['mae_std'].iloc[0]):.2f} µm".replace(".", ","),
              "### 5.6 Derin öğrenme (1B-CNN + GRU)")

    # -------------------------------------------------------------- çalıştır
    print("STAJ RAPORU SAYILARI DENETİMİ")
    print("=" * 78)
    print(f"Belge    : {document.name}  ({len(tables)} tablo)")
    print(f"Kaynaklar: reports/*.csv\n")

    checked, failures, missing = 0, [], []
    for label, row, column, expected, cell_parser in checks:
        if row is None:
            continue
        table = next((t for t in tables if t.has(row, column)), None)
        if table is None:
            missing.append(f"{label}: tablo/hücre bulunamadı ({row} / {column})")
            continue
        raw = table.cell(row, column)
        value = (cell_parser or helpers.parse_number)(raw or "")
        if value is None:
            missing.append(f"{label}: sayı çözümlenemedi (ham: {raw!r})")
            continue
        checked += 1
        tol = max(0.005, helpers._rounding_tolerance(raw))
        if abs(value - expected) > tol:
            failures.append(f"{label}: rapor {value:g} vs kayıt {expected:g}")
        elif args.verbose:
            print(f"  ✓ {label:58s} {raw!r:>14}  ≈ {expected:.4f}")

    for label, needle, heading in prose:
        checked += 1
        scope = section_text(heading)
        if not scope:
            missing.append(f"{label}: bölüm bulunamadı ({heading})")
            checked -= 1
        elif needle not in scope:
            failures.append(
                f"{label}: {heading} içinde bulunamadı -> beklenen {needle!r}")
        elif args.verbose:
            print(f"  ✓ {label:52s} {heading[:22]:24s} {needle!r}")

    print("-" * 78)
    print(f"Karşılaştırılan : {checked}  (tablo + metin içi)")
    print(f"Uyuşmazlık      : {len(failures)}")
    print(f"Bulunamayan     : {len(missing)}")

    for line in failures:
        print(f"  ✗ {line}")
    for line in missing:
        print(f"  ? {line}")

    if not failures:
        print("\nRapordaki denetlenen sayıların hepsi kayıtlı çıktılarla tutuyor.")
        print(f"(Teslim modelinin etkin eşiği: {active:.0f} µm)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
