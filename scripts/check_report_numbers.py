"""README'deki her sayıyı reports/ altındaki kayıtlı çıktılarla karşılaştırır.

NEDEN VAR
---------
Bu projede aynı hata sınıfına İKİ KEZ düşüldü:

  1. Faz 04b tablosu, `skew`/`kurtosis` koruması eklenmeden önceki koşudan
     kalma dokuz hücre taşıyordu. Üç gün fark edilmedi.
  2. Faz 04 tablosu doğruydu ama KOD ondan uzaklaşmıştı: `run_model_a.py`
     `run_time`/`cum_time` sütunlarını sessizce öznitelik olarak yutmuştu.

İkisi de gözle bakınca görünmez, çünkü sayılar makul aralıkta kalıyor. Tek
güvenilir yol makineyle karşılaştırmak.

NASIL ÇALIŞIR
-------------
README'deki markdown tabloları ayrıştırılır, her denetim bir tabloyu bir CSV
satır/sütununa bağlar ve değerler toleransla karşılaştırılır. Tolerans,
README'nin yuvarlama hassasiyetinden gelir (genellikle 2 ondalık = 0,005).

Sayılar Türkçe biçimde yazılır (`138,47`), bazı yerlerde U+2212 eksi işareti
(`−60`) kullanılır; ikisi de çözümlenir.

    python scripts/check_report_numbers.py
    python scripts/check_report_numbers.py --verbose

Çıkış kodu 0 = hepsi tutuyor, 1 = en az bir uyuşmazlık var.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tcm import PROJECT_ROOT
from tcm.cli import setup_console

README = PROJECT_ROOT / "README.md"
REPORTS = PROJECT_ROOT / "reports"

# README iki ondalıkla yazıyor; yuvarlama payı.
TOL = 0.005
# Yüzde değerleri tek ondalıkla yazılıyor.
TOL_PCT = 0.05


# --------------------------------------------------------------- ayrıştırma

def parse_number(text: str) -> float | None:
    """Türkçe biçimli bir hücreyi sayıya çevirir; sayı değilse None."""
    cleaned = (
        text.strip()
        .replace("**", "")
        .replace("`", "")
        .replace("−", "-")   # U+2212 matematiksel eksi
        .replace("−", "-")
        .replace("%", "")
        .replace("±", "")
        .replace("µm", "")
        .replace("s", "")
        .strip()
    )
    if not cleaned or cleaned in {"-", "—", "?"}:
        return None
    cleaned = cleaned.replace(".", "").replace(",", ".") if "," in cleaned else cleaned
    cleaned = cleaned.lstrip("+")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_leading_number(text: str) -> float | None:
    """Metne gömülü ilk sayıyı çeker.

    "ömrün %12'si" -> 12   |   "14/15" -> 14   |   "%15" -> 15

    Kesirlerde PAY alınır: README "14/15 takım" derken toplam zaten ayrı
    sütunda (n_takim) duruyor, karşılaştırılan sayı paydır.
    """
    match = re.search(r"-?\d+(?:[.,]\d+)?", text.replace("**", ""))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def split_cells(line: str) -> list[str]:
    """Kaçışsız `|` karakterlerinden böler (`\\|` hücre içinde kalır)."""
    parts, current, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            current += "|"
            i += 2
        elif line[i] == "|":
            parts.append(current)
            current = ""
            i += 1
        else:
            current += line[i]
            i += 1
    parts.append(current)
    return [p.strip() for p in parts[1:-1]] if len(parts) > 2 else []


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    section: str

    def cell(self, row_label, column: str) -> str | None:
        """Satır etiketiyle sütunu kesiştirir.

        ``row_label`` bir dizi ise satırın BAŞTAN o kadar hücresi eşleşmelidir.
        Faz 09 ve Faz 04d tabloları iki anahtarlı: (sınav, kural) ve
        (yöntem, girdi). Tek anahtarla aranırsa yanlış satır bulunur.
        """
        keys = (row_label,) if isinstance(row_label, str) else tuple(row_label)
        index = next(
            (i for i, h in enumerate(self.header) if _norm(h) == _norm(column)),
            None,
        )
        if index is None:
            return None
        for row in self.rows:
            if len(row) <= index or len(row) < len(keys):
                continue
            if all(_norm(row[i]) == _norm(k) for i, k in enumerate(keys)):
                return row[index]
        return None

    def has(self, row_label, column: str) -> bool:
        return self.cell(row_label, column) is not None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "")).strip().lower()


def read_tables(path: Path) -> list[Table]:
    tables: list[Table] = []
    section = ""
    header: list[str] | None = None
    rows: list[list[str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = split_cells(stripped)
            if not cells:
                continue
            if all(set(c) <= set("-: ") and c for c in cells):
                continue                      # hizalama satırı
            if header is None:
                header = cells
            else:
                rows.append(cells)
        else:
            if header is not None:
                tables.append(Table(header, rows, section))
            header, rows = None, []

    if header is not None:
        tables.append(Table(header, rows, section))
    return tables


def find_table(tables: list[Table], row_label, column: str) -> Table | None:
    """İstenen satır ve sütunu birlikte içeren ilk tabloyu bulur."""
    for table in tables:
        if table.has(row_label, column):
            return table
    return None


def _rounding_tolerance(raw: str | None) -> float:
    """Yazılan ondalık basamak sayısının izin verdiği yuvarlama payı.

    "257,6" -> 0,05   |   "138,47" -> 0,005   |   "33" -> 0,5
    """
    if not raw:
        return 0.0
    match = re.search(r"[\d]+,([\d]+)", raw)
    if match:
        return 0.5 * 10 ** (-len(match.group(1))) + 1e-9
    return 0.5 + 1e-9 if re.search(r"\d", raw) else 0.0


# ----------------------------------------------------------------- denetim

@dataclass
class Result:
    checked: int = 0
    failures: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    # Doğrulanan hücreler: (bölüm, satır etiketi, sütun) - kapsam raporu için.
    verified: set = field(default_factory=set)

    def note(self, table: "Table", row_label, column: str) -> None:
        keys = (row_label,) if isinstance(row_label, str) else tuple(row_label)
        self.verified.add((table.section, _norm(keys[0]), _norm(column)))

    def compare(self, name: str, readme: float | None, expected: float,
                tol: float = TOL, raw: str | None = None) -> None:
        if readme is None:
            self.missing.append(f"{name}: README'de bulunamadı (ham: {raw!r})")
            return
        self.checked += 1
        # Tolerans, README'nin YAZDIĞI ondalık sayısından türetilir. Sabit bir
        # tolerans yanlış: "257,6" değeri 257,625'in doğru yuvarlanmışıdır ama
        # 0,005 toleransla hata gibi görünür. Doğru ölçüt, yazılan basamak
        # sayısının izin verdiği yuvarlama payıdır.
        tol = max(tol, _rounding_tolerance(raw))
        if abs(readme - expected) > tol:
            self.failures.append(
                f"{name}: README {readme:g}  vs  kayıt {expected:g}  "
                f"(fark {abs(readme - expected):.4g})"
            )


def check_table(result: Result, tables: list[Table], *, csv: pd.DataFrame,
                key_column: str, value_column: str, readme_column: str,
                label: str, rows: dict[str, str] | None = None,
                tol: float = TOL, scale: float = 1.0, verbose: bool = False,
                parser=parse_number) -> None:
    """CSV'nin bir sütununu README'nin bir sütunuyla satır satır karşılaştırır.

    ``rows`` verilirse {csv_satır_adı: readme_satır_adı} eşlemesi kullanılır;
    verilmezse CSV'deki adlar aynen aranır.
    """
    mapping = rows or {str(v): str(v) for v in csv[key_column]}

    for csv_label, readme_label in mapping.items():
        match = csv[csv[key_column].astype(str) == csv_label]
        if match.empty:
            result.missing.append(f"{label} · {csv_label}: CSV'de yok")
            continue
        expected = float(match[value_column].iloc[0]) * scale

        table = find_table(tables, readme_label, readme_column)
        if table is None:
            result.missing.append(
                f"{label} · {readme_label} / {readme_column}: README tablosu bulunamadı")
            continue

        raw = table.cell(readme_label, readme_column)
        name = f"{label} · {readme_label} · {readme_column}"
        result.compare(name, parser(raw or ""), expected, tol, raw)
        result.note(table, readme_label, readme_column)
        if verbose:
            print(f"    {name:64s} README {raw!r:>12}  kayıt {expected:.4f}")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    tables = read_tables(README)
    result = Result()
    load = lambda name: pd.read_csv(REPORTS / f"{name}.csv")

    print("RAPOR SAYILARI DENETİMİ")
    print("=" * 78)
    print(f"README   : {README.name}  ({len(tables)} tablo ayrıştırıldı)")
    print(f"Kaynaklar: reports/*.csv\n")

    # ---------------------------------------------------- naif taban
    naive = load("naive_baseline")
    for column, readme_column in [
        ("mae_um", "MAE (µm)"), ("rmse_um", "RMSE (µm)"),
        ("crossing_delay_cuts", "Gecikme (geçiş)"),
        ("flute_spread_um", "Ağız saçılımı (µm)"),
    ]:
        check_table(result, tables, csv=naive, key_column="fold",
                    value_column=column, readme_column=readme_column,
                    label="naif taban", verbose=args.verbose)

    # ---------------------------------------------------- Faz 04 Model A
    model_a = load("model_a_summary")
    for column, readme_column in [
        ("mae_um", "MAE (µm)"), ("abs_overshoot_um", "|overshoot| (µm)"),
        ("worst_overshoot_um", "en kötü overshoot (µm)"),
    ]:
        check_table(result, tables, csv=model_a, key_column="model",
                    value_column=column, readme_column=readme_column,
                    label="Model A", verbose=args.verbose)

    channels = load("model_a_channels")
    for column, readme_column in [
        ("oznitelik", "Öznitelik"), ("mae_um", "MAE (µm)"),
        ("abs_overshoot_um", "|overshoot| (µm)"),
    ]:
        check_table(result, tables, csv=channels, key_column="kanal kümesi",
                    value_column=column, readme_column=readme_column,
                    label="Model A kanal", verbose=args.verbose)

    # ---------------------------------------------------- Faz 04b Model B-1
    b1 = load("model_b1_summary")
    names = {
        "0 · naif taban": "naif taban (koşu no)",
        "1 · sadece sensör": "sadece sensör",
        "2 · parametre + süre": "parametre + süre",
        "3 · sensör + parametre": "sensör + parametre",
        "4 · sensör + parametre + süre": "sensör + parametre + süre",
        "5 · sadece parametre (süresiz)": "sadece parametre (süresiz)",
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = b1[b1["sinav"] == exam]
        check_table(result, tables, csv=subset, key_column="model",
                    value_column="mae_um", readme_column=exam,
                    label=f"B-1 {exam}", rows=names, verbose=args.verbose)

    # ---------------------------------------------------- Faz 04b ek analizler
    extras = load("model_b1_extras")
    capped = extras[extras["analiz"] == "vb_kapali_600"]
    check_table(result, tables, csv=capped, key_column="sinav",
                value_column="deger", readme_column="VB ≤ 600 µm",
                label="B-1 VB≤600", verbose=args.verbose)
    check_table(result, tables, csv=capped, key_column="sinav",
                value_column="referans", readme_column="Tam veri",
                label="B-1 tam veri", verbose=args.verbose)

    material = extras[extras["analiz"] == "malzeme_tahmini"]
    check_table(result, tables, csv=material, key_column="sinav",
                value_column="deger", readme_column="Doğruluk",
                label="B-1 malzeme", tol=0.0005, verbose=args.verbose)

    # ---------------------------------------------------- Faz 05 derin öğrenme
    deep = load("model_deep_summary")
    for model, readme_column in [("CNN + GRU", "CNN + GRU"),
                                 ("gradyan artırma", "Gradyan artırma")]:
        subset = deep[deep["model"] == model]
        check_table(result, tables, csv=subset, key_column="sinav",
                    value_column="mae_um", readme_column=readme_column,
                    label=f"Faz 05 {model}", verbose=args.verbose)

    cnn = deep[deep["model"] == "CNN + GRU"]
    check_table(result, tables, csv=cnn, key_column="sinav",
                value_column="mae_std", readme_column="Saçılım",
                label="Faz 05 saçılım", verbose=args.verbose)

    # ---------------------------------------------------- Faz 09 karar kuralı
    # Faz 09 tablosu İKİ anahtarlı: (sınav, kural).
    decision = load("decision_rule_summary")
    rule_names = {"sabit eşik (= sınır)": "sabit (300 µm)", "ayarlı eşik": "ayarlı"}
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = decision[decision["sinav"] == exam]
        rows = {csv_name: (exam, readme_name) for csv_name, readme_name in rule_names.items()}
        for column, readme_column in [
            ("missed_worn", "Kaçırılan"), ("false_alarms", "Yanlış alarm"),
            ("maliyet", "Maliyet"), ("secilen_esik", "Seçilen eşik"),
        ]:
            check_table(result, tables, csv=subset, key_column="kural",
                        value_column=column, readme_column=readme_column,
                        label=f"Faz 09 {exam}", rows=rows, verbose=args.verbose)

    # ---------------------------------------------------- Faz 04c Model B-2
    #
    # README yüzde DEĞİŞİM yazıyor, CSV ham MAE tutuyor; taban "PHM yok".
    b2 = load("model_b2_summary")
    weight_names = {
        "PHM w=0.05": "PHM w=0,05",
        "PHM w=0.15 (eşit toplam)": "PHM w=0,15 (eşit toplam)",
        "PHM w=1.0 (ağırlıksız)": "PHM w=1,0 (ağırlıksız)",
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = b2[b2["sinav"] == exam]
        base = float(subset[subset["model"] == "PHM yok (= B-1)"]["mae_um"].iloc[0])
        for csv_name, readme_name in weight_names.items():
            mae = float(subset[subset["model"] == csv_name]["mae_um"].iloc[0])
            expected = 100.0 * (mae - base) / base
            table = find_table(tables, readme_name, exam)
            if table is None:
                result.missing.append(f"Faz 04c · {readme_name} / {exam}: tablo yok")
                continue
            raw = table.cell(readme_name, exam)
            result.compare(f"Faz 04c · {readme_name} · {exam} (% değişim)",
                           parse_number(raw or ""), expected, TOL_PCT, raw)

        # En kötü overshoot tablosu (ayrı tablo, aynı satır adları)
        for csv_name, readme_name in [("PHM yok (= B-1)", "PHM yok"), *weight_names.items()]:
            worst = float(subset[subset["model"] == csv_name]["worst_overshoot_um"].iloc[0])
            table = find_table(tables, readme_name, exam)
            if table is None or table.cell(readme_name, exam) is None:
                continue
            # İlk tablo yüzde, ikincisi overshoot - overshoot tablosunu ayırt et
            for candidate in tables:
                raw = candidate.cell(readme_name, exam)
                if raw is None:
                    continue
                value = parse_number(raw)
                if value is not None and abs(value - worst) < 0.5:
                    result.checked += 1
                    break

    # ---------------------------------------------------- Faz 04d sınıflandırma
    #
    # İki anahtarlı: (yöntem, girdi).
    classification = load("classification_summary")
    method_names = {
        "0 · naif (koşu no + eşik)": ("naif (koşu no + eşik)", "1"),
        "A · regresyon + eşik": ("A · regresyon + eşik", None),
        "B · doğrudan sınıflandırıcı": ("B · sınıflandırıcı", None),
    }
    for exam in ["vaka-dışı", "koşul-dışı", "malzeme-dışı"]:
        subset = classification[classification["sinav"] == exam]
        for row in subset.itertuples(index=False):
            if row.yöntem not in method_names:
                continue
            readme_method, forced_input = method_names[row.yöntem]
            readme_input = forced_input or str(row.girdi)
            key = (readme_method, readme_input)
            for column, readme_column in [("balanced_acc", exam), ("missed_worn", exam)]:
                table = find_table(tables, key, readme_column)
                if table is None:
                    continue
                raw = table.cell(key, readme_column)
                value = parse_number(raw or "")
                expected = float(getattr(row, column))
                if value is None:
                    continue
                # Aynı (yöntem, girdi) iki tabloda geçiyor: dengeli doğruluk
                # (0-1 arası) ve kaçırılan sayı (tam sayı). Değere göre ayrılır.
                if column == "balanced_acc" and value <= 1.0:
                    result.compare(f"Faz 04d · {readme_method} / {readme_input} · "
                                   f"{exam} dengeli doğruluk", value, expected, 0.0005, raw)
                elif column == "missed_worn" and value > 1.0:
                    result.compare(f"Faz 04d · {readme_method} / {readme_input} · "
                                   f"{exam} kaçırılan", value, expected, TOL, raw)

    # ---------------------------------------------------- eşik taraması
    sweep = load("threshold_sweep")
    sweep = sweep.assign(esik=sweep["esik_um"].map(lambda v: f"{v:.0f}"))
    for column, readme_column in [
        ("missed_worn", "Kaçırılan"), ("false_alarms", "Yanlış alarm"),
    ]:
        check_table(result, tables, csv=sweep, key_column="esik",
                    value_column=column, readme_column=readme_column,
                    label="eşik taraması", verbose=args.verbose)

    # Takım ömrü israfı sütunları: hücreler metin gömülü ("ömrün %12'si") ya
    # da kesirli ("14/15"), o yüzden ayrı çözümleyiciyle okunuyor.
    for column, readme_column, scale, tol in [
        ("omur_orani_ort", "Ort. alarm konumu", 100.0, 0.5),
        ("ilk_yarida_takim", "İlk yarıda", 1.0, TOL),
        ("ilk_geciste_takim", "İlk geçişte", 1.0, TOL),
    ]:
        for row in sweep.itertuples(index=False):
            label = f"{row.esik_um:.0f}"
            table = find_table(tables, label, readme_column)
            if table is None:
                result.missing.append(
                    f"eşik taraması · {label} / {readme_column}: tablo yok")
                continue
            raw = table.cell(label, readme_column)
            result.compare(f"eşik taraması · {label} · {readme_column}",
                           parse_leading_number(raw or ""),
                           float(getattr(row, column)) * scale, tol, raw)
            result.note(table, label, readme_column)

    # ---------------------------------------------------- Faz 12 sabit bölme
    holdout_path = REPORTS / "holdout_split_summary.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path)
        # Belgelerdeki tablo iki anahtarlı: (bölme, model). CSV'de tek
        # sütunda birleşik ("takım bazlı · CNN+GRU"), o yüzden ayrıştırılıyor.
        holdout_rows = {}
        for name in holdout["bolme"]:
            if " · " in name:
                split_name, model_name = name.split(" · ", 1)
            else:
                split_name, model_name = name, "LightGBM"
            holdout_rows[name] = (split_name, model_name)

        for column, readme_column in [
            ("agac", "Ağaç/epoch"), ("esik_um", "Eşik (µm)"),
            ("test_mae_um", "Test MAE"), ("test_rmse_um", "Test RMSE"),
            ("worn_recall", "Yakalama"), ("missed_worn", "Kaçırılan"),
            ("false_alarms", "Yanlış alarm"),
        ]:
            check_table(result, tables, csv=holdout, key_column="bolme",
                        value_column=column, readme_column=readme_column,
                        label="Faz 12 bölme", rows=holdout_rows,
                        verbose=args.verbose)

        sweep_path = REPORTS / "holdout_tree_sweep.csv"
        if sweep_path.exists():
            tree = pd.read_csv(sweep_path)
            tree = tree.assign(agac_etiket=tree["agac"].map(lambda v: f"{v:.0f}"))
            for column, readme_column in [
                ("dogrulama_mae_um", "Doğrulama MAE"),
                ("test_mae_um", "Test MAE"),
            ]:
                check_table(result, tables, csv=tree, key_column="agac_etiket",
                            value_column=column, readme_column=readme_column,
                            label="Faz 12 ağaç taraması", verbose=args.verbose)
    else:
        result.missing.append(
            "Faz 12: holdout_split_summary.csv yok - "
            "`python scripts/run_holdout_split.py --save` çalıştırın")

    # ------------------------------------------------- permütasyon önemi
    perm_path = REPORTS / "permutation_importance.csv"
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        for model in ("CNN+GRU", "LightGBM"):
            subset = perm[perm["model"] == model]
            rows = {r: f"`{r}`" for r in subset["parametre"]}
            check_table(result, tables, csv=subset, key_column="parametre",
                        value_column="onem", readme_column=model,
                        label=f"permütasyon {model}", rows=rows,
                        verbose=args.verbose)

    # ---------------------------------------------------- Faz 02 korelasyon
    #
    # Raporun en güçlü metodolojik iddiası ("0,30 tavanı") buradan geliyor.
    corr_path = REPORTS / "correlation_summary.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        rows = {
            "en güçlü öznitelik (ort. |rho|)": "En güçlü öznitelik",
            "tutarlı yönlü öznitelik sayısı": "Tutarlı yönlü öznitelik sayısı",
        }
        for column, readme_column in [("ham", "Ham Spearman"),
                                      ("egilim_cikarilmis", "Eğilim çıkarılmış")]:
            check_table(result, tables, csv=corr, key_column="olcum",
                        value_column=column, readme_column=readme_column,
                        label="Faz 02 korelasyon", rows=rows,
                        parser=parse_leading_number, verbose=args.verbose)
    else:
        result.missing.append(
            "Faz 02 korelasyon: correlation_summary.csv yok - "
            "`python scripts/explore_phm.py --save` çalıştırın")

    # ---------------------------------------------------- Faz 06 eşikler
    package = json.loads((REPORTS / "model_b1_package.json").read_text(encoding="utf-8"))
    thresholds = package["decision"]["thresholds_um"]
    for name, readme_label in [("case", "`case`"), ("material", "`material`")]:
        table = find_table(tables, readme_label, "Eşik")
        if table is None:
            result.missing.append(f"Faz 06 · {name}: README tablosu bulunamadı")
            continue
        raw = table.cell(readme_label, "Eşik")
        result.compare(f"Faz 06 · {name} eşiği", parse_number(raw or ""),
                       float(thresholds[name]), TOL, raw)

    # ---------------------------------------------------------------- rapor
    print("-" * 78)
    print(f"Karşılaştırılan sayı : {result.checked}")
    print(f"Uyuşmazlık           : {len(result.failures)}")
    print(f"Bulunamayan           : {len(result.missing)}")

    if result.failures:
        print("\n" + "=" * 78)
        print("UYUŞMAZLIKLAR")
        print("=" * 78)
        for line in result.failures:
            print(f"  ✗ {line}")

    if result.missing:
        print("\n" + "=" * 78)
        print("EŞLEŞTİRİLEMEYENLER (denetim kapsamı dışı kaldı)")
        print("=" * 78)
        for line in result.missing:
            print(f"  ? {line}")

    if not result.failures:
        print("\nREADME'deki denetlenen sayıların hepsi kayıtlı çıktılarla tutuyor.")

    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
