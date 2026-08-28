"""PHM ve NASA için ortak öznitelik uzayı (Model B-2).

İki veri setini tek tabloda birleştirmenin üç engeli var:

1. KANAL İSİMLERİ EŞLEŞMİYOR
   PHM'de titreşim üç EKSEN (vib_x/y/z), NASA'da iki KONUM (vib_table,
   vib_spindle). Bunlar aynı fiziksel ölçüm değil, isim eşlemesi yapılamaz.
   Çözüm: mantıksal kanal grupları tanımlayıp grup içindeki kanalların
   özniteliklerini özetlemek (ortalama ve maksimum). Kaç fiziksel kanal
   olduğundan bağımsız olarak sabit sayıda sütun üretir.

2. MUTLAK GENLİKLER KARŞILAŞTIRILAMAZ
   Farklı sensörler, farklı kazançlar, farklı bağlantı. Çözüm: her takımın
   özniteliklerini kendi ilk geçişlerine göre normalize etmek (bkz. normalise).

3. PARAMETRE BİRİMLERİ FARKLI
   PHM ilerlemeyi mm/dk, NASA mm/dev veriyor. Ortak birime çevriliyor.

Kuvvet (yalnız PHM) ve motor akımı (yalnız NASA) düşer - tek veri setinde
bulunan bir kanal ortak uzayda kullanılamaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mantıksal kanal grupları: her veri setinde hangi fiziksel kanallara karşılık geldiği.
CHANNEL_GROUPS = {
    "phm2010": {
        "vibration": ("vib_x", "vib_y", "vib_z"),
        "acoustic": ("ae_rms",),
    },
    "nasa": {
        "vibration": ("vib_table", "vib_spindle"),
        "acoustic": ("AE_table", "AE_spindle"),
    },
}

# PHM sabitleri - kesme parametresi sütunlarını doldurmak için.
# Malzeme kodları: 1 = dökme demir, 2 = çelik (NASA), 3 = paslanmaz (PHM).
PHM_MATERIAL_CODE = 3
PHM_FEED_MM_PER_MIN = 1555.0
PHM_DOC_MM = 0.2

PARAMETER_COLUMNS = ["material", "feed_mm_per_rev", "doc_mm", "rpm"]
TIME_COLUMN = "cum_time"


def feature_suffixes(columns: list[str], channels: tuple[str, ...]) -> list[str]:
    """Bir kanal grubunun sütunlarından öznitelik son eklerini çıkarır."""
    suffixes: set[str] = set()
    for column in columns:
        for channel in channels:
            prefix = f"{channel}_"
            if column.startswith(prefix):
                suffixes.add(column[len(prefix):])
    return sorted(suffixes)


def collapse_channel_groups(
    data: pd.DataFrame,
    groups: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Kanal gruplarını özet sütunlara indirger.

    ``vibration`` grubu için ``vib_x_rms``, ``vib_y_rms``, ``vib_z_rms``
    sütunları ``vibration_rms_mean`` ve ``vibration_rms_max`` olur.

    Ortalama grubun tipik davranışını, maksimum ise en kötü kanalı temsil eder;
    ikisi birlikte tek bir özete göre daha fazla bilgi taşır.
    """
    columns = list(data.columns)
    result = pd.DataFrame(index=data.index)

    for group, channels in groups.items():
        present = [c for c in channels if any(col.startswith(f"{c}_") for col in columns)]
        if not present:
            continue

        for suffix in feature_suffixes(columns, tuple(present)):
            members = [f"{c}_{suffix}" for c in present if f"{c}_{suffix}" in columns]
            if not members:
                continue
            block = data[members]
            result[f"{group}_{suffix}_mean"] = block.mean(axis=1)
            result[f"{group}_{suffix}_max"] = block.max(axis=1)

    return result


def prepare_phm(features: pd.DataFrame) -> pd.DataFrame:
    """PHM tablosunu ortak uzaya çevirir."""
    sensors = collapse_channel_groups(features, CHANNEL_GROUPS["phm2010"])
    rpm = 10400.0

    meta = pd.DataFrame({
        "source": "phm",
        "tool": "phm_" + features["cutter"].astype(str),
        "step": features["cut"].to_numpy(),
        "vb_um": features["vb_um"].to_numpy(),
        TIME_COLUMN: features[TIME_COLUMN].to_numpy(),
        "material": PHM_MATERIAL_CODE,
        # mm/dk -> mm/dev: NASA ile aynı birim.
        "feed_mm_per_rev": PHM_FEED_MM_PER_MIN / rpm,
        "doc_mm": PHM_DOC_MM,
        "rpm": rpm,
        # Koşul kimliği - PHM tek koşul olduğu için tek değer.
        "condition": "phm_paslanmaz",
    }, index=features.index)

    return pd.concat([sensors, meta], axis=1)


def prepare_nasa(features: pd.DataFrame) -> pd.DataFrame:
    """NASA tablosunu ortak uzaya çevirir."""
    sensors = collapse_channel_groups(features, CHANNEL_GROUPS["nasa"])

    meta = pd.DataFrame({
        "source": "nasa",
        "tool": "nasa_" + features["case"].astype(int).astype(str),
        "step": features["run"].to_numpy(),
        "vb_um": features["vb_um"].to_numpy(),
        TIME_COLUMN: features[TIME_COLUMN].to_numpy(),
        "material": features["material"].to_numpy(),
        "feed_mm_per_rev": features["feed"].to_numpy(),
        "doc_mm": features["doc"].to_numpy(),
        "rpm": features["rpm"].to_numpy(),
        "condition": features["condition"].to_numpy(),
    }, index=features.index)

    return pd.concat([sensors, meta], axis=1)


def build_shared_table(
    phm_features: pd.DataFrame,
    nasa_features: pd.DataFrame,
    normalise: bool = True,
    n_baseline: int = 5,
) -> pd.DataFrame:
    """İki veri setini ortak uzayda birleştirir.

    ``normalise=True`` ise her takımın öznitelikleri kendi ilk ``n_baseline``
    geçişine göre ölçeklenir. Bu, tezgâhlar arası kazanç farkını yok eder ve
    birleştirmenin ön koşuludur.

    NASA'da vaka başına 3-20 koşu var, o yüzden taban çizgisi PHM'deki gibi
    10 değil 5 geçişten hesaplanıyor.
    """
    from tcm.features.normalise import baseline_normalise

    phm = prepare_phm(phm_features)
    nasa = prepare_nasa(nasa_features)

    common = [c for c in phm.columns if c in nasa.columns]
    combined = pd.concat([phm[common], nasa[common]], ignore_index=True)

    meta = {"source", "tool", "step", "vb_um", "condition", *PARAMETER_COLUMNS, TIME_COLUMN}
    sensor_columns = [c for c in common if c not in meta]

    # Bir veri setinde tamamen eksik olan sütunlar (Nyquist üstü mertebeler)
    sensor_columns = [c for c in sensor_columns if combined[c].notna().any()]

    if normalise:
        combined = baseline_normalise(
            combined, sensor_columns, group_column="tool",
            sort_column="step", n_baseline=n_baseline,
        )

    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined


def sensor_columns_of(table: pd.DataFrame) -> list[str]:
    """Ortak tablodaki sensör öznitelik sütunları."""
    meta = {"source", "tool", "step", "vb_um", "condition", *PARAMETER_COLUMNS, TIME_COLUMN}
    return [c for c in table.columns if c not in meta]
