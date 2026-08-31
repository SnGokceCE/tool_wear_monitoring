"""Öznitelik çıkarımının çekirdeği - eğitim ve çıkarımın ORTAK kod yolu.

Neden ayrı bir modül
--------------------
Bir modeli üretime aldığınızda en sinsi hata sınıfı şudur: model eğitilirken
öznitelikler bir kodla, tahmin alınırken başka bir kodla hesaplanır. İki kod
başlangıçta aynıdır, sonra biri değişir. Model çökmez, hata vermez - sadece
sessizce yanlış tahmin eder. Literatürde buna *eğitim/servis sapması*
(training-serving skew) denir.

Bu projede risk somuttu: aynı döngü ``build.py`` içinde bir kez, derin öğrenme
betiğinde bir kez daha yazılmıştı. Çıkarım hattı üçüncü kopyayı yazsaydı sapma
kaçınılmazdı.

Çözüm: koşu başına öznitelik üretimi tek bir fonksiyonda toplanır
(``run_feature_row``), tablo kurulumu tek bir fonksiyonda toplanır
(``assemble_feature_table``). Eğitim de çıkarım da bunları çağırır. Bunların
aynı olduğu ``tests/test_serving.py`` içinde test edilir.

Türetilmiş sütunlar da (``cum_time``, ``condition``) buraya taşındı: çıkarımda
elle yeniden hesaplanırlarsa küçük bir tanım farkı (örneğin ``doc`` sayısının
metne çevrilme biçimi) modelin hiç görmediği bir koşul kimliği üretir.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
from tqdm import tqdm

from tcm.features.spectral import order_band_energies
from tcm.features.timedomain import frame_features, stable_region

# Koşu başına kimlik, kesme parametresi ve etiket sütunları - bu SIRAYLA.
# Sıranın korunması önemli: öznitelik tablosunun sütun düzeni değişirse
# önbellekteki csv ile yeni üretilen tablo diff'te farklı görünür ve
# "değişti mi değişmedi mi" sorusu cevapsız kalır.
RUN_META_COLUMNS = ("case", "run", "material", "feed", "doc", "rpm", "run_time", "vb_um")

# Türetilmiş sütunlar - tabloda en sonda dururlar.
DERIVED_COLUMNS = ("cum_time", "condition")


def run_feature_row(
    signals: pd.DataFrame,
    *,
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
) -> dict[str, float]:
    """Tek bir koşunun ham sinyalinden öznitelik sözlüğü.

    Üç adım:
      1. ``stable_region`` - giriş/çıkış kırpılır, ortadaki kararlı bölge kalır
      2. ``frame_features`` - kanal başına zaman alanı öznitelikleri
      3. ``order_band_energies`` - kanal başına mertebe (order) bandı enerjileri

    Anahtarlar ``<kanal>_<öznitelik>`` biçimindedir. Kimlik ve etiket sütunları
    burada YOK - onları ``assemble_feature_table`` ekler.
    """
    frame = stable_region(signals, keep=keep)

    row: dict[str, float] = dict(frame_features(frame))

    for channel in frame.columns:
        bands = order_band_energies(
            frame[channel].to_numpy(),
            sampling_rate_hz=sampling_rate_hz,
            rpm=rpm,
            max_order=max_order,
        )
        row.update({f"{channel}_{name}": value for name, value in bands.items()})

    return row


def condition_id(frame: pd.DataFrame) -> pd.Series:
    """Kesme koşulu kimliği: ``<malzeme>_ap<kesme derinliği>_f<ilerleme>``.

    Koşul-dışı sınavının gruplama anahtarı ve çıkarımdaki kapsam kontrolünün
    dayanağı. Tek yerde tanımlı olması şart: çıkarımda "2_ap1.5_f0.5" yerine
    "2.0_ap1.5_f0.5" üretilirse model, eğitimde gördüğü bir koşulu görülmemiş
    sanır ve boşuna kapsam-dışı uyarısı basar.
    """
    return (
        frame["material"].astype(str)
        + "_ap" + frame["doc"].astype(str)
        + "_f" + frame["feed"].astype(str)
    )


def add_derived_columns(features: pd.DataFrame) -> pd.DataFrame:
    """``cum_time`` ve ``condition`` sütunlarını ekler (yerinde değil, kopyada).

    ``cum_time`` - takımın o ana kadar ne kadar süredir kestiği.
    Faz 04b'de öznitelik öneminde 1. sıraya çıkan girdi budur: kesme
    parametreleri aşınma HIZINI belirler, MİKTARINI değil. Hız x süre = aşınma.

    DİKKAT: kümülatif toplam ``case`` (takım) içinde ve ``run`` sırasına göre
    alınır. Satırların sıralı olduğu varsayılır.
    """
    result = features.copy()
    result["cum_time"] = result.groupby("case")["run_time"].cumsum()
    result["condition"] = condition_id(result)
    return result


def assemble_feature_table(
    runs: pd.DataFrame,
    signal_provider: Callable[[object], pd.DataFrame],
    *,
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
    show_progress: bool = True,
    desc: str = "nasa",
) -> pd.DataFrame:
    """Koşu listesinden tam öznitelik tablosu kurar.

    ``runs``    : koşu başına bir satır; ``RUN_META_COLUMNS`` sütunlarından
                  bulunanlar alınır (``vb_um`` çıkarımda yoktur, olmayabilir).
    ``signal_provider`` : bir ``runs`` satırını alıp o koşunun ham sinyal
                  çerçevesini döndürür. Eğitimde ``mill.mat``'tan, sahada
                  tezgâhtan gelir - kodun geri kalanı ikisini ayırt etmez.

    ``rpm`` kasıtlı olarak ``runs`` sütunundan değil PARAMETREDEN alınır:
    hem tabloya yazılan değer hem mertebe analizinde kullanılan değer tek bir
    kaynaktan gelsin diye. İkisi ayrışırsa öznitelikler sessizce kayar.
    """
    available = [c for c in RUN_META_COLUMNS if c == "rpm" or c in runs.columns]

    rows: list[dict[str, float]] = []
    iterator = tqdm(
        runs.itertuples(index=False),
        total=len(runs),
        desc=desc,
        disable=not show_progress,
    )

    for entry in iterator:
        row: dict[str, float] = {}
        for column in available:
            row[column] = rpm if column == "rpm" else getattr(entry, column)

        row.update(
            run_feature_row(
                signal_provider(entry),
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                max_order=max_order,
                keep=keep,
            )
        )
        rows.append(row)

    features = pd.DataFrame(rows).sort_values(["case", "run"]).reset_index(drop=True)
    return add_derived_columns(features)


def sensor_columns_of(
    columns: Iterable[str],
    meta_columns: Iterable[str] = RUN_META_COLUMNS,
    derived_columns: Iterable[str] = DERIVED_COLUMNS,
) -> list[str]:
    """Bir tablodaki sensör öznitelik sütunları: kimlik/parametre/etiket dışı olanlar."""
    excluded = set(meta_columns) | set(derived_columns)
    return [c for c in columns if c not in excluded]
