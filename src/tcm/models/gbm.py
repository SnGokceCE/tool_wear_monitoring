"""Gradyan artırma modeli (LightGBM).

Nasıl çalışır: tek bir büyük karar ağacı yerine yüzlerce küçük ağaç kurulur ve
her yeni ağaç önceki ağaçların HATASINI düzeltmeye çalışır. Sonuç, yüzlerce
küçük düzeltmenin toplamıdır.

Neden bu seçildi: elimizde etkin olarak 2 bağımsız aşınma yörüngesi var
(kesici bazında bölünce eğitimde 2 kesici kalıyor). Bu ölçekte derin ağ eğrinin
şeklini ezberler. Ağaç tabanlı model "şu öznitelik şu eşiğin üstündeyse"
biçiminde parça parça öğrendiği için daha dirençli - ve hangi özniteliği ne
kadar kullandığını raporlayabiliyor.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor

# Mantık: az sayıda bağımsız yörünge var, o yüzden modelin kapasitesi kısıtlı
# tutuluyor. Küçük yapraklar (num_leaves), yaprak başına yüksek asgari örnek
# (min_child_samples) ve düşük öğrenme hızı (learning_rate) ezberlemeyi
# zorlaştırır. Ağaç sayısı buna karşılık yüksek tutulur.
DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "verbose": -1,
}

# SIZINTI NOTU
# ------------
# Hiperparametreleri test katlamalarında deneyip en iyisini seçmek, test
# kümesinin bilgisini modele sızdırır ve raporlanan hatayı gerçekdışı düşürür.
# Elimizde yalnızca 3 kesici olduğu için iç içe çapraz doğrulama da anlamlı
# değil (iç katlamada tek kesici kalır). Bu yüzden parametreler sabit ve
# muhafazakâr tutuldu; raporda böyle belirtilecek.


# Küçük veri için ayrı set. NASA'da ~100-130 eğitim satırı var; Model A'nın
# 630 satır için seçilmiş ayarları burada modeli felç ediyor:
# min_child_samples=30 ile 100 satırlık veride ağaç neredeyse hiç dallanamaz.
#
# Bu değerler de test sonuçlarına BAKILARAK seçilmedi. Kural: yaprak başına
# asgari örnek ~ eğitim satırının %5'i, yaprak sayısı buna göre küçük.
SMALL_DATA_PARAMS = {
    **DEFAULT_PARAMS,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 5,
    "colsample_bytree": 0.5,
}


def make_gbm_small(random_state: int = 42, **overrides) -> LGBMRegressor:
    """Küçük veri kümeleri için model (NASA: ~145 satır)."""
    params = {**SMALL_DATA_PARAMS, **overrides, "random_state": random_state}
    return LGBMRegressor(**params)


def make_gbm(random_state: int = 42, **overrides) -> LGBMRegressor:
    """Varsayılan ayarlarla yeni bir model üretir.

    Her katlamada YENİ model gerekir; aynı modeli yeniden eğitmek önceki
    katlamanın öğrendiklerini taşır.
    """
    params = {**DEFAULT_PARAMS, **overrides, "random_state": random_state}
    return LGBMRegressor(**params)


def feature_importance(model: LGBMRegressor, feature_names) -> "pd.DataFrame":
    """Modelin hangi özniteliği ne kadar kullandığı.

    UYARI: birbiriyle güçlü ilişkili öznitelikler önemi paylaşır. force_x_std
    ile force_x_order_total neredeyse aynı şeyi ölçüyorsa, model birini seçip
    diğerini görmezden gelebilir. Düşük önem "bu öznitelik işe yaramaz"
    demek değildir; "modelin ihtiyacı olan bilgiyi başka bir sütundan aldı"
    demek olabilir.
    """
    import pandas as pd

    return (
        pd.DataFrame({
            "oznitelik": list(feature_names),
            "onem": model.feature_importances_,
        })
        .sort_values("onem", ascending=False)
        .reset_index(drop=True)
    )


def channel_of(feature_name: str, channels) -> str | None:
    """Öznitelik isminden kanalı çıkarır (``force_x_rms`` -> ``force_x``)."""
    for channel in channels:
        if feature_name.startswith(f"{channel}_"):
            return channel
    return None


def select_channels(feature_names, channels) -> list[str]:
    """Yalnızca verilen kanallara ait öznitelikleri seçer.

    Faz 08'in (sensör azaltma) ve Model B'nin (kuvvet yok) temel aracı.
    """
    channels = tuple(channels)
    return [
        name for name in feature_names
        if any(name.startswith(f"{channel}_") for channel in channels)
    ]
