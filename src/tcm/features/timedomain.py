"""Zaman alanı öznitelikleri.

Aşınma arttıkça kesme kuvvetleri ve titreşim enerjisi yükselir; bu değişimi
yakalayan en basit büyüklükler burada. Faz 04'te gradyan artırma modelinin
girdisi olacaklar, Faz 02'de ise aşınmayla ilişkiyi görmek için kullanılıyorlar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_NAMES = ("rms", "std", "peak", "p2p", "skew", "kurtosis", "abs_mean")


def time_domain_features(values: np.ndarray) -> dict[str, float]:
    """Tek kanaldan zaman alanı öznitelikleri."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return {name: float("nan") for name in FEATURE_NAMES}

    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "std": float(np.std(values)),
        "peak": float(np.max(np.abs(values))),
        "p2p": float(np.ptp(values)),
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "abs_mean": float(np.mean(np.abs(values))),
    }


def frame_features(frame: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    """Çok kanallı bir çerçevenin tüm kanalları için öznitelikler.

    Sonuç anahtarları ``<kanal>_<öznitelik>`` biçiminde düzleştirilir.
    """
    result: dict[str, float] = {}
    for channel in frame.columns:
        for name, value in time_domain_features(frame[channel].to_numpy()).items():
            result[f"{prefix}{channel}_{name}"] = value
    return result


def stable_region(frame: pd.DataFrame, keep: float = 0.5) -> pd.DataFrame:
    """Geçişin ortasındaki kararlı bölgeyi döndürür.

    Kesme dosyaları takımın havada olduğu giriş ve çıkış kısımlarını da içerir;
    bu bölgelerde sinyal aşınma hakkında bilgi taşımaz ve öznitelikleri
    sulandırır. Ortadaki ``keep`` oranı alınarak giriş/çıkış atılır.

    Bu basit bir kırpma; Faz 04'te eşik tabanlı gerçek kesme tespiti ile
    değiştirilebilir. Şimdilik kasıtlı olarak sade tutuluyor.
    """
    if not 0 < keep <= 1:
        raise ValueError(f"keep 0 ile 1 arasında olmalı, {keep} verildi")

    n = len(frame)
    if n == 0:
        return frame

    margin = int(n * (1 - keep) / 2)
    return frame.iloc[margin : n - margin] if margin else frame
