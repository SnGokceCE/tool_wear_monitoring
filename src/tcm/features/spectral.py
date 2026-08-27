"""Frekans alanı öznitelikleri - MERTEBE (order) tabanlı.

Neden mertebe, neden Hz değil:

PHM 2010 ve NASA farklı iğ devirlerinde ve çok farklı örnekleme hızlarında
çalışıyor (50 kHz / 250 Hz). Mutlak frekansta tanımlanan bir öznitelik iki
veri seti arasında taşınamaz - PHM'in iğ frekansı (173 Hz) NASA'nın Nyquist
sınırının (125 Hz) bile üstünde.

Mertebe alanında ise ikisi karşılaştırılabilir hale gelir:

    mertebe 1  = iğ frekansı
    mertebe 3  = diş geçişi (3 ağızlı kesici)
    mertebe k  = k x iğ frekansı

Bu yüzden band enerjileri baştan mertebe cinsinden tanımlanıyor. Sonradan
dönüştürmek mümkün değil - Faz 07'deki transfer buna bağlı.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def spindle_frequency_hz(rpm: float) -> float:
    """İğ dönüş frekansı (mertebe 1)."""
    return rpm / 60.0


def tooth_passing_frequency_hz(rpm: float, flutes: int) -> float:
    """Diş geçiş frekansı: iğ frekansı x ağız sayısı."""
    return spindle_frequency_hz(rpm) * flutes


def max_usable_order(sampling_rate_hz: float, rpm: float) -> float:
    """Nyquist sınırının izin verdiği en yüksek mertebe.

    Transfer denemesinde iki veri setinin ortak tavanı bu değerin küçüğüdür.
    """
    return (sampling_rate_hz / 2.0) / spindle_frequency_hz(rpm)


def order_band_energies(
    values: np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    half_width_orders: float = 0.25,
    nperseg: int = 8192,
) -> dict[str, float]:
    """Her mertebenin etrafındaki band enerjisi.

    Welch yöntemiyle güç spektral yoğunluğu hesaplanır, sonra her mertebenin
    ``+/- half_width_orders`` komşuluğundaki güç toplanır.

    Ayrıca ``total`` (tüm band) ve her mertebe için ``*_ratio`` (toplam içindeki
    pay) döndürülür. Oran, sinyal genliğindeki genel kaymalardan bağımsız
    olduğu için tezgâhlar arasında daha taşınabilirdir.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 16:
        return {}

    f0 = spindle_frequency_hz(rpm)
    nyquist = sampling_rate_hz / 2.0

    nperseg = int(min(nperseg, values.size))
    freqs, psd = sp_signal.welch(values, fs=sampling_rate_hz, nperseg=nperseg)

    resolution = freqs[1] - freqs[0] if len(freqs) > 1 else sampling_rate_hz
    half_width_hz = max(half_width_orders * f0, resolution)

    total = float(np.trapezoid(psd, freqs))
    result: dict[str, float] = {"order_total": total}

    for order in range(1, max_order + 1):
        centre = order * f0
        if centre + half_width_hz > nyquist:
            # Bu mertebe örnekleme hızının izin verdiği bandın dışında.
            result[f"order_{order}"] = float("nan")
            result[f"order_{order}_ratio"] = float("nan")
            continue

        mask = (freqs >= centre - half_width_hz) & (freqs <= centre + half_width_hz)
        energy = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
        result[f"order_{order}"] = energy
        result[f"order_{order}_ratio"] = energy / total if total > 0 else float("nan")

    return result


def welch_spectrum(
    values: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Çizim için ham Welch spektrumu (frekans, güç)."""
    values = np.asarray(values, dtype=float).ravel()
    nperseg = int(min(nperseg, values.size))
    return sp_signal.welch(values, fs=sampling_rate_hz, nperseg=nperseg)
