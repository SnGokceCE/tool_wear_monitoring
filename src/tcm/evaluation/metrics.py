"""Değerlendirme metrikleri.

Üç metrik kullanılıyor. İlk ikisi standart regresyon hatası; üçüncüsü
sistemin asıl işini ölçer: aşınma sınırını ne kadar geç fark ediyoruz?

Geç alarm ile erken alarm simetrik değildir - geç alarm iş parçasını
hurdaya çıkarır - bu yüzden gecikme işaretli (signed) raporlanır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mae_um(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Ortalama mutlak hata, mikrometre."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_um(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Karekök ortalama kare hata, mikrometre."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def first_crossing(values: ArrayLike, threshold: float) -> int | None:
    """Dizinin eşiği ilk aştığı konum; hiç aşmıyorsa ``None``.

    Not: dizinin geçiş sırasına göre sıralı olduğu varsayılır.
    """
    values = np.asarray(values, dtype=float)
    above = np.flatnonzero(values >= threshold)
    return int(above[0]) if above.size else None


def crossing_delay_cuts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: float,
) -> float:
    """Eşiğin geçildiğini kaç kesme geç/erken fark ettiğimiz.

    Pozitif değer gecikme (tehlikeli), negatif değer erken alarm (maliyetli
    ama güvenli) demektir.

    Gerçek dizi eşiği hiç aşmıyorsa metrik tanımsızdır ve ``nan`` döner.
    Gerçek dizi aşıyor ama tahmin aşmıyorsa, kaçırılan alarm en kötü durum
    kabul edilerek dizinin sonuna kadar gecikme sayılır.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)

    true_idx = first_crossing(y_true, threshold)
    if true_idx is None:
        return float("nan")

    pred_idx = first_crossing(y_pred, threshold)
    if pred_idx is None:
        return float(len(y_true) - true_idx)

    return float(pred_idx - true_idx)


def alarm_overshoot_um(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: float,
) -> float:
    """Alarm geldiğinde gerçek aşınmanın eşiği ne kadar aşmış olduğu.

    ``crossing_delay_cuts`` metriğinin ciddi bir kusuru var: eğrinin eşik
    civarındaki EĞİMİNE bağlı. Aşınma eğrisi düzse küçük bir µm hatası devasa
    bir geçiş kayması üretir; dikse aynı hata birkaç geçişe karşılık gelir.
    Bu yüzden farklı kesiciler arasında karşılaştırılamaz.

    Bu metrik eğimden bağımsız ve doğrudan yorumlanabilir: *alarm çaldığında
    takım eşiği kaç mikrometre aşmıştı?*

      pozitif  -> geç alarm; takım bu kadar fazla aşınmışken haber verdik
      negatif  -> erken alarm; takım eşiğe bu kadar uzaktayken haber verdik

    Model hiç alarm vermezse, dizinin sonundaki gerçek aşınma esas alınır -
    kaçırılan alarmın bedeli budur.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)

    pred_idx = first_crossing(y_pred, threshold)
    if pred_idx is None:
        return float(y_true[-1] - threshold)

    return float(y_true[pred_idx] - threshold)


def summarise(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    wear_limit_um: float,
) -> dict[str, float]:
    """Değerlendirme metriklerini tek sözlükte döndürür."""
    return {
        "mae_um": mae_um(y_true, y_pred),
        "rmse_um": rmse_um(y_true, y_pred),
        "crossing_delay_cuts": crossing_delay_cuts(y_true, y_pred, wear_limit_um),
        "overshoot_um": alarm_overshoot_um(y_true, y_pred, wear_limit_um),
    }


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Boyutlar uyuşmuyor: y_true{y_true.shape} vs y_pred{y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Boş dizi ile metrik hesaplanamaz")
    return y_true, y_pred
