"""Karar mantığı - model çıktısını alarma çeviren katman.

Model bir VB sayısı üretir. "Takımı değiştir" demek ayrı bir karardır ve üç
şey gerektirir:

  1. MALİYET TANIMI
     Kaçırılan aşınma ile yanlış alarm aynı şey değildir. Kaçırılan aşınma
     parçayı hurdaya çıkarır; yanlış alarm sadece takım ömrü israf eder.
     Bu asimetri sayıyla ifade edilmeden "en iyi eşik" tanımsızdır.

  2. EŞİĞİN VERİDEN SEÇİLMESİ
     Ve bu seçim YALNIZCA eğitim verisiyle yapılmalıdır. Test sonuçlarına
     bakıp en iyi eşiği seçmek, bu projede iki kez düşülen sızıntı tuzağının
     aynısıdır.

  3. ARDIŞIK ONAY
     Tek bir geçişte eşiğin aşılması gürültü olabilir. "Üst üste k geçişte
     aşıldıysa alarm ver" kuralı yanlış alarmı azaltır, alarmı geciktirir.

Bu katman modelden bağımsızdır: gradyan artırma da olsa derin ağ da olsa
aynı şekilde uygulanır.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def alarm_cost(
    y_true_worn: ArrayLike,
    y_pred_worn: ArrayLike,
    cost_missed: float = 5.0,
    cost_false_alarm: float = 1.0,
) -> float:
    """Toplam maliyet: kaçırılan aşınma ve yanlış alarmların ağırlıklı toplamı.

    Varsayılan 5:1 oranı, "bir kaçırılan aşınma beş yanlış alarma bedeldir"
    demektir. Gerçek üretimde bu oran parça maliyeti ile takım maliyetinden
    çıkar; burada makul bir başlangıç değeri olarak alınmıştır.
    """
    truth = np.asarray(y_true_worn, dtype=bool).ravel()
    predicted = np.asarray(y_pred_worn, dtype=bool).ravel()
    if truth.shape != predicted.shape:
        raise ValueError(
            f"Boyutlar uyuşmuyor: {truth.shape} vs {predicted.shape}"
        )

    missed = int(np.sum(truth & ~predicted))
    false_alarms = int(np.sum(~truth & predicted))
    return cost_missed * missed + cost_false_alarm * false_alarms


def apply_consecutive(flags: ArrayLike, k: int = 1) -> np.ndarray:
    """Ardışık onay kuralı: üst üste ``k`` kez aşılmadan alarm verme.

    Alarm bir kez çaldıktan sonra sönmez - takım aşınması geri dönmez, o
    yüzden alarmın da geri dönmemesi gerekir (histerezis).

    Dizinin geçiş sırasına göre sıralı olduğu varsayılır.
    """
    flags = np.asarray(flags, dtype=bool).ravel()
    if k < 1:
        raise ValueError(f"k en az 1 olmalı, {k} verildi")
    if k == 1:
        return np.maximum.accumulate(flags)

    result = np.zeros_like(flags)
    streak = 0
    triggered = False
    for index, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= k:
            triggered = True
        result[index] = triggered
    return result


def _flags_by_group(
    pred_wear: np.ndarray,
    threshold: float,
    consecutive: int,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Alarm bayraklarını üretir; kilitlenme her takım içinde ayrı işler.

    KRİTİK: ``apply_consecutive`` alarmı kilitler (bir kez çalınca sönmez).
    Bu, tek bir takımın ömrü içinde doğrudur - aşınma geri dönmez. Ama
    birden çok takımın tahminleri arka arkaya eklenmiş bir dizide uygulanırsa
    ilk takımdaki alarm sonraki bütün takımları da alarmda gösterir.

    Bu hata gerçekten yapıldı ve eşik seçimini bozdu: optimizasyon, erken
    alarmların her şeyi zehirlemesinden kaçınmak için eşiği absürt biçimde
    yükseltiyordu (300 -> 421 µm).
    """
    raw = pred_wear >= threshold
    if groups is None:
        return apply_consecutive(raw, consecutive)

    result = np.zeros_like(raw)
    for group in np.unique(groups):
        mask = groups == group
        result[mask] = apply_consecutive(raw[mask], consecutive)
    return result


def choose_threshold(
    y_true_wear: ArrayLike,
    y_pred_wear: ArrayLike,
    wear_limit_um: float,
    cost_missed: float = 5.0,
    cost_false_alarm: float = 1.0,
    search_span: float = 0.5,
    n_candidates: int = 101,
    consecutive: int = 1,
    groups: ArrayLike | None = None,
) -> float:
    """Maliyeti en aza indiren alarm eşiğini seçer.

    DİKKAT: buraya verilen veri EĞİTİM verisi olmalıdır. Test verisiyle
    çağrılırsa sonuç iyimser çıkar ve raporlanan performans gerçek dışı olur.

    Aday eşikler ``wear_limit_um`` etrafında ``search_span`` oranında bir
    aralıkta taranır (varsayılan: sınırın %50 altı ile %50 üstü arası).

    Beraberlik durumunda DÜŞÜK eşik tercih edilir - güvenli taraf.
    """
    truth_wear = np.asarray(y_true_wear, dtype=float).ravel()
    pred_wear = np.asarray(y_pred_wear, dtype=float).ravel()
    truth_worn = truth_wear >= wear_limit_um
    group_array = None if groups is None else np.asarray(groups).ravel()

    low = wear_limit_um * (1.0 - search_span)
    high = wear_limit_um * (1.0 + search_span)
    candidates = np.linspace(low, high, n_candidates)

    best_threshold = float(wear_limit_um)
    best_cost = np.inf

    for threshold in candidates:
        flags = _flags_by_group(pred_wear, threshold, consecutive, group_array)
        cost = alarm_cost(truth_worn, flags, cost_missed, cost_false_alarm)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold


def choose_consecutive(
    y_true_wear: ArrayLike,
    y_pred_wear: ArrayLike,
    threshold: float,
    wear_limit_um: float,
    cost_missed: float = 5.0,
    cost_false_alarm: float = 1.0,
    candidates: tuple[int, ...] = (1, 2, 3),
    groups: ArrayLike | None = None,
) -> int:
    """Ardışık onay sayısını maliyete göre seçer. Yine EĞİTİM verisiyle."""
    truth_worn = np.asarray(y_true_wear, dtype=float).ravel() >= wear_limit_um
    pred_wear = np.asarray(y_pred_wear, dtype=float).ravel()
    group_array = None if groups is None else np.asarray(groups).ravel()

    best_k, best_cost = candidates[0], np.inf
    for k in candidates:
        flags = _flags_by_group(pred_wear, threshold, k, group_array)
        cost = alarm_cost(truth_worn, flags, cost_missed, cost_false_alarm)
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k
