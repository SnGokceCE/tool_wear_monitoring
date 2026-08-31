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

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
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


def alarm_flags(
    pred_wear: ArrayLike,
    threshold: float,
    consecutive: int = 1,
    groups: ArrayLike | None = None,
) -> np.ndarray:
    """Tahmin dizisinden alarm bayrakları; kilitlenme her takım içinde ayrı işler.

    BİRDEN ÇOK TAKIM İÇEREN HER DİZİDE BU FONKSİYON KULLANILMALIDIR -
    ``apply_consecutive`` doğrudan çağrılmamalıdır.

    KRİTİK: ``apply_consecutive`` alarmı kilitler (bir kez çalınca sönmez).
    Bu, tek bir takımın ömrü içinde doğrudur - aşınma geri dönmez. Ama
    birden çok takımın tahminleri arka arkaya eklenmiş bir dizide uygulanırsa
    ilk takımdaki alarm sonraki bütün takımları da alarmda gösterir.

    Bu hata bu projede İKİ KEZ yapıldı:

      1. Eşik seçiminde (Faz 09). Optimizasyon, erken alarmların her şeyi
         zehirlemesinden kaçınmak için eşiği absürt biçimde yükseltiyordu
         (300 -> 421 µm). Düzeltildi.
      2. Dış değerlendirmede (yine Faz 09, ``run_decision_rule.py``).
         Kilit katlamanın tamamına uygulanıyordu; kaçırılan aşınma sayısı
         üçte bir görünüyordu (21 yerine 7). Faz 06'da bulundu ve düzeltildi.

    İkinci hatanın hayatta kalma sebebi, bu fonksiyonun adının alt çizgiyle
    başlaması ve "özel" görünmesiydi - çağıran kod yanlış olan genel
    fonksiyonu seçti. Bu yüzden artık adı açıkça geneldir.
    """
    pred_wear = np.asarray(pred_wear, dtype=float).ravel()
    raw = pred_wear >= threshold

    if groups is None:
        return apply_consecutive(raw, consecutive)

    groups = np.asarray(groups).ravel()
    if groups.shape != raw.shape:
        raise ValueError(
            f"Boyutlar uyuşmuyor: tahmin{raw.shape} vs gruplar{groups.shape}"
        )

    result = np.zeros_like(raw)
    for group in np.unique(groups):
        mask = groups == group
        result[mask] = apply_consecutive(raw[mask], consecutive)
    return result


# Eski ad - mevcut çağrılar kırılmasın diye. Yeni kod ``alarm_flags`` kullanmalı.
_flags_by_group = alarm_flags


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
        flags = alarm_flags(pred_wear, threshold, consecutive, group_array)
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
        flags = alarm_flags(pred_wear, threshold, k, group_array)
        cost = alarm_cost(truth_worn, flags, cost_missed, cost_false_alarm)
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k


# ---------------------------------------------------------------------------
# Katlama dışı tahmin üretimi ve eşik kalibrasyonu
#
# Bu iki fonksiyon Faz 09'da ``scripts/run_decision_rule.py`` içinde yerel
# olarak yazılmıştı. Faz 06'da nihai modelin eşiği de aynı mantıkla seçilecek;
# kodu kopyalamak yerine buraya taşındı. İki fazın AYNI kodu çağırması, "Faz
# 09'daki mantıkla seçildi" ifadesinin denetlenebilir olmasının tek yolu.
# ---------------------------------------------------------------------------


@dataclass
class OutOfFoldPredictions:
    """Katlama dışı tahminler ve nasıl üretildiklerinin kaydı.

    ``split_column`` gerçekte kullanılan bölme sütunudur; istenen sütunda
    yeterli grup yoksa ``fallback_column``'a düşülür ve ``fell_back`` True olur.
    Bu bilgi künyeye yazılır: eşiğin hangi senaryoya kalibre edildiğini
    değiştirir, sessiz kalması yanıltıcı olur.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray
    requested_column: str
    split_column: str
    n_folds: int

    @property
    def fell_back(self) -> bool:
        return self.split_column != self.requested_column

    def describe(self) -> str:
        note = (
            f" (istenen '{self.requested_column}' sütununda yeterli grup yok, "
            f"'{self.split_column}' ile bölündü)"
            if self.fell_back else ""
        )
        return f"{self.n_folds} katlama, {len(self.y_true)} tahmin{note}"


def oof_predictions(
    data: pd.DataFrame,
    group_column: str,
    feature_columns: Sequence[str],
    model_factory: Callable[[], object],
    *,
    target_column: str = "vb_um",
    sort_column: str = "run",
    latch_column: str = "case",
    fallback_column: str = "case",
    min_groups: int = 3,
) -> OutOfFoldPredictions:
    """Verinin kendi içinde katlama dışı (out-of-fold) tahminler üretir.

    Her satır, o satırı EĞİTİMDE GÖRMEMİŞ bir modelden tahmin alır. Eşik bu
    tahminlerden seçilirse sızıntı olmaz: model kendi eğitim verisindeki
    iyimser hatasına göre değil, gerçek genelleme hatasına göre kalibre edilir.

    ``min_groups``: istenen sütunda bu kadar grup yoksa ``fallback_column``
    kullanılır. Faz 09'daki değer 3'tü. Malzeme kalibrasyonunda 2 verilir -
    NASA'da iki malzeme var (dökme demir, çelik) ve "birinde eğit, diğerinde
    ölç" tam olarak teslim senaryosunun kendisidir; ``case``'e düşmek o sınavı
    kolaylaştırır ve eşiği olduğundan gevşek seçtirir.

    ``latch_column``: alarm kilitlenmesinin uygulanacağı birim (takım).
    Katlamalardan değil, satırların kendisinden gelir - birleştirilmiş dizide
    kilitlemenin takımlar arasına taşmaması için gerekli (bkz. ``alarm_flags``).
    """
    feature_columns = list(feature_columns)

    split_column = group_column
    if data[group_column].nunique() < min_groups:
        split_column = fallback_column

    truths, preds, groups = [], [], []
    n_folds = 0

    for held_out in sorted(data[split_column].unique()):
        train = data[data[split_column] != held_out]
        test = data[data[split_column] == held_out].sort_values(sort_column)
        if train.empty or test.empty:
            continue

        model = model_factory()
        model.fit(train[feature_columns], train[target_column])

        truths.append(test[target_column].to_numpy(dtype=float))
        preds.append(np.asarray(model.predict(test[feature_columns]), dtype=float))
        groups.append(test[latch_column].to_numpy())
        n_folds += 1

    if not truths:
        raise RuntimeError(
            f"'{split_column}' ile çapraz doğrulama kurulamadı - "
            "veride yeterli grup yok."
        )

    return OutOfFoldPredictions(
        y_true=np.concatenate(truths),
        y_pred=np.concatenate(preds),
        groups=np.concatenate(groups),
        requested_column=group_column,
        split_column=split_column,
        n_folds=n_folds,
    )


@dataclass
class ThresholdCalibration:
    """Bir eşiğin değeri ve hangi kurguyla seçildiği."""

    threshold: float
    consecutive: int
    wear_limit_um: float
    cost_missed: float
    cost_false_alarm: float
    requested_column: str
    split_column: str
    n_folds: int
    n_predictions: int
    cost: float

    @property
    def fell_back(self) -> bool:
        return self.split_column != self.requested_column

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold_um": self.threshold,
            "consecutive_k": self.consecutive,
            "wear_limit_um": self.wear_limit_um,
            "cost_missed": self.cost_missed,
            "cost_false_alarm": self.cost_false_alarm,
            "requested_split": self.requested_column,
            "actual_split": self.split_column,
            "fell_back": self.fell_back,
            "n_folds": self.n_folds,
            "n_predictions": self.n_predictions,
            "inner_cost": self.cost,
        }


def calibrate_threshold(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    model_factory: Callable[[], object],
    *,
    group_column: str,
    wear_limit_um: float,
    cost_missed: float = 5.0,
    cost_false_alarm: float = 1.0,
    search_span: float = 0.5,
    consecutive_candidates: Sequence[int] | None = None,
    target_column: str = "vb_um",
    sort_column: str = "run",
    latch_column: str = "case",
    fallback_column: str = "case",
    min_groups: int = 3,
) -> ThresholdCalibration:
    """Alarm eşiğini katlama dışı tahminlerden seçer - Faz 09 mantığı.

    ``consecutive_candidates`` verilirse ardışık onay sayısı da aynı
    tahminlerden seçilir; verilmezse k = 1 (anında alarm) sabitlenir.

    Dönen eşik ``data``'nın TAMAMINDAN türetilir. Nihai model de aynı veriyle
    eğitildiği için bu tutarlıdır: eşik, modelin bu veri üzerindeki genelleme
    hatasına göre ayarlanmış olur.
    """
    oof = oof_predictions(
        data, group_column, feature_columns, model_factory,
        target_column=target_column, sort_column=sort_column,
        latch_column=latch_column, fallback_column=fallback_column,
        min_groups=min_groups,
    )

    k = 1
    if consecutive_candidates:
        k = choose_consecutive(
            oof.y_true, oof.y_pred, wear_limit_um, wear_limit_um,
            cost_missed=cost_missed, cost_false_alarm=cost_false_alarm,
            candidates=tuple(consecutive_candidates), groups=oof.groups,
        )

    threshold = choose_threshold(
        oof.y_true, oof.y_pred, wear_limit_um,
        cost_missed=cost_missed, cost_false_alarm=cost_false_alarm,
        search_span=search_span, consecutive=k, groups=oof.groups,
    )

    flags = alarm_flags(oof.y_pred, threshold, k, oof.groups)
    cost = alarm_cost(
        oof.y_true >= wear_limit_um, flags, cost_missed, cost_false_alarm
    )

    return ThresholdCalibration(
        threshold=float(threshold),
        consecutive=int(k),
        wear_limit_um=float(wear_limit_um),
        cost_missed=float(cost_missed),
        cost_false_alarm=float(cost_false_alarm),
        requested_column=oof.requested_column,
        split_column=oof.split_column,
        n_folds=oof.n_folds,
        n_predictions=int(len(oof.y_true)),
        cost=float(cost),
    )
