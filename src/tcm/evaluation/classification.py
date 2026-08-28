"""worn / unworn sınıflandırma metrikleri.

Neden tek bir "doğruluk" sayısı yetmiyor:

İki hata türü var ve maliyetleri simetrik DEĞİL.

  Kaçırılan aşınma (worn'a unworn demek)
      Aşınmış takım kesmeye devam eder. Yüzey kalitesi bozulur, ölçü kaçar,
      parça hurdaya gider; takım kırılıp tezgâha zarar verebilir.
      Havacılık parçasında bu maliyet çok yüksektir.

  Yanlış alarm (unworn'a worn demek)
      Sağlam takım erken değiştirilir. Takım ömrü israf olur, tezgâh boşta
      kalır. Can sıkıcı ama telafi edilebilir.

Bu yüzden asıl bakılacak sayı, aşınmış takımların kaçta kaçını yakaladığımız
(worn_recall). Doğruluk (accuracy) yanıltıcıdır: sınıflar dengesizse "hepsine
unworn de" diyen bir model bile yüksek doğruluk alabilir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def confusion(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """Karışıklık matrisi. Pozitif sınıf = worn (aşınmış)."""
    y_true = np.asarray(y_true, dtype=bool).ravel()
    y_pred = np.asarray(y_pred, dtype=bool).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Boyutlar uyuşmuyor: y_true{y_true.shape} vs y_pred{y_pred.shape}"
        )

    return {
        "tp": int(np.sum(y_true & y_pred)),      # aşınmış, doğru yakalandı
        "fn": int(np.sum(y_true & ~y_pred)),     # aşınmış, KAÇIRILDI
        "fp": int(np.sum(~y_true & y_pred)),     # sağlam, boşuna alarm
        "tn": int(np.sum(~y_true & ~y_pred)),    # sağlam, doğru
    }


def classification_scores(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Sınıflandırma metrikleri.

    ``worn_recall``  : aşınmış takımların kaçta kaçını yakaladık (asıl metrik)
    ``unworn_recall``: sağlam takımların kaçta kaçına doğru "sağlam" dedik
    ``balanced_acc`` : ikisinin ortalaması; sınıf dengesizliğine karşı dayanıklı
    ``worn_precision``: "aşınmış" dediklerimizin kaçta kaçı gerçekten aşınmış
    ``missed_worn``  : kaçırılan aşınmış takım SAYISI (yüzde değil, adet)
    """
    matrix = confusion(y_true, y_pred)
    tp, fn, fp, tn = matrix["tp"], matrix["fn"], matrix["fp"], matrix["tn"]
    total = tp + fn + fp + tn

    worn_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    unworn_recall = tn / (tn + fp) if (tn + fp) else float("nan")
    worn_precision = tp / (tp + fp) if (tp + fp) else float("nan")

    if np.isnan(worn_recall) or np.isnan(unworn_recall):
        balanced = float("nan")
    else:
        balanced = (worn_recall + unworn_recall) / 2

    return {
        "accuracy": (tp + tn) / total if total else float("nan"),
        "balanced_acc": balanced,
        "worn_recall": worn_recall,
        "unworn_recall": unworn_recall,
        "worn_precision": worn_precision,
        "missed_worn": float(fn),
        "false_alarms": float(fp),
        "n": float(total),
    }


def majority_baseline(y_true: ArrayLike) -> dict[str, float]:
    """"Hep çoğunluk sınıfını söyle" tabanı.

    Doğruluğun neden yanıltıcı olduğunu göstermek için: bu model hiçbir şey
    öğrenmez ama sınıflar dengesizse yüksek doğruluk alır - ve aşınmış
    takımların HEPSİNİ kaçırır.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    predict_worn = bool(np.mean(y_true) > 0.5)
    return classification_scores(y_true, np.full(y_true.shape, predict_worn))
