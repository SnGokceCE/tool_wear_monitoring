"""Naif taban model - sinyale hiç bakmaz.

Yalnızca "kaçıncı geçişteyiz" bilgisinden VB tahmin eder. Aşınma zamanla
monoton arttığı için bu şaşırtıcı derecede iyi çalışır; **her modelin
yenmesi gereken çizgi budur.** Yenemeyen model, sensörlerden hiçbir bilgi
çıkaramamış demektir.

Tasarım notu - neden ham geçiş numarası, normalize edilmiş ilerleme değil:
takımın toplam ömrünü baştan bilmek, gerçek kullanımda sahip olmadığımız
bir bilgidir. Normalize etmek sızıntı olur ve tabanı haksız yere güçlendirir.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression


class NaiveWearBaseline:
    """Geçiş numarasından izotonik (monoton artan) VB tahmini.

    Eğitim: birden çok kesicinin (geçiş numarası, VB) çiftleri havuzlanır ve
    tek bir monoton eğri uydurulur. Tahmin: test kesicisinin geçiş numaraları
    bu eğriden okunur.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self._fitted = False

    def fit(self, cut_indices: ArrayLike, wear_um: ArrayLike) -> "NaiveWearBaseline":
        cuts = np.asarray(cut_indices, dtype=float).ravel()
        wear = np.asarray(wear_um, dtype=float).ravel()
        if cuts.shape != wear.shape:
            raise ValueError(
                f"Boyutlar uyuşmuyor: cut_indices{cuts.shape} vs wear_um{wear.shape}"
            )
        if cuts.size == 0:
            raise ValueError("Boş veri ile eğitim yapılamaz")

        self._model.fit(cuts, wear)
        self._fitted = True
        return self

    def predict(self, cut_indices: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Önce fit() çağrılmalı")
        cuts = np.asarray(cut_indices, dtype=float).ravel()
        return np.asarray(self._model.predict(cuts), dtype=float)

    def fit_predict(
        self,
        train_cuts: ArrayLike,
        train_wear: ArrayLike,
        test_cuts: ArrayLike,
    ) -> np.ndarray:
        return self.fit(train_cuts, train_wear).predict(test_cuts)


def enforce_monotonic(predictions: ArrayLike) -> np.ndarray:
    """Tahmin dizisini monoton artan hale getirir (kümülatif maksimum).

    Aşınma fiziksel olarak azalamaz; geçiş bazında bağımsız tahmin üreten
    modellerin çıktısındaki zikzak fiziğe aykırı gürültüdür. Bu düzeltme
    hiçbir model değişikliği gerektirmeden hatayı düşürür ve arayüzdeki
    eğriyi operatör için okunabilir kılar.

    Dizinin geçiş sırasına göre sıralı olduğu varsayılır.
    """
    values = np.asarray(predictions, dtype=float).ravel()
    return np.maximum.accumulate(values)
