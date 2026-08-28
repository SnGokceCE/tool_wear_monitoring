"""Değerlendirme protokolü - grup bazında dışarıda bırakma.

Tek bir yerde tutulmasının sebebi: Model A (kesici bazında), Model B-1 ve B-2
(koşul bazında) aynı protokolü kullanacak. Aynı kodu kullanmazlarsa sonuçları
karşılaştırmak anlamsız olur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from tcm.evaluation.metrics import summarise


@dataclass
class FoldResult:
    """Tek bir katlamanın sonucu."""

    name: str
    n_train: int
    n_test: int
    scores: dict[str, float]
    predictions: pd.DataFrame = field(repr=False)


def run_grouped_cv(
    data: pd.DataFrame,
    group_column: str,
    feature_columns: Sequence[str],
    target_column: str,
    model_factory: Callable[[], object],
    wear_limit_um: float,
    sort_column: str | None = None,
    sample_weight_column: str | None = None,
    postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    extra_train: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Her grubu sırayla test kümesi yaparak modeli değerlendirir.

    ``model_factory`` her katlamada YENİ bir model üretmelidir - aynı modeli
    yeniden kullanmak, önceki katlamanın öğrendiklerinin sızmasına yol açar.

    ``postprocess`` tahmin dizisine uygulanır (örneğin monoton düzleştirme).
    Sıralamaya bağlı olduğu için ``sort_column`` ile birlikte kullanılmalıdır.

    ``extra_train`` her katlamada eğitime eklenen, HİÇBİR ZAMAN test olmayan
    satırlar. Model B-2'de PHM verisi böyle giriyor: sınav NASA'nın kendi
    grupları üzerinde yapılıyor, PHM yalnızca yardımcı eğitim verisi.
    """
    groups = sorted(data[group_column].unique())
    if len(groups) < 2:
        raise ValueError(
            f"Grup bazında bölme için en az 2 grup gerekir, {len(groups)} bulundu"
        )

    feature_columns = list(feature_columns)
    results: list[FoldResult] = []

    for held_out in groups:
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out]
        if sort_column is not None:
            test = test.sort_values(sort_column)
        if extra_train is not None and len(extra_train):
            train = pd.concat([train, extra_train], ignore_index=True)

        model = model_factory()
        fit_kwargs = {}
        if sample_weight_column is not None:
            fit_kwargs["sample_weight"] = train[sample_weight_column].to_numpy()

        model.fit(train[feature_columns], train[target_column], **fit_kwargs)
        predictions = np.asarray(model.predict(test[feature_columns]), dtype=float)

        if postprocess is not None:
            predictions = postprocess(predictions)

        truth = test[target_column].to_numpy(dtype=float)
        frame = pd.DataFrame({
            group_column: held_out,
            "y_true": truth,
            "y_pred": predictions,
        })
        if sort_column is not None:
            frame[sort_column] = test[sort_column].to_numpy()

        results.append(
            FoldResult(
                name=str(held_out),
                n_train=len(train),
                n_test=len(test),
                scores=summarise(truth, predictions, wear_limit_um),
                predictions=frame,
            )
        )

    table = pd.DataFrame([
        {"fold": r.name, "n_train": r.n_train, "n_test": r.n_test, **r.scores}
        for r in results
    ])
    return table, results


def summarise_folds(table: pd.DataFrame) -> dict[str, float]:
    """Katlama tablosundan tek satırlık özet.

    Gecikme için hem mutlak ortalama hem en kötü değer verilir: işaretli
    ortalama yanıltıcıdır, erken ve geç alarmlar birbirini götürür.
    """
    delay = table["crossing_delay_cuts"]
    overshoot = table["overshoot_um"]
    return {
        "mae_um": float(table["mae_um"].mean()),
        "rmse_um": float(table["rmse_um"].mean()),
        "abs_overshoot_um": float(overshoot.abs().mean()),
        "worst_overshoot_um": float(overshoot.max()),
        "abs_delay_cuts": float(delay.abs().mean()),
    }
