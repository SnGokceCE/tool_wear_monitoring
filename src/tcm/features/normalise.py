"""Takım bazında taban çizgisine göre normalizasyon.

Sorun: her takımın mutlak sinyal seviyesi farklı. Bağlantı sıkılığı, sensör
konumu, takımın kendi geometrisi... Model "yüksek kuvvet = yüksek aşınma"
öğrenirse, sinyali baştan yüksek olan bir takımda aşınmayı olduğundan fazla
tahmin eder - eğitimde görmediği bir takımda tam olarak bu olur.

Çözüm: her takımın özniteliklerini KENDİ ilk geçişlerindeki değere böl. Model
artık mutlak seviyeyi değil, "takım yeniyken neydi, şimdi kaç katı" bilgisini
görür. Takımlar arası ofset ortadan kalkar.

Bu aynı zamanda sahada uygulanabilir: yeni takım takıldığında ilk birkaç geçiş
kaydedilir, taban çizgisi olur.

Kısıt: "ilk geçişler = yeni takım" varsayımı tam doğru değil. PHM 2010'da
başlangıç aşınmaları farklı (c1 48,9 / c4 31,4 / c6 62,8 µm). Yani taban
çizgisi tamamen temiz değil; yine de ofsetin büyük kısmını alır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-12


def baseline_normalise(
    data: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    sort_column: str,
    n_baseline: int = 10,
    mode: str = "ratio",
) -> pd.DataFrame:
    """Her grubun özniteliklerini kendi ilk ``n_baseline`` geçişine göre ölçekler.

    ``mode``:
      - ``"ratio"``  : değer / taban   (oransal artış; ölçek farkını yok eder)
      - ``"delta"``  : değer - taban   (mutlak artış; birimi korur)

    Taban çizgisi olarak ortalama değil **medyan** kullanılır: ilk geçişlerde
    tek bir aykırı değer tabanı bozmasın diye.
    """
    if mode not in {"ratio", "delta"}:
        raise ValueError(f"mode 'ratio' veya 'delta' olmalı, '{mode}' verildi")
    if n_baseline < 1:
        raise ValueError(f"n_baseline en az 1 olmalı, {n_baseline} verildi")

    result = data.copy()

    for group, subset in data.groupby(group_column):
        head = subset.sort_values(sort_column).head(n_baseline)
        baseline = head[feature_columns].median()
        mask = result[group_column] == group

        if mode == "ratio":
            # Tabanı sıfıra çok yakın olan öznitelikler oransal ölçekte patlar;
            # onlarda fark almaya düşülür.
            safe = baseline.abs() > EPSILON
            ratio_columns = [c for c in feature_columns if safe.get(c, False)]
            delta_columns = [c for c in feature_columns if not safe.get(c, False)]

            if ratio_columns:
                result.loc[mask, ratio_columns] = (
                    subset[ratio_columns] / baseline[ratio_columns]
                ).to_numpy()
            if delta_columns:
                result.loc[mask, delta_columns] = (
                    subset[delta_columns] - baseline[delta_columns]
                ).to_numpy()
        else:
            result.loc[mask, feature_columns] = (
                subset[feature_columns] - baseline[feature_columns]
            ).to_numpy()

    return result


def describe_offsets(
    data: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    sort_column: str,
    n_baseline: int = 10,
) -> pd.DataFrame:
    """Gruplar arası taban çizgisi farkını ölçer - normalizasyon gerekli mi?

    Her öznitelik için grupların taban değerleri arasındaki oranı döndürür.
    Oran 1'e yakınsa takımlar benzer seviyeden başlıyor; büyükse ofset var.
    """
    rows = []
    for column in feature_columns:
        baselines = []
        for _, subset in data.groupby(group_column):
            head = subset.sort_values(sort_column).head(n_baseline)
            baselines.append(float(head[column].median()))

        baselines = np.array(baselines, dtype=float)
        finite = baselines[np.isfinite(baselines) & (np.abs(baselines) > EPSILON)]
        if finite.size < 2:
            continue

        rows.append({
            "oznitelik": column,
            "min_taban": float(np.min(finite)),
            "max_taban": float(np.max(finite)),
            "oran": float(np.max(np.abs(finite)) / np.min(np.abs(finite))),
        })

    return pd.DataFrame(rows).sort_values("oran", ascending=False).reset_index(drop=True)
