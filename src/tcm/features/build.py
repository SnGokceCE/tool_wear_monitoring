"""Geçiş başına öznitelik tablosu üretimi.

945 geçiş dosyasının her biri ~127 bin satır. Her çalıştırmada yeniden okumak
anlamsız olduğu için sonuç bir kez üretilip ``data/processed`` altına yazılıyor.
Faz 02'nin keşif betiği de, Faz 04'ün modeli de aynı tabloyu kullanır.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tcm.datasets.phm2010 import PHM2010
from tcm.features.spectral import order_band_energies
from tcm.features.timedomain import frame_features, stable_region


def build_cut_features(
    dataset: PHM2010,
    cutters: list[str],
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
    limit: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Her geçiş için zaman ve mertebe alanı özniteliklerini çıkarır.

    Sonuç, ``cutter`` ve ``cut`` sütunlarıyla aşınma tablosuna birleştirilebilir.
    """
    rows: list[dict[str, float]] = []

    for cutter in cutters:
        refs = dataset.cut_refs(cutter)
        if limit is not None:
            refs = refs[:limit]

        iterator = tqdm(refs, desc=f"{cutter}", disable=not show_progress)
        for ref in iterator:
            frame = stable_region(dataset.load_cut(cutter, ref.index), keep=keep)

            row: dict[str, float] = {"cutter": cutter, "cut": ref.index}
            row.update(frame_features(frame))

            for channel in frame.columns:
                bands = order_band_energies(
                    frame[channel].to_numpy(),
                    sampling_rate_hz=sampling_rate_hz,
                    rpm=rpm,
                    max_order=max_order,
                )
                row.update({f"{channel}_{name}": value for name, value in bands.items()})

            rows.append(row)

    return pd.DataFrame(rows)


def attach_wear(features: pd.DataFrame, dataset: PHM2010) -> pd.DataFrame:
    """Öznitelik tablosuna aşınma etiketlerini ekler."""
    wear_frames = []
    for cutter in features["cutter"].unique():
        wear = dataset.wear(cutter)[["cut", "vb_um", "flute_spread_um"]].copy()
        wear["cutter"] = cutter
        wear_frames.append(wear)

    wear_all = pd.concat(wear_frames, ignore_index=True)
    return features.merge(wear_all, on=["cutter", "cut"], how="inner")


def load_or_build(
    cache_path: str | Path,
    dataset: PHM2010,
    cutters: list[str],
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
    rebuild: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Önbellek varsa okur, yoksa üretip yazar."""
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild:
        print(f"Önbellekten okunuyor: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Öznitelikler üretiliyor ({len(cutters)} kesici)...")
    features = build_cut_features(
        dataset,
        cutters,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        max_order=max_order,
        keep=keep,
        limit=limit,
    )
    features = attach_wear(features, dataset)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, index=False)
    print(f"Kaydedildi: {cache_path}  ({len(features)} satır, {features.shape[1]} sütun)")
    return features
