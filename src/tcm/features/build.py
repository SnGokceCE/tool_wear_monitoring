"""Geçiş başına öznitelik tablosu üretimi.

945 geçiş dosyasının her biri ~127 bin satır. Her çalıştırmada yeniden okumak
anlamsız olduğu için sonuç bir kez üretilip ``data/processed`` altına yazılıyor.
Faz 02'nin keşif betiği de, Faz 04'ün modeli de aynı tabloyu kullanır.

Öznitelik hesabının kendisi burada DEĞİL, ``tcm.features.extract`` içinde:
çıkarım hattı da aynı kodu çağırsın diye (bkz. o modülün açıklaması). Bu dosya
yalnızca veri kümesine özgü sarmalayıcıları ve önbelleği tutar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tcm.datasets.nasa import run_table as nasa_run_table
from tcm.datasets.phm2010 import PHM2010
from tcm.features.extract import assemble_feature_table, run_feature_row


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
            full = dataset.load_cut(cutter, ref.index)

            row: dict[str, float] = {
                "cutter": cutter,
                "cut": ref.index,
                # Geçişin süresi. NASA tarafında kümülatif kesme süresi en güçlü
                # girdi çıktı; iki veri setini birleştirebilmek için PHM'de de
                # aynı büyüklüğün olması gerekiyor. Dosya uzunluğundan hesaplanır.
                "run_time": len(full) / sampling_rate_hz,
            }
            row.update(
                run_feature_row(
                    full,
                    sampling_rate_hz=sampling_rate_hz,
                    rpm=rpm,
                    max_order=max_order,
                    keep=keep,
                )
            )

            rows.append(row)

    features = pd.DataFrame(rows).sort_values(["cutter", "cut"]).reset_index(drop=True)
    features["cum_time"] = features.groupby("cutter")["run_time"].cumsum()
    return features


def attach_wear(features: pd.DataFrame, dataset: PHM2010) -> pd.DataFrame:
    """Öznitelik tablosuna aşınma etiketlerini ekler."""
    wear_frames = []
    for cutter in features["cutter"].unique():
        wear = dataset.wear(cutter)[["cut", "vb_um", "flute_spread_um"]].copy()
        wear["cutter"] = cutter
        wear_frames.append(wear)

    wear_all = pd.concat(wear_frames, ignore_index=True)
    return features.merge(wear_all, on=["cutter", "cut"], how="inner")


def build_nasa_features(
    dataset,
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
    drop_unlabelled: bool = True,
    drop_cases: tuple[int, ...] = (),
    show_progress: bool = True,
) -> pd.DataFrame:
    """NASA Milling için koşu başına öznitelik + kesme parametresi tablosu.

    PHM'den iki farkı var:
      - Kesme parametreleri (malzeme, ilerleme, kesme derinliği) sütun olarak
        eklenir. Model B'nin varlık sebebi bunlar.
      - Etiket mm cinsinden geliyor, mikrometreye çevrilir (PHM ile aynı birim).

    ``drop_cases`` ile bilinen bozuk vakalar dışlanır (vaka 6'da tek koşu var).

    Öznitelik hesabı ve türetilmiş sütunlar ``extract.assemble_feature_table``
    içinde; bu fonksiyon yalnızca ``mill.mat`` alan adlarını ortak şemaya
    çevirir. Çıkarım hattı aynı ortak şemayı doldurup aynı fonksiyonu çağırır.
    """
    runs = nasa_run_table(
        dataset.metadata(),
        drop_unlabelled=drop_unlabelled,
        drop_cases=drop_cases,
    )

    return assemble_feature_table(
        runs,
        lambda entry: dataset.signals(int(entry.entry)),
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        max_order=max_order,
        keep=keep,
        show_progress=show_progress,
    )


def load_or_build_nasa(
    cache_path: str | Path,
    dataset,
    sampling_rate_hz: float,
    rpm: float,
    max_order: int = 8,
    keep: float = 0.5,
    drop_cases: tuple[int, ...] = (),
    rebuild: bool = False,
) -> pd.DataFrame:
    """NASA öznitelik tablosu; önbellek varsa okur."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        print(f"Önbellekten okunuyor: {cache_path}")
        return pd.read_csv(cache_path)

    print("NASA öznitelikleri üretiliyor...")
    features = build_nasa_features(
        dataset,
        sampling_rate_hz=sampling_rate_hz,
        rpm=rpm,
        max_order=max_order,
        keep=keep,
        drop_cases=drop_cases,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache_path, index=False)
    print(f"Kaydedildi: {cache_path}  ({len(features)} satır, {features.shape[1]} sütun)")
    return features


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
