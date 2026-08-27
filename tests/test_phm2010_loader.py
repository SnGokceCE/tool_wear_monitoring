"""PHM 2010 yükleyicisinin testleri - sentetik dosyalarla.

Gerçek arşivin klasör yerleşimi kaynağa göre değişiyor; yükleyici bu yüzden
sabit yol beklemek yerine isim örüntüsüne göre tarama yapıyor. Bu testler
tam olarak o tarama mantığını doğruluyor: iç içe klasörler, eksik etiket,
ve uçtan uca naif taban akışı.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcm.datasets import PHM2010
from tcm.evaluation import leave_one_cutter_out, summarise
from tcm.models import NaiveWearBaseline

N_CHANNELS = 7


def _write_cut(path, n_rows=64, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_rows, N_CHANNELS)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(signal).to_csv(path, header=False, index=False)


def _write_wear(path, n_cuts, start=20.0, end=180.0, seed=0):
    """Üç ağızlı, monoton artan, hafif saçılımlı sentetik aşınma tablosu."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(start, end, n_cuts)
    frame = pd.DataFrame(
        {
            "cut": np.arange(1, n_cuts + 1),
            "flute_1": trend + rng.normal(0, 1.5, n_cuts),
            "flute_2": trend + rng.normal(0, 1.5, n_cuts),
            "flute_3": trend + rng.normal(0, 1.5, n_cuts),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def dataset_root(tmp_path):
    """Gerçek arşivi taklit eden, kasten iç içe bir yerleşim kurar."""
    root = tmp_path / "phm2010"
    for cutter_number, n_cuts in ((1, 12), (4, 12), (6, 12)):
        # Kasıtlı olarak iki kat derin: c1/c1/c_1_001.csv
        cut_dir = root / f"c{cutter_number}" / f"c{cutter_number}"
        for cut_index in range(1, n_cuts + 1):
            _write_cut(cut_dir / f"c_{cutter_number}_{cut_index:03d}.csv", seed=cut_index)
        _write_wear(
            root / f"c{cutter_number}" / f"c{cutter_number}_wear.csv",
            n_cuts,
            seed=cutter_number,
        )
    return root


class TestDiscovery:
    def test_finds_cutters_through_nested_folders(self, dataset_root):
        dataset = PHM2010(dataset_root)
        assert dataset.available_cutters() == ["c1", "c4", "c6"]

    def test_finds_wear_files(self, dataset_root):
        assert PHM2010(dataset_root).labelled_cutters() == ["c1", "c4", "c6"]

    def test_counts_cuts(self, dataset_root):
        assert PHM2010(dataset_root).n_cuts("c4") == 12

    def test_missing_root_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data"):
            PHM2010(tmp_path / "yok")

    def test_unknown_cutter_lists_alternatives(self, dataset_root):
        with pytest.raises(KeyError, match="c1"):
            PHM2010(dataset_root).n_cuts("c9")

    def test_cutter_without_labels_is_excluded(self, dataset_root):
        (dataset_root / "c4" / "c4_wear.csv").unlink()
        dataset = PHM2010(dataset_root)
        assert dataset.available_cutters() == ["c1", "c4", "c6"]
        assert dataset.labelled_cutters() == ["c1", "c6"]


class TestWearLabels:
    def test_returns_expected_columns(self, dataset_root):
        wear = PHM2010(dataset_root).wear("c1")
        assert {"cut", "vb_um", "flute_spread_um"} <= set(wear.columns)

    def test_max_aggregation_is_at_least_mean(self, dataset_root):
        by_max = PHM2010(dataset_root, wear_aggregation="max").wear("c1")["vb_um"]
        by_mean = PHM2010(dataset_root, wear_aggregation="mean").wear("c1")["vb_um"]
        assert (by_max >= by_mean - 1e-9).all()

    def test_rows_are_sorted_by_cut(self, dataset_root):
        wear = PHM2010(dataset_root).wear("c1")
        assert wear["cut"].is_monotonic_increasing

    def test_invalid_aggregation_rejected(self, dataset_root):
        with pytest.raises(ValueError, match="wear_aggregation"):
            PHM2010(dataset_root, wear_aggregation="median")

    def test_unlabelled_cutter_raises(self, dataset_root):
        (dataset_root / "c6" / "c6_wear.csv").unlink()
        with pytest.raises(KeyError, match="aşınma etiketi yok"):
            PHM2010(dataset_root).wear("c6")


class TestSignals:
    def test_loads_seven_channels(self, dataset_root):
        signal = PHM2010(dataset_root).load_cut("c1", 1)
        assert signal.shape[1] == N_CHANNELS
        assert list(signal.columns)[:3] == ["force_x", "force_y", "force_z"]

    def test_max_rows_limits_read(self, dataset_root):
        assert len(PHM2010(dataset_root).load_cut("c1", 1, max_rows=10)) == 10

    def test_unknown_cut_index_raises(self, dataset_root):
        with pytest.raises(KeyError, match="numaralı geçiş yok"):
            PHM2010(dataset_root).load_cut("c1", 999)


class TestEndToEnd:
    def test_naive_baseline_runs_over_all_folds(self, dataset_root):
        """Faz 00'ın asıl çıktısı: protokol uçtan uca dönüyor mu?"""
        dataset = PHM2010(dataset_root)
        cutters = dataset.labelled_cutters()
        wear = {c: dataset.wear(c) for c in cutters}

        results = []
        for split in leave_one_cutter_out(cutters):
            train = pd.concat([wear[c] for c in split.train], ignore_index=True)
            test = wear[split.test[0]]
            predictions = NaiveWearBaseline().fit_predict(
                train["cut"], train["vb_um"], test["cut"]
            )
            results.append(summarise(test["vb_um"], predictions, 150))

        assert len(results) == 3
        # Sentetik veri tek bir trendden üretildiği için taban çok iyi olmalı.
        assert all(scores["mae_um"] < 10 for scores in results)
