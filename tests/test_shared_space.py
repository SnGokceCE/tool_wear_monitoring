"""Ortak öznitelik uzayının testleri (Model B-2).

Bu modül iki farklı tezgâhın verisini tek tabloda birleştiriyor. Sessizce
yanlış hizalanması en tehlikeli hata türü olurdu - sonuç üretilir ama anlamsız
olur. Testler hizalamanın gerçekten doğru olduğunu doğruluyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcm.features.shared import (
    PARAMETER_COLUMNS,
    PHM_MATERIAL_CODE,
    TIME_COLUMN,
    build_shared_table,
    collapse_channel_groups,
    prepare_nasa,
    prepare_phm,
    sensor_columns_of,
)


def _phm_table(n=20):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "cutter": ["c1"] * n,
        "cut": np.arange(1, n + 1),
        "vb_um": np.linspace(50, 200, n),
        "cum_time": np.arange(1, n + 1) * 2.5,
    })
    for channel in ("vib_x", "vib_y", "vib_z", "ae_rms", "force_x"):
        for suffix in ("rms", "order_3", "order_3_ratio"):
            frame[f"{channel}_{suffix}"] = rng.uniform(1, 2, n)
    return frame


def _nasa_table(n=12):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "case": [1] * n,
        "run": np.arange(1, n + 1),
        "vb_um": np.linspace(0, 400, n),
        "cum_time": np.arange(1, n + 1) * 30.0,
        "material": [1.0] * n,
        "feed": [0.5] * n,
        "doc": [1.5] * n,
        "rpm": [826.0] * n,
        "condition": ["1.0_ap1.5_f0.5"] * n,
    })
    for channel in ("vib_table", "vib_spindle", "AE_table", "AE_spindle", "smcAC"):
        for suffix in ("rms", "order_3", "order_3_ratio"):
            frame[f"{channel}_{suffix}"] = rng.uniform(1, 2, n)
    return frame


class TestCollapse:
    def test_produces_mean_and_max_per_group(self):
        frame = pd.DataFrame({
            "vib_x_rms": [1.0, 2.0],
            "vib_y_rms": [3.0, 4.0],
            "vib_z_rms": [5.0, 6.0],
        })
        result = collapse_channel_groups(frame, {"vibration": ("vib_x", "vib_y", "vib_z")})

        assert list(result["vibration_rms_mean"]) == [3.0, 4.0]
        assert list(result["vibration_rms_max"]) == [5.0, 6.0]

    def test_column_count_is_independent_of_channel_count(self):
        """PHM'de 3 titreşim kanalı, NASA'da 2 - ikisi de aynı sayıda sütun vermeli."""
        three = collapse_channel_groups(
            pd.DataFrame({f"vib_{a}_rms": [1.0] for a in "xyz"}),
            {"vibration": ("vib_x", "vib_y", "vib_z")},
        )
        two = collapse_channel_groups(
            pd.DataFrame({"vib_table_rms": [1.0], "vib_spindle_rms": [1.0]}),
            {"vibration": ("vib_table", "vib_spindle")},
        )
        assert list(three.columns) == list(two.columns)


class TestPrepare:
    def test_phm_feed_converted_to_mm_per_rev(self):
        """PHM mm/dk veriyor, NASA mm/dev - ortak birime çevrilmeli."""
        shared = prepare_phm(_phm_table())
        assert shared["feed_mm_per_rev"].iloc[0] == pytest.approx(1555.0 / 10400.0)

    def test_phm_material_gets_its_own_code(self):
        """Paslanmaz, NASA'nın iki malzemesinden de farklı bir kategori."""
        shared = prepare_phm(_phm_table())
        assert (shared["material"] == PHM_MATERIAL_CODE).all()
        assert PHM_MATERIAL_CODE not in {1, 2}

    def test_force_channels_are_dropped(self):
        """Kuvvet yalnız PHM'de var - ortak uzayda kullanılamaz."""
        shared = prepare_phm(_phm_table())
        assert not [c for c in shared.columns if "force" in c]

    def test_current_channels_are_dropped(self):
        """Motor akımı yalnız NASA'da var."""
        shared = prepare_nasa(_nasa_table())
        assert not [c for c in shared.columns if "smc" in c]

    def test_tool_ids_cannot_collide_between_datasets(self):
        phm = prepare_phm(_phm_table())
        nasa = prepare_nasa(_nasa_table())
        assert not set(phm["tool"]) & set(nasa["tool"])


class TestSharedTable:
    def test_both_datasets_present(self):
        table = build_shared_table(_phm_table(), _nasa_table(), normalise=False)
        assert set(table["source"]) == {"phm", "nasa"}

    def test_sensor_columns_are_identical_for_both_sources(self):
        """Birleştirmenin anlamlı olması için iki tarafın da aynı sütunları
        doldurması gerekir; biri boş kalırsa model kaynağı ayırt eder."""
        table = build_shared_table(_phm_table(), _nasa_table(), normalise=False)
        sensors = sensor_columns_of(table)

        for source in ("phm", "nasa"):
            subset = table[table["source"] == source]
            assert subset[sensors].notna().all().all(), f"{source} tarafında boş sütun var"

    def test_parameter_and_time_columns_survive(self):
        table = build_shared_table(_phm_table(), _nasa_table(), normalise=False)
        for column in [*PARAMETER_COLUMNS, TIME_COLUMN]:
            assert column in table.columns

    def test_normalisation_starts_every_tool_near_one(self):
        """Taban normalizasyonundan sonra her takım kendi 1'inden başlamalı -
        tezgâhlar arası kazanç farkını yok eden mekanizma budur."""
        table = build_shared_table(
            _phm_table(), _nasa_table(), normalise=True, n_baseline=5
        )
        sensors = sensor_columns_of(table)

        for tool, subset in table.groupby("tool"):
            head = subset.sort_values("step").head(5)[sensors]
            assert head.median().median() == pytest.approx(1.0, abs=0.05)

    def test_no_infinities_after_normalisation(self):
        table = build_shared_table(_phm_table(), _nasa_table(), normalise=True)
        sensors = sensor_columns_of(table)
        assert not np.isinf(table[sensors].to_numpy(dtype=float)).any()