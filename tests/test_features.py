"""Öznitelik modüllerinin testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcm.features import (
    frame_features,
    max_usable_order,
    order_band_energies,
    spindle_frequency_hz,
    stable_region,
    time_domain_features,
    tooth_passing_frequency_hz,
)

PHM_FS = 50_000.0
PHM_RPM = 10_400.0
PHM_FLUTES = 3


class TestFrequencies:
    def test_spindle_frequency(self):
        assert spindle_frequency_hz(PHM_RPM) == pytest.approx(173.33, abs=0.01)

    def test_tooth_passing_frequency(self):
        assert tooth_passing_frequency_hz(PHM_RPM, PHM_FLUTES) == pytest.approx(520.0, abs=0.1)

    def test_phm_allows_many_orders(self):
        assert max_usable_order(PHM_FS, PHM_RPM) > 100

    def test_nasa_allows_only_about_nine_orders(self):
        """Transfer tavanını belirleyen sayı - Faz 07'nin temel kısıtı."""
        assert max_usable_order(250.0, 826.0) == pytest.approx(9.08, abs=0.05)


class TestTimeDomain:
    def test_rms_of_known_signal(self):
        values = np.array([3.0, -3.0, 3.0, -3.0])
        assert time_domain_features(values)["rms"] == pytest.approx(3.0)

    def test_peak_uses_absolute_value(self):
        assert time_domain_features([1.0, -5.0, 2.0])["peak"] == pytest.approx(5.0)

    def test_empty_input_returns_nan_not_crash(self):
        result = time_domain_features([])
        assert all(np.isnan(v) for v in result.values())

    def test_constant_signal_gives_zero_shape_features(self):
        """Sabit sinyalde çarpıklık/basıklık tanımsız - uyarı yerine 0 dönmeli."""
        result = time_domain_features(np.full(1000, 3.7))
        assert result["skew"] == 0.0
        assert result["kurtosis"] == 0.0
        assert result["std"] == pytest.approx(0.0)
        assert result["rms"] == pytest.approx(3.7)

    def test_near_constant_signal_does_not_produce_nan(self):
        values = np.full(1000, 5.0)
        values[0] = 5.0 + 1e-15
        result = time_domain_features(values)
        assert np.isfinite(result["skew"])
        assert np.isfinite(result["kurtosis"])

    def test_frame_features_are_flattened_per_channel(self):
        frame = pd.DataFrame({"force_x": [1.0, 2.0], "vib_x": [3.0, 4.0]})
        result = frame_features(frame)
        assert "force_x_rms" in result
        assert "vib_x_kurtosis" in result


class TestStableRegion:
    def test_keeps_middle_half_by_default(self):
        frame = pd.DataFrame({"a": range(100)})
        trimmed = stable_region(frame)
        assert len(trimmed) == 50
        assert trimmed["a"].iloc[0] == 25

    def test_keep_one_returns_everything(self):
        frame = pd.DataFrame({"a": range(10)})
        assert len(stable_region(frame, keep=1.0)) == 10

    def test_invalid_keep_rejected(self):
        with pytest.raises(ValueError, match="keep"):
            stable_region(pd.DataFrame({"a": [1]}), keep=0)


class TestOrderBands:
    def _tone(self, frequency, fs=PHM_FS, seconds=1.0):
        t = np.arange(0, seconds, 1 / fs)
        return np.sin(2 * np.pi * frequency * t)

    def test_energy_concentrates_at_the_injected_order(self):
        """Diş geçiş frekansına ton koyarsak enerji 3. mertebede çıkmalı."""
        tone = self._tone(tooth_passing_frequency_hz(PHM_RPM, PHM_FLUTES))
        bands = order_band_energies(tone, PHM_FS, PHM_RPM, max_order=8)

        others = [bands[f"order_{k}"] for k in range(1, 9) if k != 3]
        assert bands["order_3"] > 10 * max(others)

    def test_ratios_are_between_zero_and_one(self):
        rng = np.random.default_rng(0)
        bands = order_band_energies(rng.normal(size=50_000), PHM_FS, PHM_RPM)
        ratios = [v for k, v in bands.items() if k.endswith("_ratio")]
        assert all(0.0 <= r <= 1.0 for r in ratios)

    def test_orders_beyond_nyquist_are_nan(self):
        """NASA'nın 250 Hz'inde 9. mertebe ve üstü ölçülemez."""
        rng = np.random.default_rng(0)
        bands = order_band_energies(rng.normal(size=9000), 250.0, 826.0, max_order=12)
        assert np.isnan(bands["order_12"])
        assert not np.isnan(bands["order_1"])

    def test_short_signal_returns_empty(self):
        assert order_band_energies(np.zeros(4), PHM_FS, PHM_RPM) == {}
