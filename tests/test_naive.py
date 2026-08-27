"""Naif taban modelin testleri."""

from __future__ import annotations

import numpy as np
import pytest

from tcm.models import NaiveWearBaseline, enforce_monotonic


class TestNaiveBaseline:
    def test_learns_monotonic_trend(self):
        cuts = np.arange(1, 101)
        wear = cuts * 1.5

        model = NaiveWearBaseline().fit(cuts, wear)
        predictions = model.predict(cuts)

        assert np.allclose(predictions, wear, atol=1e-6)

    def test_predictions_never_decrease(self):
        rng = np.random.default_rng(42)
        cuts = np.arange(1, 51)
        wear = cuts * 2.0 + rng.normal(0, 10, size=cuts.size)  # gürültülü

        predictions = NaiveWearBaseline().fit(cuts, wear).predict(cuts)

        assert np.all(np.diff(predictions) >= -1e-9)

    def test_extrapolation_is_clipped_not_extended(self):
        cuts = np.arange(1, 21)
        wear = cuts * 1.0

        model = NaiveWearBaseline().fit(cuts, wear)
        beyond = model.predict([100])

        assert beyond[0] == pytest.approx(wear.max())

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            NaiveWearBaseline().predict([1, 2, 3])

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="Boyutlar uyuşmuyor"):
            NaiveWearBaseline().fit([1, 2, 3], [1.0, 2.0])


class TestEnforceMonotonic:
    def test_removes_dips(self):
        result = enforce_monotonic([10, 30, 20, 25, 60, 55])
        assert list(result) == [10, 30, 30, 30, 60, 60]

    def test_leaves_monotonic_sequence_untouched(self):
        values = [10.0, 20.0, 30.0]
        assert list(enforce_monotonic(values)) == values

    def test_never_lowers_a_value(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=200).cumsum()
        assert np.all(enforce_monotonic(values) >= values)
