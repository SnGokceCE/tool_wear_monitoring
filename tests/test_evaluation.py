"""Değerlendirme çatısının testleri.

Bu testler veri gerektirmez - Faz 00'da protokolün doğru kurulduğunu
doğrulamak için yazıldı.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcm.evaluation import (
    alarm_overshoot_um,
    crossing_delay_cuts,
    first_crossing,
    leave_one_cutter_out,
    mae_um,
    rmse_um,
    summarise,
)


class TestSplits:
    def test_leave_one_cutter_out_produces_one_fold_per_cutter(self):
        splits = leave_one_cutter_out(["c1", "c4", "c6"])
        assert len(splits) == 3
        assert [s.test for s in splits] == [("c1",), ("c4",), ("c6",)]

    def test_train_and_test_never_overlap(self):
        for split in leave_one_cutter_out(["c1", "c4", "c6"]):
            assert not set(split.train) & set(split.test)

    def test_every_cutter_is_tested_exactly_once(self):
        splits = leave_one_cutter_out(["c1", "c4", "c6"])
        tested = [c for split in splits for c in split.test]
        assert sorted(tested) == ["c1", "c4", "c6"]

    def test_rejects_single_cutter(self):
        with pytest.raises(ValueError, match="en az 2"):
            leave_one_cutter_out(["c1"])

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="tekrar"):
            leave_one_cutter_out(["c1", "c1", "c4"])


class TestMetrics:
    def test_perfect_prediction_scores_zero(self):
        values = [10.0, 20.0, 30.0]
        assert mae_um(values, values) == 0.0
        assert rmse_um(values, values) == 0.0

    def test_mae_is_mean_absolute_difference(self):
        assert mae_um([0.0, 0.0], [3.0, 5.0]) == pytest.approx(4.0)

    def test_rmse_penalises_outliers_more_than_mae(self):
        y_true = [0.0, 0.0, 0.0, 0.0]
        y_pred = [0.0, 0.0, 0.0, 8.0]
        assert rmse_um(y_true, y_pred) > mae_um(y_true, y_pred)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="Boyutlar uyuşmuyor"):
            mae_um([1.0, 2.0], [1.0])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="Boş dizi"):
            mae_um([], [])


class TestCrossing:
    def test_first_crossing_finds_threshold(self):
        assert first_crossing([10, 50, 100, 160, 200], 150) == 3

    def test_first_crossing_returns_none_when_never_reached(self):
        assert first_crossing([10, 20, 30], 150) is None

    def test_late_alarm_is_positive(self):
        y_true = [100, 140, 160, 180]  # gerçek geçiş: 2. dizin
        y_pred = [100, 130, 140, 155]  # tahmin geçişi: 3. dizin
        assert crossing_delay_cuts(y_true, y_pred, 150) == 1.0

    def test_early_alarm_is_negative(self):
        y_true = [100, 140, 160, 180]
        y_pred = [100, 155, 170, 190]
        assert crossing_delay_cuts(y_true, y_pred, 150) == -1.0

    def test_missed_alarm_counts_to_end_of_sequence(self):
        y_true = [100, 140, 160, 180]
        y_pred = [100, 110, 120, 130]
        assert crossing_delay_cuts(y_true, y_pred, 150) == 2.0

    def test_undefined_when_truth_never_crosses(self):
        assert np.isnan(crossing_delay_cuts([10, 20], [10, 20], 150))

    def test_summarise_returns_all_metrics(self):
        scores = summarise([100, 160], [110, 155], 150)
        assert set(scores) == {
            "mae_um", "rmse_um", "crossing_delay_cuts", "overshoot_um",
        }


class TestOvershoot:
    """Eğimden bağımsız alarm metriği.

    ``crossing_delay_cuts`` eğrinin eğimine bağlıdır: düz bölgede küçük bir
    µm hatası devasa bir geçiş kayması üretir. Bu metrik onun yerine
    "alarm çaldığında takım eşiği kaç µm aşmıştı" sorusunu sorar.
    """

    def test_late_alarm_is_positive(self):
        y_true = [100, 140, 160, 180]
        y_pred = [100, 130, 140, 155]  # alarm 3. dizinde, gerçek VB orada 180
        assert alarm_overshoot_um(y_true, y_pred, 150) == pytest.approx(30.0)

    def test_early_alarm_is_negative(self):
        y_true = [100, 140, 160, 180]
        y_pred = [100, 155, 170, 190]  # alarm 1. dizinde, gerçek VB orada 140
        assert alarm_overshoot_um(y_true, y_pred, 150) == pytest.approx(-10.0)

    def test_perfect_alarm_is_the_first_true_crossing(self):
        y_true = [100, 140, 160, 180]
        assert alarm_overshoot_um(y_true, y_true, 150) == pytest.approx(10.0)

    def test_missed_alarm_uses_final_wear(self):
        """Model hiç alarm vermezse bedeli, sona kadar biriken aşınmadır."""
        y_true = [100, 140, 160, 180]
        y_pred = [100, 110, 120, 130]
        assert alarm_overshoot_um(y_true, y_pred, 150) == pytest.approx(30.0)

    def test_same_delay_can_mean_very_different_damage(self):
        """Aynı '1 geçiş geç' iki durumda bambaşka fiziksel sonuç doğurur.

        Metriği değiştirmemizin sebebi bu: gecikme (geçiş) eğrinin eğimini
        göremez, overshoot (µm) görür.
        """
        # Dik eğri: geçiş başına 20 µm ilerliyor
        steep_true = [110, 130, 150, 170]
        steep_pred = [90, 110, 130, 150]
        # Düz eğri: geçiş başına 2 µm ilerliyor
        flat_true = [146, 148, 150, 152]
        flat_pred = [144, 146, 148, 150]

        # İkisi de tam olarak 1 geçiş geç alarm veriyor
        assert crossing_delay_cuts(steep_true, steep_pred, 150) == 1.0
        assert crossing_delay_cuts(flat_true, flat_pred, 150) == 1.0

        # Ama takımın gördüğü hasar 10 kat farklı
        assert alarm_overshoot_um(steep_true, steep_pred, 150) == pytest.approx(20.0)
        assert alarm_overshoot_um(flat_true, flat_pred, 150) == pytest.approx(2.0)
