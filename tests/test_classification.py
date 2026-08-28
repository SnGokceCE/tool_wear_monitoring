"""worn / unworn sınıflandırma metriklerinin testleri."""

from __future__ import annotations

import numpy as np
import pytest

from tcm.evaluation.classification import (
    classification_scores,
    confusion,
    majority_baseline,
)

W, U = True, False  # worn / unworn


class TestConfusion:
    def test_counts_all_four_cells(self):
        result = confusion([W, W, U, U], [W, U, W, U])
        assert result == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}

    def test_perfect_prediction_has_no_errors(self):
        truth = [W, U, W, U, W]
        result = confusion(truth, truth)
        assert result["fn"] == 0 and result["fp"] == 0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="Boyutlar uyuşmuyor"):
            confusion([W, U], [W])


class TestScores:
    def test_worn_recall_counts_caught_worn_tools(self):
        # 4 aşınmış takımın 3'ü yakalandı
        scores = classification_scores([W, W, W, W, U], [W, W, W, U, U])
        assert scores["worn_recall"] == pytest.approx(0.75)
        assert scores["missed_worn"] == 1.0

    def test_false_alarms_counted_separately(self):
        scores = classification_scores([U, U, U, W], [W, W, U, W])
        assert scores["false_alarms"] == 2.0
        assert scores["missed_worn"] == 0.0

    def test_balanced_accuracy_is_mean_of_two_recalls(self):
        scores = classification_scores([W, W, U, U], [W, U, U, U])
        # worn_recall = 1/2, unworn_recall = 2/2
        assert scores["balanced_acc"] == pytest.approx(0.75)

    def test_precision_measures_alarm_reliability(self):
        # 3 kez "aşınmış" dedik, 2'si doğru
        scores = classification_scores([W, W, U, U], [W, W, W, U])
        assert scores["worn_precision"] == pytest.approx(2 / 3)


class TestWhyAccuracyMisleads:
    """Doğruluğun neden tek başına yetmediğinin testi.

    Bu, projenin metrik seçiminin gerekçesi - dengesiz sınıfta hiçbir şey
    öğrenmeyen bir model yüksek doğruluk alır ama tüm aşınmış takımları
    kaçırır.
    """

    def test_majority_baseline_scores_high_accuracy_but_catches_nothing(self):
        # 90 sağlam, 10 aşınmış
        truth = [U] * 90 + [W] * 10
        scores = majority_baseline(truth)

        assert scores["accuracy"] == pytest.approx(0.90)
        assert scores["worn_recall"] == pytest.approx(0.0)
        assert scores["missed_worn"] == 10.0
        # Dengeli doğruluk bu tuzağı yakalar
        assert scores["balanced_acc"] == pytest.approx(0.50)

    def test_balanced_accuracy_does_not_reward_majority_guessing(self):
        truth = [U] * 95 + [W] * 5
        assert majority_baseline(truth)["balanced_acc"] == pytest.approx(0.50)


class TestEdgeCases:
    def test_single_class_truth_gives_nan_for_missing_recall(self):
        scores = classification_scores([U, U, U], [U, U, U])
        assert np.isnan(scores["worn_recall"])
        assert scores["unworn_recall"] == pytest.approx(1.0)

    def test_no_alarms_gives_nan_precision(self):
        scores = classification_scores([W, U], [U, U])
        assert np.isnan(scores["worn_precision"])
