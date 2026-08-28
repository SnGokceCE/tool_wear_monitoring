"""Karar mantığının testleri (Faz 09)."""

from __future__ import annotations

import numpy as np
import pytest

from tcm.decision import (
    alarm_cost,
    apply_consecutive,
    choose_consecutive,
    choose_threshold,
)


class TestAlarmCost:
    def test_missed_wear_costs_more_than_false_alarm(self):
        truth = [True, False]
        missed = alarm_cost(truth, [False, False], cost_missed=5, cost_false_alarm=1)
        false_alarm = alarm_cost(truth, [True, True], cost_missed=5, cost_false_alarm=1)
        assert missed > false_alarm

    def test_perfect_prediction_costs_nothing(self):
        truth = [True, True, False, False]
        assert alarm_cost(truth, truth) == 0.0

    def test_cost_ratio_is_respected(self):
        # 1 kaçırma, 0 yanlış alarm
        assert alarm_cost([True], [False], cost_missed=7, cost_false_alarm=1) == 7.0
        # 0 kaçırma, 1 yanlış alarm
        assert alarm_cost([False], [True], cost_missed=7, cost_false_alarm=1) == 1.0


class TestConsecutive:
    def test_alarm_latches_once_triggered(self):
        """Aşınma geri dönmez, alarm da sönmemeli."""
        result = apply_consecutive([False, True, False, False], k=1)
        assert list(result) == [False, True, True, True]

    def test_requires_k_consecutive_hits(self):
        # k=2: tek bir sıçrama alarm vermemeli
        assert list(apply_consecutive([False, True, False, True, True], k=2)) == \
            [False, False, False, False, True]

    def test_k_one_is_immediate(self):
        assert list(apply_consecutive([True, False], k=1)) == [True, True]

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="k en az 1"):
            apply_consecutive([True], k=0)


class TestThresholdSelection:
    def _wear_curve(self, n=20, start=100.0, end=500.0):
        return np.linspace(start, end, n)

    def test_picks_lower_threshold_when_missing_is_expensive(self):
        """Kaçırma pahalıysa eşik güvenli tarafa, yani AŞAĞI kaymalı.

        Not: tahmin gürültüsüz olsaydı hatasız bir eşik bulunurdu ve maliyet
        oranı hiç fark etmezdi. Takasın ortaya çıkması için modelin hata
        yapıyor olması gerekiyor - gerçek durum da budur.
        """
        rng = np.random.default_rng(3)
        truths, preds, groups = [], [], []
        for tool in range(8):
            curve = self._wear_curve()
            truths.append(curve)
            preds.append(curve + rng.normal(0, 60, curve.size))
            groups.append(np.full(curve.size, tool))

        truth = np.concatenate(truths)
        pred = np.concatenate(preds)
        group = np.concatenate(groups)

        expensive = choose_threshold(truth, pred, 300.0, groups=group,
                                     cost_missed=20.0, cost_false_alarm=1.0)
        cheap = choose_threshold(truth, pred, 300.0, groups=group,
                                 cost_missed=1.0, cost_false_alarm=20.0)
        assert expensive < cheap

    def test_threshold_stays_within_search_span(self):
        truth = self._wear_curve()
        threshold = choose_threshold(truth, truth, 300.0, search_span=0.2)
        assert 240.0 <= threshold <= 360.0

    def test_perfect_model_keeps_threshold_near_limit(self):
        truth = self._wear_curve()
        threshold = choose_threshold(truth, truth, 300.0, cost_missed=5.0)
        assert threshold == pytest.approx(300.0, abs=15.0)


class TestLatchingAcrossToolsRegression:
    """Gerçekten yapılan bir hatanın tekrar etmemesi için.

    apply_consecutive alarmı KİLİTLER. Bu tek bir takımın ömrü içinde
    doğrudur. Ama birden çok takımın tahminleri arka arkaya eklenmiş bir
    dizide uygulanırsa, ilk takımdaki alarm sonraki bütün takımları da
    alarmda gösterir.

    Bu hata eşik seçimini bozmuştu: optimizasyon, erken alarmların her şeyi
    zehirlemesinden kaçınmak için eşiği 300'den 421 µm'ye çıkarıyordu -
    yani tam ters yöne, güvensiz tarafa.
    """

    def test_alarm_does_not_leak_between_tools(self):
        # İki takım: birincisi aşınıyor, ikincisi yepyeni
        truth = np.array([100.0, 500.0, 100.0, 110.0])
        pred = np.array([100.0, 500.0, 100.0, 110.0])
        groups = np.array(["a", "a", "b", "b"])

        threshold = choose_threshold(truth, pred, 300.0, groups=groups)

        # Gruplar dikkate alınınca ikinci takım alarmsız kalabilmeli
        from tcm.decision import _flags_by_group
        flags = _flags_by_group(pred, threshold, 1, groups)
        assert not flags[2] and not flags[3], "alarm ikinci takıma taştı"

    def test_without_groups_alarm_leaks_as_documented(self):
        """Grup verilmezse kilitlenme tüm diziyi kapsar - beklenen davranış."""
        pred = np.array([100.0, 500.0, 100.0, 110.0])
        flags = apply_consecutive(pred >= 300.0, 1)
        assert flags[2] and flags[3]

    def test_grouped_selection_prefers_safe_threshold(self):
        """Gruplu seçim, eşiği sınırın üstüne itmemeli."""
        rng = np.random.default_rng(0)
        truths, preds, groups = [], [], []
        for tool in range(6):
            curve = np.linspace(100, 500, 12)
            truths.append(curve)
            preds.append(curve - 30 + rng.normal(0, 10, curve.size))
            groups.append(np.full(curve.size, tool))

        threshold = choose_threshold(
            np.concatenate(truths), np.concatenate(preds), 300.0,
            cost_missed=5.0, cost_false_alarm=1.0,
            groups=np.concatenate(groups),
        )
        assert threshold < 300.0, "eksik tahmin eden modelde eşik yukarı çıkmamalı"


class TestConsecutiveSelection:
    def test_returns_a_candidate(self):
        truth = np.linspace(100, 500, 20)
        pred = truth - 20
        k = choose_consecutive(truth, pred, 280.0, 300.0, candidates=(1, 2, 3))
        assert k in (1, 2, 3)
