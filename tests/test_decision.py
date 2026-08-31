"""Karar mantığının testleri (Faz 09)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcm.decision import (
    alarm_cost,
    apply_consecutive,
    choose_consecutive,
    choose_threshold,
    oof_predictions,
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
        from tcm.decision import alarm_flags
        flags = alarm_flags(pred, threshold, 1, groups)
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


class TestOutOfFoldPredictions:
    """Eşiğin sızıntısız seçilmesinin dayanağı (Faz 06 ve Faz 09 ortak kodu).

    Eşik, modelin hatasına göre ayarlanır. Model kendi eğitim verisinde
    neredeyse hatasız göründüğü için, eşik oradan seçilirse aşınma sınırının
    kendisine yapışır ve karar kuralı hiçbir şey düzeltmez. Katlama dışı
    tahmin üretmenin tek sebebi budur.
    """

    def _table(self, n_cases=6, n_runs=8):
        rows = []
        for case in range(1, n_cases + 1):
            for run in range(1, n_runs + 1):
                rows.append({
                    "case": float(case),
                    "run": float(run),
                    "material": 1.0 if case % 2 else 2.0,
                    "x": float(run) + 0.1 * case,
                    "vb_um": 40.0 * run,
                })
        return pd.DataFrame(rows)

    class _Recorder:
        """Hangi satırlarla eğitildiğini kaydeden sahte model."""

        seen: list[set] = []

        def fit(self, X, y, **_):
            type(self).seen.append(set(X["x"].round(6)))
            self._mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self._mean)

    def test_every_row_gets_a_prediction_exactly_once(self):
        data = self._table()
        oof = oof_predictions(
            data, "case", ["x"], lambda: self._Recorder(), min_groups=3
        )
        assert len(oof.y_pred) == len(data)
        assert len(oof.y_true) == len(data)
        assert oof.n_folds == data["case"].nunique()

    def test_no_row_is_used_to_predict_itself(self):
        """Sızıntı testi: bir katlamanın test satırları o katlamanın
        eğitiminde bulunmamalı."""
        data = self._table()
        self._Recorder.seen = []

        oof = oof_predictions(
            data, "case", ["x"], lambda: self._Recorder(), min_groups=3
        )

        for held_out, train_rows in zip(sorted(data["case"].unique()),
                                        self._Recorder.seen):
            test_x = set(data.loc[data["case"] == held_out, "x"].round(6))
            assert not (test_x & train_rows), \
                f"katlama {held_out}: test satırları eğitime sızmış"
        assert oof.split_column == "case"

    def test_falls_back_when_there_are_too_few_groups(self):
        """İki malzeme varken min_groups=3 istersek ``case``'e düşülür."""
        data = self._table()
        oof = oof_predictions(
            data, "material", ["x"], lambda: self._Recorder(),
            fallback_column="case", min_groups=3,
        )
        assert oof.fell_back
        assert oof.split_column == "case"
        assert oof.requested_column == "material"

    def test_min_groups_two_keeps_the_material_split(self):
        """Teslim senaryosu: dökme demirde eğit, çelikte ölç."""
        data = self._table()
        oof = oof_predictions(
            data, "material", ["x"], lambda: self._Recorder(), min_groups=2
        )
        assert not oof.fell_back
        assert oof.split_column == "material"
        assert oof.n_folds == 2

    def test_latch_groups_are_tools_not_folds(self):
        """Alarm kilidi takım bazında işlemeli - katlama bazında değil."""
        data = self._table()
        oof = oof_predictions(
            data, "material", ["x"], lambda: self._Recorder(), min_groups=2
        )
        assert set(np.unique(oof.groups)) == set(data["case"].unique())

    def test_describe_mentions_the_fallback(self):
        data = self._table()
        oof = oof_predictions(
            data, "material", ["x"], lambda: self._Recorder(), min_groups=3
        )
        assert "yeterli grup yok" in oof.describe()
