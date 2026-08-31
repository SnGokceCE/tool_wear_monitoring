"""Faz 05 KARAR bloğunun testleri.

Buradaki asıl test ``test_undecided_is_not_counted_as_a_win``: betikte gerçekten
bulunan bir hataya karşı regresyon testi. Özet bloğu kazananı yalnızca ortalama
MAE karşılaştırmasıyla sayıyordu, dolayısıyla tohumlar arası saçılımın farktan
büyük olduğu (yani hiçbir şey kanıtlanmamış) sınavları da "geçti" sayıyordu.

PyTorch gerektirmez - hüküm mantığı kasıtlı olarak ağdan ayrı bir modülde.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcm.evaluation.verdict import (
    FAILED,
    PASSED,
    UNDECIDED,
    count_decisive_wins,
    describe_wins,
    seed_stability_verdict,
)


class TestSeedStabilityVerdict:
    def test_clear_win_is_a_win(self):
        # fark 20 µm, saçılım 5 µm -> üstünlük gürültüden büyük
        assert seed_stability_verdict(120.0, 140.0, 5.0) == PASSED

    def test_clear_loss_is_a_loss(self):
        assert seed_stability_verdict(160.0, 140.0, 5.0) == FAILED

    def test_win_smaller_than_spread_is_undecided(self):
        """Asıl mesele: küçük bir üstünlük, büyük bir saçılım."""
        # fark 5,4 µm, saçılım 18,3 µm - Faz 05'in malzeme-dışı sınavı
        assert seed_stability_verdict(252.26, 257.65, 18.29) == UNDECIDED

    def test_loss_smaller_than_spread_is_also_undecided(self):
        """Kararsızlık simetriktir: küçük bir kayıp da kanıt değildir."""
        assert seed_stability_verdict(257.65, 252.26, 18.29) == UNDECIDED

    def test_spread_exactly_equal_to_gap_is_undecided(self):
        """Sınırda ihtiyatlı taraf seçilir."""
        assert seed_stability_verdict(100.0, 120.0, 20.0) == UNDECIDED

    def test_real_phase_05_results(self):
        """Faz 05'in üç tohumla ölçülen gerçek sayıları."""
        # koşul-dışı: fark 22,30 > saçılım 10,17 -> gerçek kazanç
        assert seed_stability_verdict(137.44, 159.74, 10.17) == PASSED
        # malzeme-dışı: fark 5,38 < saçılım 18,29 -> kanıt yok
        assert seed_stability_verdict(252.26, 257.65, 18.29) == UNDECIDED

    def test_zero_spread_falls_back_to_plain_comparison(self):
        """Tek tohumda saçılım 0 varsayılır - hüküm kesin GÖRÜNÜR ama değildir."""
        assert seed_stability_verdict(120.0, 140.0, 0.0) == PASSED

    def test_nan_input_is_undecided_not_a_crash(self):
        assert seed_stability_verdict(np.nan, 140.0, 5.0) == UNDECIDED
        assert seed_stability_verdict(120.0, 140.0, np.nan) == UNDECIDED


class TestCountDecisiveWins:
    def test_undecided_is_not_counted_as_a_win(self):
        """REGRESYON TESTİ - betikte gerçekten yapılan hata.

        Eski kod ``mae_cnn < mae_gbm`` sayıyordu. Faz 05'in malzeme-dışı
        sınavında CNN 252,26 < GBM 257,65 olduğu için o sınav "geçti"
        sayılıyordu - oysa saçılım (±18,29) farktan (5,38) üç kat büyük ve
        ortada kanıtlanmış bir üstünlük yok.
        """
        counts = count_decisive_wins([PASSED, UNDECIDED])
        assert counts["passed"] == 1
        assert counts["undecided"] == 1
        assert counts["total"] == 2

    def test_undecided_is_not_counted_as_a_loss_either(self):
        """Kararsızı kayba yazmak da yanlış olurdu - kendi başına bir sonuçtur."""
        counts = count_decisive_wins([UNDECIDED, UNDECIDED])
        assert counts["passed"] == 0
        assert counts["failed"] == 0
        assert counts["undecided"] == 2

    def test_all_three_outcomes_are_counted_separately(self):
        counts = count_decisive_wins([PASSED, PASSED, UNDECIDED, FAILED])
        assert counts == {"passed": 2, "undecided": 1, "failed": 1, "total": 4}

    def test_unknown_verdict_is_rejected(self):
        """Sessizce yok saymak yerine hata: sayım eksik kalmasın."""
        with pytest.raises(ValueError, match="Bilinmeyen hüküm"):
            count_decisive_wins(["belki"])


class TestDescribeWins:
    def test_mentions_undecided_when_present(self):
        line = describe_wins(
            count_decisive_wins([PASSED, UNDECIDED]), "CNN+GRU", "gradyan artırma"
        )
        assert "1/2" in line
        assert "KARARSIZ" in line

    def test_stays_quiet_when_everything_is_decisive(self):
        line = describe_wins(
            count_decisive_wins([PASSED, PASSED]), "CNN+GRU", "gradyan artırma"
        )
        assert "KARARSIZ" not in line
        assert "2/2" in line
