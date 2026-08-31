"""``scripts/run_decision_rule.py`` dış değerlendirmesinin regresyon testleri.

NEDEN BU DOSYA VAR
------------------
Alarm kilidinin takımlar arasına taşması hatası bu projede İKİ KEZ yapıldı.
Birincisi eşik seçiminde bulundu, düzeltildi ve ``test_decision.py`` içine
regresyon testi yazıldı.

Ama o test kütüphane fonksiyonunu (``alarm_flags``) sınıyordu; betiğin O
FONKSİYONU ÇAĞIRIP ÇAĞIRMADIĞINI sınamıyordu. Betik yanlış fonksiyonu
(``apply_consecutive``, gruplama olmadan) çağırmaya devam etti ve hata dış
değerlendirmede yaşamaya devam etti - kaçırılan aşınma sayısını üçte bir
gösterecek şekilde.

Ders: kütüphaneyi test etmek, kütüphaneyi doğru kullandığınızı test etmez.
Bu dosya betiği doğrudan çağırır.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcm.decision import alarm_flags, apply_consecutive
from tcm.evaluation.classification import classification_scores
from tcm.models.gbm import make_gbm_small

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIMIT = 300.0
SEED = 42
COLUMNS = ["cum_time"]


def _load_script():
    """Betiği modül olarak yükler - ``scripts/`` bir paket değil."""
    path = PROJECT_ROOT / "scripts" / "run_decision_rule.py"
    spec = importlib.util.spec_from_file_location("run_decision_rule_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Kilit taşmasının fark yaratacağı sentetik veri.

    Kurgu, gerçek malzeme-dışı sınavının sadeleştirilmiş hali:

      malzeme 1 : aşınma = 1,05 x kümülatif süre  (yavaş aşınan malzeme)
      malzeme 2 : aşınma = 2,50 x kümülatif süre  (hızlı aşınan malzeme)

    Model yalnızca ``cum_time`` görüyor ve malzemeler arası aşınma HIZI farkını
    bilemiyor. Malzeme 1'de eğitilip malzeme 2'de test edildiğinde sistematik
    olarak EKSİK tahmin ediyor - Faz 04b'de gerçek veride ölçülen davranışın
    ta kendisi.

    Eksik tahmin kaçırılan aşınma üretir. Aynı katlamada uzun süre kesen bir
    takım eşiği aşıp alarm verir; kilit katlamanın tamamına uygulanırsa o
    alarm, hiç alarm vermemesi gereken takımların satırlarını da "alarm
    verildi" sayar ve kaçırılan aşınmayı gizler. İki uygulama burada ayrışır.
    """
    plan = [
        # (malzeme, geçiş başına süre, aşınma hızı)
        (1.0, 60.0, 1.05),
        (1.0, 40.0, 1.05),
        (1.0, 20.0, 1.05),
        (2.0, 60.0, 2.50),
        (2.0, 15.0, 2.50),
        (2.0, 8.0, 2.50),
    ]

    rows = []
    for case, (material, rate, wear_rate) in enumerate(plan, start=1):
        cumulative = 0.0
        for run in range(1, 11):
            cumulative += rate
            rows.append({
                "case": float(case),
                "run": float(run),
                "material": material,
                "cum_time": cumulative,
                # Aşınma kümülatif süreyle monoton artar - fiziğe uygun.
                "vb_um": cumulative * wear_rate,
            })

    frame = pd.DataFrame(rows)
    frame["worn"] = frame["vb_um"] >= LIMIT
    return frame


def _reference_scores(data: pd.DataFrame, per_tool: bool) -> dict[str, float]:
    """Dış değerlendirmenin iki olası uygulaması - referans olarak yeniden kurulur.

    ``per_tool=True``  : kilit her takım içinde (DOĞRU)
    ``per_tool=False`` : kilit katlamanın tamamında (ESKİ HATA)
    """
    truths, flags = [], []
    for held_out in sorted(data["material"].unique()):
        train = data[data["material"] != held_out]
        test = data[data["material"] == held_out].sort_values("run")

        model = make_gbm_small(random_state=SEED)
        model.fit(train[COLUMNS], train["vb_um"])
        predicted = np.asarray(model.predict(test[COLUMNS]), dtype=float)

        if per_tool:
            flag = alarm_flags(predicted, LIMIT, 1, test["case"].to_numpy())
        else:
            flag = apply_consecutive(predicted >= LIMIT, 1)

        truths.append(test["worn"].to_numpy())
        flags.append(flag)

    return classification_scores(np.concatenate(truths), np.concatenate(flags))


class TestOuterEvaluationLatching:
    def test_the_scenario_actually_discriminates(self, data):
        """Önce testin anlamlı olduğunu kanıtla.

        İki uygulama aynı sonucu verseydi bu dosya hiçbir şeyi korumazdı -
        hata geri gelse bile test geçerdi. Bu yüzden ilk iş, seçilen veride
        ikisinin GERÇEKTEN ayrıştığını göstermek.
        """
        per_tool = _reference_scores(data, per_tool=True)
        fold_wide = _reference_scores(data, per_tool=False)
        assert per_tool["missed_worn"] != fold_wide["missed_worn"], \
            "sentetik veri iki uygulamayı ayırt etmiyor - test değersiz"

    def test_outer_evaluation_latches_per_tool(self, script, data):
        """REGRESYON: betik kilidi takım bazında uygulamalı."""
        scores = script._run(
            data, "material", COLUMNS, LIMIT, SEED,
            cost_missed=5.0, cost_false=1.0, tune=False, consecutive=1,
        )
        expected = _reference_scores(data, per_tool=True)

        assert scores["missed_worn"] == expected["missed_worn"]
        assert scores["false_alarms"] == expected["false_alarms"]

    def test_outer_evaluation_does_not_latch_across_the_fold(self, script, data):
        """REGRESYON: eski hatalı davranışa geri dönülmemeli.

        Bu, önceki testin aynadaki hali. Ayrı yazılmasının sebebi hata
        mesajının doğrudan olması: birisi ``apply_consecutive``e geri dönerse
        "kilit katlamaya taşıyor" diye okunabilir bir başarısızlık versin.
        """
        scores = script._run(
            data, "material", COLUMNS, LIMIT, SEED,
            cost_missed=5.0, cost_false=1.0, tune=False, consecutive=1,
        )
        buggy = _reference_scores(data, per_tool=False)

        assert scores["missed_worn"] != buggy["missed_worn"], \
            "kilit katlamanın tamamına uygulanıyor - kaçırılan aşınma eksik sayılıyor"

    def test_fold_wide_latching_understates_missed_wear(self, data):
        """Hatanın YÖNÜ: kaçırılan aşınmayı olduğundan AZ gösterir.

        Yönü sabitlemek önemli çünkü hatanın tehlikeli tarafı bu. Kilit
        taştığında satırlar "alarm verildi" sayılır; gerçekte aşınmış olan
        ve alarm almayan satırlar kaçırılan aşınma olarak sayılmaz. Yani
        hata sistemi olduğundan GÜVENLİ gösterir.
        """
        per_tool = _reference_scores(data, per_tool=True)
        fold_wide = _reference_scores(data, per_tool=False)
        assert fold_wide["missed_worn"] < per_tool["missed_worn"]

    def test_tuned_path_also_latches_per_tool(self, script, data):
        """Eşik iç ÇD'den seçilirken de dış kilit takım bazında kalmalı."""
        scores = script._run(
            data, "material", COLUMNS, LIMIT, SEED,
            cost_missed=5.0, cost_false=1.0, tune=True, consecutive=1,
        )
        # Ayarlı eşik sabit eşikten farklı olabilir; burada sınanan şey
        # kaçırılan aşınmanın hatalı uygulamadaki kadar düşük OLMAMASI.
        buggy = _reference_scores(data, per_tool=False)
        assert scores["missed_worn"] >= buggy["missed_worn"]
        assert 0.0 <= scores["worn_recall"] <= 1.0


class TestInnerSelectionStillUsesSharedCode:
    def test_inner_predictions_delegates_to_the_library(self, script, data):
        """Faz 09 iç ÇD'si ile Faz 06 kalibrasyonu aynı kodu çağırmalı."""
        from tcm.decision import oof_predictions

        truth, pred, groups = script._inner_predictions(data, "case", COLUMNS, SEED)
        reference = oof_predictions(
            data, "case", COLUMNS, lambda: make_gbm_small(random_state=SEED),
            min_groups=3,
        )
        np.testing.assert_allclose(truth, reference.y_true)
        np.testing.assert_allclose(pred, reference.y_pred)
        np.testing.assert_array_equal(groups, reference.groups)
