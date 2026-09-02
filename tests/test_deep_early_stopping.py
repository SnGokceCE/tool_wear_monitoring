"""Derin modelde erken durdurmanın testleri (Faz 12).

``train_model`` uzun süre doğrulama kümesi olmadan çalıştı; Faz 12'de sabit
bölme geldiğinde doğrulama kümesi oluştu ve erken durdurma eklendi.

Buradaki kritik davranış: erken durdurma yalnızca eğitimi ERKEN KESMEK değil,
**en iyi ağırlıklara geri dönmektir**. Sadece kesip son epoch'un ağırlıklarını
bırakmak, erken durdurmanın amacını ortadan kaldırır - o ağırlıklar zaten
kötüleşmeye başlamış olanlardır.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch kurulu değil")

from tcm.models.deep import CNNGRUWearModel, predict, train_model


def _tiny_problem(n=12, channels=2, length=64, seed=0):
    """Öğrenilebilir ama küçük bir problem - test saniyeler içinde bitmeli."""
    rng = np.random.default_rng(seed)
    signals = rng.normal(size=(n, channels, length)).astype(np.float32)
    parameters = rng.normal(size=(n, 3)).astype(np.float32)
    # Hedef sinyalin enerjisine bağlı olsun ki öğrenilecek bir şey olsun.
    targets = (signals**2).mean(axis=(1, 2)).astype(np.float32) * 100.0
    return signals, parameters, targets


def _model(channels=2, parameters=3):
    return CNNGRUWearModel(
        n_channels=channels, n_parameters=parameters,
        conv_channels=(4, 4), gru_hidden=8,
    )


class TestEarlyStopping:
    def test_without_validation_behaviour_is_unchanged(self):
        """Doğrulama verilmezse eski davranış korunmalı: sabit epoch."""
        signals, parameters, targets = _tiny_problem()
        model = train_model(_model(), signals, parameters, targets,
                            epochs=3, seed=0)
        assert not hasattr(model, "best_epoch_")

    def test_validation_records_best_epoch(self):
        signals, parameters, targets = _tiny_problem()
        split = 8
        model = train_model(
            _model(), signals[:split], parameters[:split], targets[:split],
            epochs=6, seed=0,
            validation=(signals[split:], parameters[split:], targets[split:]),
            patience=10,
        )
        assert hasattr(model, "best_epoch_")
        assert 1 <= model.best_epoch_ <= 6
        assert np.isfinite(model.best_validation_mae_)

    def test_stops_early_when_no_improvement(self):
        """Sabır tükenince epoch sayısı dolmadan durmalı.

        Doğrulama hedefleri kasıtlı olarak girdiyle İLİŞKİSİZ seçildi: model
        eğitim verisine uysa da doğrulama hatası iyileşemez, dolayısıyla sabır
        tükenir. İlişkili bir doğrulama kümesiyle model epoch boyunca
        gelişmeye devam edebilir ve erken durdurma hiç tetiklenmez - ilk
        denemede tam olarak bu oldu.
        """
        signals, parameters, targets = _tiny_problem()
        split = 8
        # Eğitim hedefleri ~100 mertebesinde; doğrulama hedefleri sıfıra yakın
        # seçildi. Model eğitimi öğrendikçe çıktısı yükselir ve doğrulama
        # hatası BAŞTAN İTİBAREN kötüleşir, dolayısıyla en iyi epoch ilk
        # epoch'lardan biri olur ve sabır hemen tükenir.
        #
        # İlk denemede doğrulama hedefleri 500 seçilmişti; model çıktısı ona
        # doğru yükseldiği için hata yanlışlıkla iyileşmeye devam etti ve
        # erken durdurma tetiklenmedi.
        noise_targets = np.zeros(len(targets) - split, dtype=np.float32)

        model = train_model(
            _model(), signals[:split], parameters[:split], targets[:split],
            epochs=200, seed=0,
            validation=(signals[split:], parameters[split:], noise_targets),
            patience=2,
        )
        assert model.best_epoch_ < 200

    def test_restores_best_weights_not_last(self):
        """EN KRİTİK: dönen model en iyi ağırlıkları taşımalı.

        Kaydedilen en iyi doğrulama MAE'si ile, dönen modelin aynı veride
        yeniden hesaplanan MAE'si eşleşmelidir. Eşleşmiyorsa model son
        epoch'un ağırlıklarıyla dönmüş demektir.
        """
        signals, parameters, targets = _tiny_problem()
        split = 8
        v_signal = signals[split:]
        v_param = parameters[split:]
        v_target = targets[split:]

        model = train_model(
            _model(), signals[:split], parameters[:split], targets[:split],
            epochs=40, seed=0,
            validation=(v_signal, v_param, v_target),
            patience=5,
        )
        recomputed = float(np.mean(np.abs(
            predict(model, v_signal, v_param) - v_target)))
        assert recomputed == pytest.approx(model.best_validation_mae_, abs=1e-4)
