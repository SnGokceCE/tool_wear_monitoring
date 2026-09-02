"""Sabit bölme betiğinin testleri (Faz 12).

En kritik olan ``test_random_split_keeps_original_index``: derin model kolu
ham sinyal dizisine satır İNDEKSİYLE erişiyor. Bölme fonksiyonu indeksi
sıfırlarsa sinyaller etiketlerle eşleşmez ve model sessizce yanlış veriyle
eğitilir - hata vermez, sadece kötü sonuç verir.

Bu gerçekten oldu ve 308 µm'lik bir MAE olarak göründü; modelin
başarısızlığı sanıldı.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts" / "run_holdout_split.py"
    spec = importlib.util.spec_from_file_location("run_holdout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# NASA'nın gerçek takım/koşu dağılımı (145 koşu, 15 takım). Bölme boyutları
# bu dağılıma bağlı olduğu için test verisi de aynı olmalı.
TOOL_SIZES = [(1, 13), (2, 13), (3, 14), (4, 7), (5, 6), (7, 7), (8, 5),
              (9, 9), (10, 10), (11, 20), (12, 12), (13, 13), (14, 7),
              (15, 6), (16, 3)]


@pytest.fixture
def data():
    """Gerçek dağılımı taklit eden tablo - takıma göre sıralı, 145 satır."""
    rows = []
    for case, n_runs in TOOL_SIZES:
        for run in range(1, n_runs + 1):
            rows.append({"case": float(case), "run": float(run),
                         "vb_um": 20.0 * run, "material": 1.0})
    frame = pd.DataFrame(rows).reset_index(drop=True)
    assert len(frame) == 145
    return frame


class TestToolSplit:
    def test_train_and_test_share_no_tool(self, script, data):
        parts = script._split_by_tool(data)
        shared = set(parts["eğitim"]["case"]) & set(parts["test"]["case"])
        assert not shared

    def test_validation_is_also_disjoint(self, script, data):
        parts = script._split_by_tool(data)
        assert not set(parts["eğitim"]["case"]) & set(parts["doğrulama"]["case"])
        assert not set(parts["test"]["case"]) & set(parts["doğrulama"]["case"])

    def test_every_row_lands_in_exactly_one_part(self, script, data):
        parts = script._split_by_tool(data)
        total = sum(len(p) for p in parts.values())
        assert total == len(data)
        indices = np.concatenate([p.index.to_numpy() for p in parts.values()])
        assert len(np.unique(indices)) == len(data)


class TestRandomSplit:
    def test_index_label_identifies_the_same_row(self, script, data):
        """REGRESYON: indeks etiketi orijinal tabloda AYNI satırı göstermeli.

        Derin model ``signals[part.index]`` ile ham sinyale erişiyor. İndeks
        sıfırlanırsa etiket başka bir satıra işaret eder; sinyal ile etiket
        eşleşmez ve model sessizce yanlış veriyle eğitilir.

        DİKKAT - bu testin ilk hali işe yaramıyordu: yalnızca indekslerin
        0..144 aralığını kapladığını kontrol ediyordu, ki sıfırlanmış indeks
        de bu koşulu sağlıyor. Ayırt edici kontrol, indeksle ORİJİNAL tablodan
        çekilen satırın parçadaki satırla aynı olmasıdır.
        """
        for split_name, parts in [
            ("rastgele", script._split_random(data, seed=42)),
            ("takım bazlı", script._split_by_tool(data)),
        ]:
            for name, part in parts.items():
                fetched = data.loc[part.index]
                np.testing.assert_array_equal(
                    fetched["case"].to_numpy(), part["case"].to_numpy(),
                    err_msg=f"{split_name} / {name}: indeks yanlış satırı gösteriyor")
                np.testing.assert_array_equal(
                    fetched["run"].to_numpy(), part["run"].to_numpy(),
                    err_msg=f"{split_name} / {name}: indeks yanlış satırı gösteriyor")

    def test_signal_lookup_stays_aligned(self, script, data):
        """Sinyal dizisinden indeksle çekilen satır, etiketiyle eşleşmeli.

        Sinyal yerine her satırın (takım, koşu) çiftini taşıyan sahte bir
        dizi kullanılıyor; hiza bozulursa değerler tutmaz.
        """
        fake_signals = data[["case", "run"]].to_numpy(dtype=float)

        for split_name, parts in [
            ("rastgele", script._split_random(data, seed=42)),
            ("takım bazlı", script._split_by_tool(data)),
        ]:
            for name, part in parts.items():
                picked = fake_signals[part.index.to_numpy()]
                expected = part[["case", "run"]].to_numpy(dtype=float)
                np.testing.assert_array_equal(
                    picked, expected,
                    err_msg=f"{split_name} / {name}: sinyal-satır hizası bozuk")

    def test_random_split_actually_mixes_tools(self, script, data):
        """Rastgele bölme takım sınırını GÖZETMEMELİ - kurgunun tanımı bu.

        Bu test, rastgele bölmenin sessizce bir takım bölmesine dönüşmediğini
        garanti eder. İndeks sıfırlandığında tam olarak bu oluyordu.
        """
        parts = script._split_random(data, seed=42)
        shared = set(parts["eğitim"]["case"]) & set(parts["test"]["case"])
        assert shared, "rastgele bölme takım bölmesine dönüşmüş"

    def test_sizes_match_targets(self, script, data):
        parts = script._split_random(data, seed=42)
        assert len(parts["eğitim"]) == script.TARGET_SIZES["eğitim"]
        assert len(parts["doğrulama"]) == script.TARGET_SIZES["doğrulama"]
