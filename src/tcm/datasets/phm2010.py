"""PHM 2010 veri seti yükleyicisi (birincil veri seti).

Yapı: her kesici için bir klasör dolusu geçiş dosyası (``c_1_001.csv`` gibi,
başlıksız, 7 sütun) ve bir aşınma dosyası (``c1_wear.csv``).

Arşivin klasör yerleşimi kaynağa göre değişebildiği için burada sabit bir
yol beklemiyoruz: kök altındaki tüm dosyalar taranıp isim örüntüsünden
eşleştiriliyor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

CUT_FILE_PATTERN = re.compile(r"^c_(\d+)_(\d+)\.csv$", re.IGNORECASE)
WEAR_FILE_PATTERN = re.compile(r"^c(\d+)[_-]?wear\.csv$", re.IGNORECASE)

DEFAULT_CHANNELS = (
    "force_x",
    "force_y",
    "force_z",
    "vib_x",
    "vib_y",
    "vib_z",
    "ae_rms",
)


@dataclass(frozen=True)
class CutRef:
    """Tek bir kesme geçişine işaret eder."""

    cutter: str
    index: int
    path: Path


class PHM2010:
    """PHM 2010 verisine tembel (lazy) erişim.

    Sinyal dosyaları büyüktür (geçiş başına ~200 bin satır × 7 kanal), bu
    yüzden hiçbir şey önceden belleğe alınmaz; yalnızca istendiğinde okunur.
    """

    def __init__(
        self,
        root: str | Path,
        channels: tuple[str, ...] = DEFAULT_CHANNELS,
        wear_aggregation: str = "max",
    ) -> None:
        self.root = Path(root)
        self.channels = tuple(channels)
        if wear_aggregation not in {"max", "mean"}:
            raise ValueError(
                f"wear_aggregation 'max' veya 'mean' olmalı, '{wear_aggregation}' verildi"
            )
        self.wear_aggregation = wear_aggregation

        if not self.root.exists():
            raise FileNotFoundError(
                f"PHM 2010 kökü bulunamadı: {self.root}\n"
                "Veriyi indirmek için: python scripts/download_data.py --dataset phm2010"
            )

    # ------------------------------------------------------------------ tarama

    @cached_property
    def _index(self) -> dict[str, dict[int, Path]]:
        """Kesici -> {geçiş numarası: dosya yolu}."""
        found: dict[str, dict[int, Path]] = {}
        for path in self.root.rglob("*.csv"):
            match = CUT_FILE_PATTERN.match(path.name)
            if match is None:
                continue
            cutter = f"c{int(match.group(1))}"
            cut_index = int(match.group(2))
            found.setdefault(cutter, {})[cut_index] = path
        return found

    @cached_property
    def _wear_files(self) -> dict[str, Path]:
        """Kesici -> aşınma dosyası yolu."""
        found: dict[str, Path] = {}
        for path in self.root.rglob("*.csv"):
            match = WEAR_FILE_PATTERN.match(path.name)
            if match is not None:
                found[f"c{int(match.group(1))}"] = path
        return found

    def available_cutters(self) -> list[str]:
        """Sinyal dosyası bulunan kesiciler, isim sırasına göre."""
        return sorted(self._index, key=lambda name: int(name[1:]))

    def labelled_cutters(self) -> list[str]:
        """Hem sinyali hem aşınma etiketi olan kesiciler (beklenen: c1, c4, c6)."""
        return [c for c in self.available_cutters() if c in self._wear_files]

    def n_cuts(self, cutter: str) -> int:
        return len(self._require_cutter(cutter))

    # ------------------------------------------------------------------ etiket

    def wear(self, cutter: str) -> pd.DataFrame:
        """Kesicinin aşınma tablosu.

        Döndürülen sütunlar: ``cut``, her ağız için bir sütun, ve birleştirilmiş
        hedef ``vb_um``. Değerler mikrometre (kaynak dosyada 10^-3 mm).
        """
        if cutter not in self._wear_files:
            raise KeyError(
                f"'{cutter}' için aşınma etiketi yok. "
                f"Etiketli kesiciler: {self.labelled_cutters()}"
            )

        frame = pd.read_csv(self._wear_files[cutter])
        frame.columns = [str(c).strip().lower() for c in frame.columns]

        cut_column = next((c for c in frame.columns if c.startswith("cut")), None)
        if cut_column is None:
            raise ValueError(
                f"Aşınma dosyasında geçiş sütunu bulunamadı: {self._wear_files[cutter]}"
            )

        flute_columns = [c for c in frame.columns if c != cut_column]
        if not flute_columns:
            raise ValueError(
                f"Aşınma dosyasında ağız sütunu yok: {self._wear_files[cutter]}"
            )

        result = frame[[cut_column] + flute_columns].copy()
        result = result.rename(columns={cut_column: "cut"})
        result["cut"] = result["cut"].astype(int)

        flute_values = result[flute_columns].astype(float)
        if self.wear_aggregation == "max":
            result["vb_um"] = flute_values.max(axis=1)
        else:
            result["vb_um"] = flute_values.mean(axis=1)

        # Ağızlar arası saçılım, ölçüm belirsizliğinin doğrudan tahminidir.
        # Model sıralaması yaparken bu taban gürültünün altındaki farklar
        # anlamsızdır (bkz. yol haritası, bölüm 07 / Hata 3).
        result["flute_spread_um"] = flute_values.max(axis=1) - flute_values.min(axis=1)

        return result.sort_values("cut").reset_index(drop=True)

    # ------------------------------------------------------------------ sinyal

    def cut_refs(self, cutter: str) -> list[CutRef]:
        """Kesicinin tüm geçişleri, sıralı."""
        cuts = self._require_cutter(cutter)
        return [CutRef(cutter, index, cuts[index]) for index in sorted(cuts)]

    def load_cut(self, cutter: str, cut_index: int, max_rows: int | None = None) -> pd.DataFrame:
        """Tek bir geçişin ham sinyalini okur.

        ``max_rows`` keşifsel analizde tüm dosyayı okumamak için kullanılır.
        """
        cuts = self._require_cutter(cutter)
        if cut_index not in cuts:
            raise KeyError(
                f"'{cutter}' içinde {cut_index} numaralı geçiş yok "
                f"(mevcut aralık: {min(cuts)}-{max(cuts)})"
            )

        return pd.read_csv(
            cuts[cut_index],
            header=None,
            names=list(self.channels),
            dtype=np.float32,
            nrows=max_rows,
        )

    # ------------------------------------------------------------------ yardımcı

    def _require_cutter(self, cutter: str) -> dict[int, Path]:
        if cutter not in self._index:
            raise KeyError(
                f"'{cutter}' bulunamadı. Mevcut kesiciler: {self.available_cutters()}"
            )
        return self._index[cutter]

    def summary(self) -> pd.DataFrame:
        """Veri setinin durumunu özetler - indirme sonrası doğrulama için."""
        rows = []
        for cutter in self.available_cutters():
            has_labels = cutter in self._wear_files
            row = {
                "cutter": cutter,
                "n_cuts": self.n_cuts(cutter),
                "labelled": has_labels,
                "n_labels": len(self.wear(cutter)) if has_labels else 0,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - yalnızca hata ayıklama
        return f"PHM2010(root={self.root}, cutters={self.available_cutters()})"
