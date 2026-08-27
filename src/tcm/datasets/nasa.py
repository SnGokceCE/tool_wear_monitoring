"""NASA Milling veri seti yükleyicisi (ikincil veri seti).

DİKKAT: Bu veri seti eğitimde kullanılmaz. Faz 07'deki çapraz veri seti
genelleme sınavı için ayrılmıştır - eğitime katıldığı anda sınav geçersiz
olur. ``config/default.yaml`` içindeki ``nasa.use_for_training: false``
bunu belgeler; ``ensure_not_used_for_training()`` de kod tarafında hatırlatır.

Kaynak dosya ``mill.mat``, MATLAB struct dizisidir. Alanlar:
``case, run, VB, time, DOC, feed, material`` (koşul ve etiket) ve
``smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle`` (sinyal).

Aşınma her koşudan sonra ölçülmemiştir; etiketsiz koşular ``VB = NaN``
olarak gelir. Bu, veri setinin bilinen bir kısıtıdır.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_FIELDS = ("case", "run", "VB", "time", "DOC", "feed", "material")
SIGNAL_FIELDS = (
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
)


class NASAMilling:
    """``mill.mat`` üzerine tembel erişim."""

    def __init__(self, root: str | Path, mat_file: str = "mill.mat") -> None:
        self.root = Path(root)
        self.mat_path = self._locate(mat_file)

    def _locate(self, mat_file: str) -> Path:
        direct = self.root / mat_file
        if direct.exists():
            return direct
        matches = sorted(self.root.rglob(mat_file)) if self.root.exists() else []
        if not matches:
            raise FileNotFoundError(
                f"'{mat_file}' bulunamadı: {self.root}\n"
                "Veriyi indirmek için: python scripts/download_data.py --dataset nasa"
            )
        return matches[0]

    @cached_property
    def _entries(self) -> list:
        from scipy.io import loadmat

        mat = loadmat(self.mat_path, struct_as_record=False, squeeze_me=True)
        if "mill" not in mat:
            raise ValueError(
                f"{self.mat_path} içinde 'mill' değişkeni yok. "
                f"Bulunanlar: {[k for k in mat if not k.startswith('__')]}"
            )
        entries = mat["mill"]
        return list(np.atleast_1d(entries))

    def metadata(self) -> pd.DataFrame:
        """Koşu başına koşul ve etiket tablosu.

        ``VB`` ölçülmemiş koşularda ``NaN``'dır. Kullanılabilir etiket sayısı
        toplam koşu sayısından belirgin şekilde azdır - bu yüzden veri seti
        eğitim için değil, sınav için uygundur.
        """
        rows = []
        for position, entry in enumerate(self._entries):
            row = {"entry": position}
            for field in METADATA_FIELDS:
                row[field] = _scalar(getattr(entry, field, None))
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame["has_label"] = frame["VB"].notna()
        return frame

    def signals(self, entry_index: int) -> pd.DataFrame:
        """Tek bir koşunun sinyal kanalları."""
        entry = self._entries[entry_index]
        columns = {}
        for field in SIGNAL_FIELDS:
            values = getattr(entry, field, None)
            if values is None:
                continue
            columns[field] = np.asarray(values, dtype=np.float32).ravel()

        if not columns:
            raise ValueError(f"{entry_index}. koşuda sinyal kanalı bulunamadı")

        length = min(len(v) for v in columns.values())
        return pd.DataFrame({name: values[:length] for name, values in columns.items()})

    def summary(self) -> pd.DataFrame:
        """Vaka başına koşu ve etiket sayısı - indirme sonrası doğrulama için."""
        meta = self.metadata()
        return (
            meta.groupby("case")
            .agg(
                n_runs=("run", "count"),
                n_labels=("has_label", "sum"),
                material=("material", "first"),
                doc=("DOC", "first"),
                feed=("feed", "first"),
            )
            .reset_index()
        )

    def __repr__(self) -> str:  # pragma: no cover - yalnızca hata ayıklama
        return f"NASAMilling(mat_path={self.mat_path})"


def ensure_not_used_for_training(config) -> None:
    """Yapılandırma NASA'yı eğitime açmışsa hata verir.

    Bu kasıtlı bir engel: Faz 07'deki sınavın anlamlı olması, NASA'nın
    eğitim boyunca hiç görülmemesine bağlıdır.
    """
    if config.get("nasa.use_for_training", False):
        raise RuntimeError(
            "NASA Milling eğitimde kullanılamaz - Faz 07 genelleme sınavı için "
            "ayrılmıştır. Bilerek değiştiriyorsanız config/default.yaml içindeki "
            "nasa.use_for_training alanını ve bu kontrolü birlikte güncelleyin."
        )


def _scalar(value):
    """MATLAB'dan gelen 0-boyutlu / boş dizileri düz Python değerine çevirir."""
    if value is None:
        return np.nan
    array = np.atleast_1d(np.asarray(value).ravel())
    if array.size == 0:
        return np.nan
    item = array[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8", errors="replace")
    if isinstance(item, str):
        return item
    try:
        return float(item)
    except (TypeError, ValueError):
        return item
