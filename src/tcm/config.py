"""Yapılandırma yükleme.

Sabitler koda gömülmez; hepsi ``config/default.yaml`` içinde durur. Böylece
raporda tek yerden okunabilir ve veri setinden doğrulandıkça güncellenebilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class Config:
    """Sözlük tabanlı yapılandırma; nokta yoluyla erişim ve yol çözümleme sağlar."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._data = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get("phm2010.sampling_rate_hz")`` biçiminde erişim."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Yapılandırmadaki göreli bir yolu proje köküne göre mutlak yola çevirir."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Yapılandırmada yol bulunamadı: {dotted}")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:  # pragma: no cover - yalnızca hata ayıklama
        return f"Config(root={self.root}, keys={sorted(self._data)})"


def load_config(path: str | Path | None = None) -> Config:
    """Yapılandırmayı yükler. ``path`` verilmezse ``config/default.yaml`` kullanılır."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Yapılandırma dosyası yok: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)
