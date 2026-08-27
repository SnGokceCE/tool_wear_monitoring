"""Komut satırı betikleri için ortak yardımcılar."""

from __future__ import annotations

import sys


def setup_console() -> None:
    """Konsol çıktısını UTF-8'e sabitler.

    Windows konsolu öntanımlı olarak cp1254 / cp857 kullanır ve Türkçe
    karakterler bozuk görünür. Kütüphane kodunda genel durum değiştirmek
    doğru olmadığı için bu yalnızca betiklerin giriş noktasından çağrılır.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Yeniden yönlendirilmiş ya da desteklemeyen akış - sessizce geç.
                pass
