"""Bir sonucun hangi çalıştırmadan geldiğini sabitleyen künye bilgisi.

Neden gerekli
-------------
Rapordaki "MAE 138,61 µm" sayısı tek başına doğrulanabilir değil. Hangi kodla,
hangi yapılandırmayla, hangi kütüphane sürümüyle üretildiği bilinmiyorsa üç ay
sonra aynı sayıyı yeniden üretmek mümkün olmaz - ve jüri "bunu nasıl aldınız"
diye sorduğunda cevap "çalıştırmıştık" olur.

Bu modül her çıktının yanına şunları yazar:

  git_hash        hangi kod   (kirliyse ``git_dirty`` işaretlenir)
  config_digest   hangi ayar  (dosya içeriğinin sha256'sı)
  versions        hangi kütüphane sürümleri - LightGBM sürümü değişirse
                  ağaçlar da değişir, sayı birebir tutmayabilir
  timestamp       ne zaman

``git_dirty`` özellikle önemli: commit edilmemiş değişikliklerle üretilen bir
sonuç, o git hash'inden yeniden üretilemez. Sessizce geçmek yerine künyeye
yazılır.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tcm.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT

UNKNOWN = "bilinmiyor"

# Sayısal sonucu etkileyebilecek kütüphaneler. Sürüm değişince sonuç birebir
# tutmayabilir; künyede durmaları bunu açıklar.
TRACKED_PACKAGES = ("lightgbm", "scikit-learn", "numpy", "pandas", "scipy", "torch")


def _git(args: list[str], root: Path) -> str | None:
    """Git komutunu çalıştırır; git yoksa ya da depo değilse ``None`` döner."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_hash(root: Path | str = PROJECT_ROOT) -> tuple[str, bool]:
    """``(commit_hash, kirli_mi)``.

    Depo değilse ya da git kurulu değilse ``("bilinmiyor", False)`` döner -
    künye üretimi bu yüzden hiçbir zaman patlamamalı.
    """
    root = Path(root)
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return UNKNOWN, False

    status = _git(["status", "--porcelain"], root)
    return commit, bool(status)


def file_digest(path: Path | str) -> str:
    """Dosya içeriğinin sha256'sı; dosya yoksa ``"bilinmiyor"``."""
    path = Path(path)
    if not path.is_file():
        return UNKNOWN

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest(path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    """Yapılandırma dosyasının içerik özeti - hangi ayarla çalışıldığını sabitler."""
    return file_digest(path)


def relative_path(path: Path | str, root: Path | str = PROJECT_ROOT) -> str:
    """Künyeye yazılacak yol: proje içindeyse göreli, değilse mutlak.

    Mutlak yollar künyeyi üretildiği makineye bağlar - başka bir bilgisayarda
    ``D:\\Tomtaş...`` diye bir dizin yoktur ve künye okunamaz hale gelir.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def package_versions() -> dict[str, str]:
    """İzlenen kütüphanelerin sürümleri; kurulu olmayanlar atlanır."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def run_stamp(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    root: Path | str = PROJECT_ROOT,
    command: str | None = None,
) -> dict[str, object]:
    """Bir çalıştırmayı tanımlayan künye sözlüğü.

    ``command`` verilmezse ``sys.argv``'den kurulur; rapordaki bir sayının
    hangi komutla üretildiği böyle sabitlenir.
    """
    commit, dirty = git_hash(root)
    return {
        "git_hash": commit,
        "git_dirty": dirty,
        "config_path": relative_path(config_path, root),
        "config_digest": config_digest(config_path),
        "command": command if command is not None else " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "versions": package_versions(),
    }


def format_stamp(stamp: dict[str, object]) -> str:
    """Künyeyi konsola basılacak birkaç satıra çevirir."""
    dirty = "  (KİRLİ - commit edilmemiş değişiklik var)" if stamp["git_dirty"] else ""
    versions = ", ".join(f"{k} {v}" for k, v in dict(stamp["versions"]).items())
    return (
        f"git       : {stamp['git_hash']}{dirty}\n"
        f"config    : {str(stamp['config_digest'])[:16]}…\n"
        f"zaman     : {stamp['timestamp']}\n"
        f"sürümler  : {versions}"
    )
