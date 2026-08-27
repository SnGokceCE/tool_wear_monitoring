"""Veri kümesi indirme ve yerleşim doğrulama.

NASA Milling doğrudan indirilebilir. PHM 2010 erişim için kayıt istediğinden
otomatik indirilemez; burada yalnızca nereden alınacağı ve nereye konacağı
anlatılır.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import requests

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
CHUNK_BYTES = 1 << 20  # 1 MiB
MAX_ATTEMPTS = 5

# S3 uçları User-Agent'sız akış isteklerini sıfırlayabiliyor; curl çalışıp
# requests'in düşmesinin sebebi buydu.
HEADERS = {"User-Agent": "tcm-dataset-downloader/0.1 (+https://phmsociety.org)"}

PHM2010_INSTRUCTIONS = """
PHM 2010 otomatik indirilemiyor - kaynak erişim için kayıt istiyor.

Kaynaklar:
  1. PHM Society  https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/
  2. IEEE DataPort https://ieee-dataport.org/documents/phm2010-dataset
  3. Kaggle aynası  "PHM 2010 milling" araması

İndirdikten sonra arşivi şuraya açın:

  {target}

Beklenen içerik (klasör yerleşimi değişebilir, yükleyici iç içe klasörleri
kendisi tarar - önemli olan dosya isimleri):

  c_1_001.csv ... c_1_315.csv      geçiş sinyalleri, başlıksız, 7 sütun
  c1_wear.csv                      aşınma etiketleri
  ... aynısı c4 ve c6 için

Yerleşimi doğrulamak için:

  python scripts/download_data.py --verify
""".strip()


def download_nasa(target_dir: str | Path, force: bool = False) -> Path:
    """NASA Milling arşivini indirip açar. Açılan klasörü döndürür."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = list(target_dir.rglob("mill.mat"))
    if existing and not force:
        print(f"[nasa] Zaten mevcut: {existing[0]}")
        return target_dir

    archive_path = target_dir / "milling.zip"
    print(f"[nasa] İndiriliyor: {NASA_URL}")
    _download(NASA_URL, archive_path)

    print(f"[nasa] Açılıyor: {archive_path.name}")
    _extract_recursive(archive_path, target_dir)
    archive_path.unlink(missing_ok=True)

    found = list(target_dir.rglob("mill.mat"))
    if not found:
        raise RuntimeError(
            f"Arşiv açıldı ama mill.mat bulunamadı: {target_dir}\n"
            "Arşivin içeriğini elle kontrol edin."
        )
    print(f"[nasa] Hazır: {found[0]}")
    return target_dir


def phm2010_instructions(target_dir: str | Path) -> str:
    """PHM 2010 için elle yerleştirme talimatı."""
    return PHM2010_INSTRUCTIONS.format(target=Path(target_dir).resolve())


def verify(phm_root: str | Path, nasa_root: str | Path) -> bool:
    """Her iki veri kümesinin yerleşimini doğrular. Hepsi hazırsa True."""
    ok = True

    print("PHM 2010")
    phm_root = Path(phm_root)
    if not phm_root.exists():
        print(f"  [eksik] klasör yok: {phm_root}")
        ok = False
    else:
        try:
            from tcm.datasets.phm2010 import PHM2010

            dataset = PHM2010(phm_root)
            summary = dataset.summary()
            if summary.empty:
                print(f"  [eksik] {phm_root} altında geçiş dosyası bulunamadı")
                ok = False
            else:
                print(summary.to_string(index=False))
                labelled = dataset.labelled_cutters()
                if not labelled:
                    print("  [uyarı] etiketli kesici yok - aşınma dosyaları eksik")
                    ok = False
        except Exception as error:  # noqa: BLE001 - kullanıcıya ham hata gösterilir
            print(f"  [hata] {error}")
            ok = False

    print("\nNASA Milling")
    nasa_root = Path(nasa_root)
    if not nasa_root.exists():
        print(f"  [eksik] klasör yok: {nasa_root}")
        ok = False
    else:
        try:
            from tcm.datasets.nasa import NASAMilling

            dataset = NASAMilling(nasa_root)
            meta = dataset.metadata()
            print(f"  koşu sayısı : {len(meta)}")
            print(f"  etiketli    : {int(meta['has_label'].sum())}")
            print(f"  vaka sayısı : {meta['case'].nunique()}")
        except Exception as error:  # noqa: BLE001
            print(f"  [hata] {error}")
            ok = False

    return ok


# ---------------------------------------------------------------- yardımcılar


def _download(url: str, destination: Path) -> None:
    """Dosyayı indirir; bağlantı düşerse kaldığı yerden devam eder.

    S3 uzun akışlarda bağlantıyı sıfırlayabildiği için Range başlığıyla
    devam etme ve üstel bekleme ile yeniden deneme uygulanıyor.
    """
    total = _content_length(url)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        written = destination.stat().st_size if destination.exists() else 0

        if total and written >= total:
            print()
            return

        headers = dict(HEADERS)
        mode = "wb"
        if written:
            headers["Range"] = f"bytes={written}-"
            mode = "ab"
            print(f"\n  {written >> 20} MiB inmiş, devam ediliyor (deneme {attempt})")

        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                # Sunucu Range'i yok sayarsa baştan başlıyoruz demektir.
                if written and response.status_code == 200:
                    written, mode = 0, "wb"
                response.raise_for_status()

                with destination.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)
                        if total:
                            percent = 100 * written / total
                            print(
                                f"\r  {written >> 20} / {total >> 20} MiB ({percent:.0f}%)",
                                end="",
                            )
                        else:
                            print(f"\r  {written >> 20} MiB", end="")
            print()
            return

        except (requests.RequestException, OSError) as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{MAX_ATTEMPTS} denemede indirilemedi: {url}\n"
                    f"Son hata: {error}\n"
                    "Dosyayı tarayıcıdan indirip data/raw/ altına elle açabilirsiniz."
                ) from error
            wait = 2**attempt
            print(f"\n  [uyarı] bağlantı düştü ({error.__class__.__name__}), "
                  f"{wait} sn sonra yeniden denenecek")
            time.sleep(wait)


def _content_length(url: str) -> int:
    """Dosya boyutu; sunucu vermezse 0."""
    try:
        response = requests.head(url, timeout=30, headers=HEADERS, allow_redirects=True)
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    except (requests.RequestException, ValueError):
        return 0


def _extract_recursive(archive: Path, target_dir: Path, depth: int = 0) -> None:
    """Zip açar; içinden yeni zip çıkarsa onu da açar (NASA arşivi iç içe)."""
    if depth > 3:
        return
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)

    for nested in list(target_dir.rglob("*.zip")):
        if nested.resolve() == archive.resolve():
            continue
        _extract_recursive(nested, nested.parent, depth + 1)
        nested.unlink(missing_ok=True)
