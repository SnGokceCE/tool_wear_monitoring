"""İki modelin karşılaştırılmasında "kazandı" demenin ölçütü.

Sorun
-----
Küçük veride tek bir sayıya bakarak model seçmek yanıltıcıdır. Derin ağın
sonucu ağırlık başlangıcına ve yığın sırasına - yani rastgele tohuma - bağlıdır.
Aynı model aynı veriyle farklı tohumlarda farklı MAE verir.

Dolayısıyla "CNN 123, GBM 144, demek ki CNN kazandı" çıkarımı ancak aradaki
fark tohumlar arası saçılımdan BÜYÜKSE geçerlidir. Saçılım farktan büyükse
gözlenen üstünlük gürültüden ayırt edilemez ve dürüst cevap "kararsız"dır.

Bu, istatistikteki etki büyüklüğü / gürültü karşılaştırmasının en sade hali.
Kasıtlı olarak t-testi kullanılmıyor: 3 tohumla t-testinin varsayımları zaten
karşılanmaz, sonuç bilimsel görünen ama dayanaksız bir p-değeri olurdu.

Neden ayrı modül
----------------
Bu mantık ``run_model_deep.py`` içine gömülüydü ve betiğin KARAR bloğu onu
görmüyordu: özetteki kazanç sayımı yalnızca ortalama MAE'ye bakıyor, kararsız
sınavları da geçmiş sayıyordu. Burada durunca hem tek yerden okunur hem de
PyTorch kurulu olmadan test edilebilir.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

PASSED = "GEÇTİ"
UNDECIDED = "KARARSIZ"
FAILED = "geçemedi"


def seed_stability_verdict(
    mae_candidate: float,
    mae_reference: float,
    spread: float,
) -> str:
    """Aday modelin referansı geçip geçmediğine hüküm verir.

    ``spread`` adayın tohumlar arası MAE standart sapmasıdır.

      ``KARARSIZ`` : saçılım >= iki model arasındaki fark. Gözlenen üstünlük
                     (ya da düşüklük) rastgelelikten ayırt edilemiyor.
      ``GEÇTİ``    : fark saçılımdan büyük VE aday daha düşük MAE veriyor.
      ``geçemedi`` : fark saçılımdan büyük ama aday daha yüksek MAE veriyor.

    Tek tohumla çalışıldıysa ``spread`` 0'dır ve hüküm doğrudan MAE
    karşılaştırmasına düşer - ama o sonucun saçılımı ÖLÇÜLMEMİŞTİR, sadece
    varsayılmamıştır. Raporda bu ayrım belirtilmelidir.
    """
    gap = abs(float(mae_candidate) - float(mae_reference))
    spread = float(spread)

    if not np.isfinite(gap) or not np.isfinite(spread):
        return UNDECIDED
    if spread >= gap:
        return UNDECIDED
    return PASSED if mae_candidate < mae_reference else FAILED


def count_decisive_wins(verdicts: Iterable[str]) -> Mapping[str, int]:
    """Hükümleri sayar. YALNIZCA ``GEÇTİ`` kazanç sayılır.

    Buradaki asıl mesele ``KARARSIZ``'ın nereye yazıldığı. Kazanca eklenirse
    rapor modeli olduğundan iyi gösterir; kayba eklenirse olduğundan kötü.
    İkisi de yanlış - kararsız kendi başına bir sonuçtur ve öyle raporlanır.
    """
    counts = {PASSED: 0, UNDECIDED: 0, FAILED: 0}
    total = 0
    for verdict in verdicts:
        total += 1
        if verdict in counts:
            counts[verdict] += 1
        else:
            raise ValueError(f"Bilinmeyen hüküm: {verdict!r}")

    return {
        "passed": counts[PASSED],
        "undecided": counts[UNDECIDED],
        "failed": counts[FAILED],
        "total": total,
    }


def describe_wins(counts: Mapping[str, int], candidate: str, reference: str) -> str:
    """Sayımı insan okunur tek cümleye çevirir - rapora doğrudan girer."""
    line = (
        f"{candidate}, {counts['passed']}/{counts['total']} sınavda "
        f"{reference} modelini KESİN olarak geçti."
    )
    if counts["undecided"]:
        line += (
            f" {counts['undecided']} sınav KARARSIZ: tohumlar arası saçılım "
            "iki model arasındaki farktan büyük, üstünlük gürültüden "
            "ayırt edilemiyor."
        )
    if counts["failed"]:
        line += f" {counts['failed']} sınavda geçemedi."
    return line
