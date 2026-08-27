"""Bölme stratejileri.

Bu projede tek meşru bölme, **kesici bazında** olandır: bir kesicinin tüm
geçişleri ya eğitimde ya testtedir.

Kayan pencereleri rastgele bölmek bu literatürdeki şişirilmiş sonuçların
başlıca kaynağıdır: komşu pencereler neredeyse aynı olduğu için model
ezberler ve hata gerçekdışı derecede düşük çıkar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Split:
    """Tek bir katlama: hangi kesiciler eğitimde, hangisi testte."""

    train: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"test={'+'.join(self.test)}"

    def __repr__(self) -> str:  # pragma: no cover - yalnızca hata ayıklama
        return f"Split(train={list(self.train)}, test={list(self.test)})"


def leave_one_cutter_out(cutters: Sequence[str]) -> list[Split]:
    """Her kesici sırayla test kümesi olur.

    PHM 2010'un üç etiketli kesicisi için üç katlama üretir::

        c1 + c4 -> c6
        c1 + c6 -> c4
        c4 + c6 -> c1

    Sonuç, üç katlamanın ortalamasıdır. Tek katlama raporlamak yanıltıcıdır:
    üç yörünge birbirinden belirgin şekilde farklı davranabilir.
    """
    cutters = tuple(cutters)
    if len(cutters) < 2:
        raise ValueError(
            f"Kesici bazında bölme için en az 2 kesici gerekir, {len(cutters)} verildi"
        )
    if len(set(cutters)) != len(cutters):
        raise ValueError(f"Kesici listesinde tekrar var: {cutters}")

    return [
        Split(train=tuple(c for c in cutters if c != held_out), test=(held_out,))
        for held_out in cutters
    ]


def describe(splits: Iterable[Split]) -> str:
    """Bölmeleri insan okunur biçimde döndürür (log ve rapor için)."""
    lines = []
    for index, split in enumerate(splits, start=1):
        lines.append(
            f"  {index}. eğitim: {', '.join(split.train)}"
            f"  ->  test: {', '.join(split.test)}"
        )
    return "\n".join(lines)
