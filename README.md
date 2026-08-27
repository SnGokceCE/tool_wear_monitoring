# Takım Aşınması Tahmin Sistemi

CNC frezelemede sensör verisinden takım yan yüzey aşınmasını (VB) tahmin eden uçtan uca sistem.

Yol haritası: https://claude.ai/code/artifact/395988f2-9463-43ae-b252-62984fe329e6

## Durum

| Faz | İş | Durum |
|-----|----|-------|
| 00 | Kurulum, değerlendirme çatısı, naif taban | devam ediyor |
| 01 | Derin literatür taraması | — |
| 02 | Veri hattı ve keşifsel analiz | — |
| 03 | Değerlendirme çatısı | — |
| 04 | Öznitelik hattı + gradyan artırma | — |
| 05 | Derin öğrenme modelleri | — |
| 06 | Sistemi ayağa kaldır (v1) | — |
| 07 | Çapraz veri seti genelleme sınavı | — |
| 08 | Sensör azaltma çalışması | — |
| 09 | Karar mantığı ve belirsizlik | — |
| 10 | Sağlamlaştırma ve paketleme | — |
| 11 | Rapor ve sunum | — |

## Kabul kriterleri

Faz 00'da sabitlenen hedefler. Bunlar `config/default.yaml` içinde de tanımlıdır ve
değerlendirme kodu doğrudan oradan okur.

1. **Doğruluk** — Görülmemiş kesicide (leave-one-cutter-out) VB tahmininde
   **MAE < 15 µm**, üç katlamanın ortalaması.
2. **Zamanlama** — Aşınma sınırının (VB = 150 µm) aşıldığı geçişi
   **5 geçişten fazla gecikmeden** yakalama.
3. **Taban çizgisi** — Her model, sinyale hiç bakmayan naif tabanı (geçiş sayısından
   izotonik tahmin) her iki metrikte de geçmek zorundadır. Geçemeyen model sisteme girmez.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Derin öğrenme fazına (Faz 05) gelindiğinde ek olarak:

```bash
pip install -r requirements-dl.txt
```

## Veri

Veri depoya dahil değildir; `data/raw/` git tarafından yok sayılır.

```bash
python scripts/download_data.py --dataset nasa      # otomatik iner
python scripts/download_data.py --dataset phm2010   # elle yerleştirme talimatı verir
python scripts/download_data.py --verify            # yerleşimi doğrular
```

**NASA Milling** doğrudan indirilebilir. **PHM 2010** erişim için kayıt istediğinden
otomatik indirilemez; betik nereden alınacağını ve nereye konacağını söyler.

## Kullanım

```bash
python scripts/run_naive_baseline.py
```

## Yapı

```
config/           yapılandırma (yollar, sabitler, kabul kriterleri)
data/raw/         ham veri kümeleri (git dışı)
data/processed/   ara çıktılar (git dışı)
notebooks/        keşifsel analiz
reports/          şekiller ve sonuç tabloları
scripts/          çalıştırılabilir giriş noktaları
src/tcm/          kütüphane kodu
  datasets/       veri kümesi yükleyicileri
  features/       öznitelik çıkarımı
  models/         modeller
  evaluation/     bölme stratejileri ve metrikler
tests/            testler
```

## Bilinen sınırlar

- Hiçbir açık veri setinde **alüminyum** frezeleme aşınma verisi yok. PHM 2010 paslanmaz
  çelik, NASA dökme demir ve çelik. Sistem alüminyum için doğrulanmamıştır.
- Saha (üretim tezgâhı) verisi kullanılmamıştır.
- Siemens 840D entegrasyonu yazılmamıştır; yalnızca giriş arayüzü sözleşmesi tanımlıdır.
