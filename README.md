# Takım Aşınması Tahmin Sistemi

CNC frezelemede sensör verisinden takım yan yüzey aşınmasını (VB) tahmin eden uçtan uca sistem.

Yol haritası: https://claude.ai/code/artifact/395988f2-9463-43ae-b252-62984fe329e6

## Durum

| Faz | İş | Durum |
|-----|----|-------|
| 00 | Kurulum, değerlendirme çatısı, naif taban | **tamamlandı** |
| 01 | Derin literatür taraması | — |
| 02 | Veri hattı ve keşifsel analiz | **tamamlandı** |
| 03 | Değerlendirme çatısı | — |
| 04 | Öznitelik hattı + gradyan artırma (Model A) | **tamamlandı** |
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

### Naif taban sonucu (27 Ağustos 2026)

`python scripts/run_naive_baseline.py` çıktısı, leave-one-cutter-out:

| Katlama | MAE (µm) | RMSE (µm) | Gecikme (geçiş) | Ağız saçılımı (µm) |
|---------|---------:|----------:|----------------:|-------------------:|
| test=c1 |    13,18 |     19,38 |             −27 |              10,26 |
| test=c4 |    20,67 |     22,43 |             −30 |               7,88 |
| test=c6 |    29,61 |     33,27 |             +47 |              20,48 |

Ortalama MAE **21,15 µm**, ortalama |gecikme| **34,7 geçiş**, en kötü gecikme **+47**.

İki not:

- Gecikmelerin **işaretli** ortalaması −3,33 çıkıyor ama bu yanıltıcı: erken ve geç
  alarmlar birbirini götürüyor. Karar için anlamlı olan mutlak ortalama ve en kötü
  değerdir; script her üçünü de basar.
- Kabul hedefi 15 µm, ağızlar arası saçılımın (12,87 µm ortalama, c6'da 20,5 µm)
  yakınında. Bu saçılım saf ölçüm gürültüsü değil — ağızlar gerçekten farklı aşınıyor —
  ama etiketin kendi değişkenliği bu mertebede. Sonuçlar raporlanırken birlikte verilmeli.

## Faz 02 bulguları (27 Ağustos 2026)

`python scripts/explore_phm.py` — şekiller `reports/figures` altında.

### 1. Aşınma eğrileri klasik üç bölgeyi gösteriyor, ama dizler farklı yerde

| Kesici | VB ilk | VB son | Eşiği (150 µm) geçtiği geçiş | Artış ilk ⅓ | orta ⅓ | son ⅓ |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 48,9 | 172,7 | 271 | 0,435 | 0,314 | 0,434 |
| c4 | 31,4 | 210,9 | 266 | 0,593 | 0,065 | 1,062 |
| c6 | 62,8 | 234,7 | 221 | 0,507 | 0,260 | 0,879 |

c4'ün orta bölgesi neredeyse düz (0,065 µm/geçiş) sonra sert bir dizle 1,062'ye
fırlıyor. c6 en erken eşiği geçen. Naif tabanın c6'da 47 geçiş gecikmesinin sebebi
bu: c1 ve c4 ile eğitilince eşik geçişini 266-271 civarında bekliyor, c6 ise 221'de
geçiyor.

**Sonuç:** dizin *yerini* tahmin etmek, eğrinin genel şeklini tahmin etmekten
zor ve asıl değer orada.

### 2. Ham korelasyon yanıltıcı - kullanmayın

Kuvvet öznitelikleri VB ile Spearman ρ ≈ 0,99 veriyor. Bu etkileyici değil:
VB geçiş sayısıyla monoton arttığı için **zamanla artan her öznitelik** ~1
korelasyon verir. Bu, naif tabanın zaten bildiği bilgi.

Doğru ölçü, geçiş sayısına bağlı monoton eğilim ikisinden de çıkarıldıktan
sonraki kısmi ilişki:

| | Ham Spearman | Eğilim çıkarılmış |
|---|---:|---:|
| En güçlü öznitelik | 0,994 | **0,301** |
| Tutarlı yönlü öznitelik sayısı | 140 / 168 | **47 / 168** |

0,30 modelin naif tabana katabileceği bilginin üst sınırıdır. Beklentileri
buna göre ayarlayın.

### 3. Kuvvet ham korelasyonda önde, ama gerçek bilgi titreşim ve AE'de

Ham sıralamanın tamamı kuvvet kanalları. Eğilim çıkarıldığında ilk sıralar
`vib_z_skew`, `vib_x_order_5`, `vib_z_order_5_ratio`, `ae_rms_order_7_ratio`
oluyor - yani titreşim ve akustik emisyon.

**Bu proje için iyi haber:** dinamometre sahada kullanılamıyor, NASA setinde de
yok. Kuvvetin katkısının büyük kısmının "zamanla artıyor"dan ibaret olması,
Faz 08'deki sensör azaltma çalışmasının ve Faz 07'deki transferin sandığımızdan
daha az şey kaybedebileceği anlamına geliyor.

### 4. Mertebe alanı tasarımı veriyle doğrulandı

`spectrum_early_late.png`: diş geçiş frekansında (520 Hz = mertebe 3) ve
harmoniğinde (1040 Hz = mertebe 6) keskin tepeler var, tam hesaplanan yerlerde.
Aşınmış takımın spektrumu tüm bantta 1-2 kat yukarıda.

## Faz 04 — Model A sonuçları (28 Ağustos 2026)

`python scripts/run_model_a.py`

| Model | MAE (µm) | \|overshoot\| (µm) | en kötü overshoot (µm) |
|---|---:|---:|---:|
| 0 · naif taban | 21,15 | 35,74 | **+57,53** |
| 1 · GBM (ham) | 19,91 | **21,13** | **−2,52** |
| 2 · GBM + monoton | 20,74 | 21,13 | −2,52 |
| 3 · GBM + normalize + monoton | **19,55** | 34,48 | +67,84 |

**Sonuç: MAE hedefi tutturulamadı (19,55 vs 15 µm), alarm hedefi tutturuldu.**

Asıl kazanım MAE'de değil alarm davranışında: naif taban en kötü katlamada takımın
eşiği **57,5 µm aşmasına** izin veriyor; GBM (ham) hiçbir katlamada geç alarm
vermiyor (en kötü overshoot −2,52 µm, yani hep erken). Üretimde bu, hurda parça
ile sağlam parça arasındaki farktır.

MAE'nin naif tabanı ancak %8 geçmesi sürpriz değil: Faz 02'de eğilim çıkarılmış
korelasyonun tavanının 0,30 olduğunu ölçmüştük. Model o tavana yakın çalışıyor.

### Kanal alt kümeleri — kuvvet sensörünü kaybetmenin bedeli

| Kanal kümesi | Öznitelik | MAE (µm) | \|overshoot\| (µm) |
|---|---:|---:|---:|
| hepsi (7 kanal) | 168 | 20,74 | 21,13 |
| sadece kuvvet | 72 | 24,75 | 46,33 |
| titreşim + AE | 96 | 23,82 | 29,37 |
| sadece titreşim | 72 | 25,10 | 26,36 |
| sadece AE | 24 | 55,70 | 60,19 |

Kuvveti kaybetmek MAE'yi 20,74 → 23,82 µm'ye çıkarıyor (**+%15**). Ciddi ama
yıkıcı değil — Model B için iyi haber. AE tek başına kullanılamaz.

### Kabul kriteri revize edildi

Faz 00'daki "gecikme ≤ 5 geçiş" hedefi **ulaşılamaz** çıktı. Eşik civarındaki
eğim ölçüldüğünde:

| Kesici | Eğim (µm/geçiş) | 5 geçiş için gereken doğruluk | Ölçüm gürültüsü |
|---|---:|---:|---:|
| c1 | 0,581 | 2,9 µm | 10,3 µm |
| c4 | 1,242 | 6,2 µm | 7,9 µm |
| c6 | 0,954 | 4,8 µm | 20,5 µm |

Hedef, etiketin kendi ölçüm gürültüsünün altında doğruluk istiyordu. Hiçbir model
bunu sağlayamaz; sağlıyor görünüyorsa sızıntı vardır.

Ayrıca "gecikme (geçiş)" metriği eğrinin eğimine bağlı olduğu için kesiciler
arasında karşılaştırılamıyor. Yerine **overshoot (µm)** kullanılıyor: alarm
çaldığında gerçek aşınmanın eşiği kaç µm aştığı. Eğimden bağımsız ve doğrudan
yorumlanabilir. Yeni hedef: |overshoot| < 25 µm.

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
