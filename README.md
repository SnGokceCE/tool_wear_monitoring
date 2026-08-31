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
| 04b | Model B-1: NASA + kesme parametreleri | **tamamlandı** |
| 04c | Model B-2: NASA + ağırlıklı PHM | **tamamlandı** |
| 04d | worn/unworn sınıflandırma çıktısı | **tamamlandı** |
| 05 | Derin öğrenme modelleri | **tamamlandı** |
| 06 | Sistemi ayağa kaldır (v1) | **tamamlandı** |
| 07 | Çapraz veri seti genelleme sınavı | — |
| 08 | Sensör azaltma çalışması | — |
| 09 | Karar mantığı — alarm eşiği | **tamamlandı** |
| 10 | Sağlamlaştırma ve paketleme | — |
| 11 | Rapor ve sunum | — |

## Sonuçların kaynağı (provenance)

Rapordaki her sayı bir çalıştırmadan gelir. Hangi kodla, hangi ayarla ve hangi
kütüphane sürümleriyle üretildiği bilinmiyorsa o sayı doğrulanamaz — ve üç ay
sonra yeniden üretilemez.

Bu yüzden `train_model.py`, `run_model_deep.py` ve `threshold_sweep.py` her
çıktının yanına künye yazar (`tcm.provenance`): git commit'i, çalışma ağacının
kirli olup olmadığı, config dosyasının sha256'sı, komut satırı, zaman damgası
ve LightGBM / scikit-learn / NumPy / pandas / SciPy / PyTorch sürümleri.

| README bölümü | Kaynak dosya | Üretme komutu |
|---|---|---|
| Naif taban | `reports/naive_baseline.csv` | `python scripts/run_naive_baseline.py` |
| Faz 02 keşif | `reports/figures/` | `python scripts/explore_phm.py` |
| Faz 04 · Model A | `reports/model_a_*.csv` | `python scripts/run_model_a.py --save` |
| Faz 04b · Model B-1 | `reports/model_b1_*.csv` | `python scripts/run_model_b1.py --save` |
| Faz 04c · Model B-2 | `reports/model_b2_summary.csv` | `python scripts/run_model_b2.py --save` |
| Faz 04d · sınıflandırma | `reports/classification_summary.csv` | `python scripts/run_classification.py --save` |
| Faz 05 · derin öğrenme | `reports/model_deep_summary.csv` + `.provenance.json` | `python scripts/run_model_deep.py --seeds 3 --save` |
| Faz 06 · teslim modeli | `reports/model_b1_package.json`, `reports/model_b1_baselines.csv` | `python scripts/train_model.py` |
| Faz 06 · eşik taraması | `reports/threshold_sweep.csv` | `python scripts/threshold_sweep.py --save` |
| Faz 09 · karar kuralı | `reports/decision_rule_summary.csv` | `python scripts/run_decision_rule.py --save` |

Üç uyarı:

- **Faz 04 ve 04b–04d sonuçları künye eklenmeden önce (28 Ağustos 2026)
  üretildi.** Sayıları geçerli — öznitelik tablosu ve model kodu o günden beri
  değişmedi, Faz 06'daki refaktör birebir aynı tabloyu ürettiği doğrulanarak
  yapıldı — ama künyeleri kayıtlı değil. Yeniden çalıştırılırlarsa künye
  kazanırlar.
- **Faz 09 sayıları 31 Ağustos 2026'da değişti** (kilitlenme hatası
  düzeltildi). Eski ve yeni değerlerin karşılaştırması "Yol boyunca yanlış
  çıkan şeyler" bölümünde.
- **Faz 05 ve Faz 06 künyeleri "kirli" (dirty) işaretli.** Sayılar, henüz
  commit edilmemiş bir çalışma ağacıyla üretildi. Faz 06 commit edildikten
  sonra bu betikler yeniden çalıştırılırsa künyeler temiz commit'e bağlanır.

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

## Faz 04b — Model B-1 sonuçları (28 Ağustos 2026)

`python scripts/run_model_b1.py` — NASA, 145 koşu, 15 vaka, 8 koşul.

MAE (µm), üç ayrı sınav:

| Girdi kümesi | Öznitelik | vaka-dışı | koşul-dışı | **malzeme-dışı** |
|---|---:|---:|---:|---:|
| naif taban (koşu no) | 1 | 145,99 | 164,59 | 308,96 |
| sadece sensör | 144 | 140,19 | 166,57 | **216,85** |
| **parametre + süre** | 5 | **108,38** | **114,19** | 388,16 |
| sensör + parametre | 148 | 139,42 | 163,99 | 226,66 |
| sensör + parametre + süre | 149 | 138,47 | 164,27 | 271,49 |

> **UYARI — sayılar açıklanmamış bir kayma içeriyor (31 Ağustos 2026).**
> Yukarıdaki değerler `reports/model_b1_summary.csv` dosyasından, 31 Ağustos
> 2026'da `b5ff7e8` üzerine commit edilmemiş Faz 06 değişiklikleriyle yapılan
> koşudan geliyor. Betik iki kez çalıştırıldı ve CSV'yi birebir yeniden üretti,
> yani koşu deterministik.
>
> Ancak README'de daha önce **başka** sayılar yazıyordu (`145,38 / 173,26 /
> 221,34` gibi) ve o sayıların hangi koşudan geldiği kayıtlı değil. İki koşu
> arasındaki fark **0,14 – 6,69 µm** aralığında; en büyüğü `sadece sensör`
> satırının koşul-dışı hücresinde (173,26 → 166,57). Fark gürültü olamaz —
> tohum sabit ve koşu deterministik — dolayısıyla arada bir kod ya da veri
> değişikliği olmuş olmalı. **Sebep henüz teşhis edilmedi**; bkz. `NEXT.md`.
>
> Aşağıdaki üç bulgunun yönü bu kaymadan etkilenmiyor, ama metindeki tek tek
> sayılar eski koşuya ait olabilir.

### Üç bulgu

**1. Kesme parametreleri TEK BAŞINA işe yaramıyor (MAE 205 µm), ama süreyle
birlikte her şeyi geçiyor (108 µm).**

Fizik açık: kesme parametreleri aşınma **hızını** belirler, **miktarını** değil.
"Çelikte, 0,5 mm/dev ilerlemeyle" bilgisi takımın şu an ne kadar aşındığını
söylemez — ne kadar süredir kestiğini de bilmek gerekir. Hız × süre = aşınma.

Bu, istenen girdi tanımının (malzeme + ilerleme + kesme parametreleri)
**eksik** olduğu anlamına gelir: kümülatif kesme süresi olmadan çalışmaz.
Sahada bilinen bir değerdir (yeni takımda sayaç sıfırlanır).

**2. NASA'da 5 girdilik parametre modeli, 144 öznitelikli sensör modelini
eziyor** — ve sensör eklemek onu bozuyor (108,38 → 138,61).

Sebebi NASA'nın 250 Hz örneklemesi. PHM'de (50 kHz) sensörler naif tabanı
geçmişti; burada geçemiyor. Sensörün değeri, sensörün kalitesine bağlı.

**3. Ama malzeme değişince tablo tersine dönüyor.** Görülmemiş malzemede
parametre modeli çöküyor (388,16 — naif tabandan bile kötü), sensör modeli
en iyisi oluyor (221,34).

Sebep: parametre modeli dökme demirin aşınma hızını öğrenip çeliğe uyguluyor
ve yanılıyor. Sensör modeli hızı çıkarsamak yerine **durumu ölçtüğü** için
daha zarif bozuluyor.

**Bu doğrudan Tomtaş'ın alüminyum sorusuna denk geliyor:** hiç görülmemiş bir
malzemede parametre tabanlı model güvenilmez; orada yalnızca sensör işe yarar.

### Yan gözlemler

- `cum_time` öznitelik öneminde 1. sırada; `feed` ve `rpm` sıfır. (`rpm` NASA'da
  sabit olduğu için sıfır çıkması beklenen davranış.)
- Aşırı aşınmış koşular MAE'yi ikiye katlıyor: VB ≤ 600 µm ile sınırlandığında
  (126/145 koşu) sensör modelinin MAE'si 140,45 → **78,46**'ya düşüyor.
- Sensörden malzeme tahmini %79,5 doğruluk (taban %67,6) — sinyal malzemeyi
  kısmen ele veriyor ama güçlü değil.

## Faz 04c — Model B-2 sonuçları (28 Ağustos 2026)

`python scripts/run_model_b2.py`

Soru: **PHM'i eğitime eklemek NASA'daki performansı iyileştiriyor mu?**

Kurgu: sınav NASA'nın kendi grupları üzerinde (B-1 ile aynı), PHM satırları her
katlamada eğitime ekleniyor ama hiçbir zaman test olmuyor. Ortak öznitelik
uzayında 96 sensör özniteliği (kuvvet ve motor akımı düştü), takım bazında
normalize edilmiş.

### Cevap: hayır

MAE değişimi, taban çizgisine (PHM yok) göre:

| Ağırlık | vaka-dışı | koşul-dışı | malzeme-dışı | Sonuç |
|---|---:|---:|---:|---|
| PHM w=0,05 | +3,1% | +12,0% | −14,3% | 1/3 |
| PHM w=0,15 (eşit toplam) | +1,5% | +8,9% | −3,1% | 1/3 |
| PHM w=1,0 (ağırlıksız) | −5,4% | −0,3% | +6,4% | 2/3 |

**Hiçbir ağırlık üç sınavda birden taban çizgisini geçmiyor.** Tek tek
bakıldığında bazı hücreler iyi görünüyor, ama ağırlığı sonuca bakarak seçmek
gerekiyor — yani gerçek kazanç değil, seçim yanlılığı.

### Daha kötüsü: alarm davranışı bozuluyor

En kötü overshoot (µm), malzeme-dışı sınavda:

| | vaka-dışı | koşul-dışı | **malzeme-dışı** |
|---|---:|---:|---:|
| PHM yok | 320 | 320 | **−60** |
| PHM w=0,05 | 320 | 320 | **+1230** |
| PHM w=0,15 | 320 | 320 | **+1230** |
| PHM w=1,0 | 320 | 320 | **+1230** |

PHM'siz model görülmemiş malzemede hep erken alarm veriyor (−60 µm, güvenli).
PHM eklenince **her ağırlıkta** 1230 µm geç alarma dönüyor — takımın eşiği
dört katına çıkmasına izin veriyor. MAE'deki marjinal oynamalar bunun yanında
önemsiz.

### Yorum

PHM'in 945 satırı tek kesme koşulundan geliyor ve parametre etkisini öğretmeye
katkısı sıfır. Sensör→aşınma ilişkisinin şeklini öğretebilirdi, ama iki tezgâh
arasında o ilişki yeterince farklı: PHM'in paslanmaz çelikteki davranışını
öğrenen model, NASA'nın dökme demir/çelik verisinde yanlış genelliyor.

**Karar: Model B-1 (sadece NASA) teslim edilecek yapıdır.** Birleştirme
denendi, ölçüldü, işe yaramadı ve raporda böyle yazılacak.

## Faz 04d — worn/unworn sınıflandırma (28 Ağustos 2026)

`python scripts/run_classification.py`

Mentor güncellemesi: **girdiler serbest, çıktılar arasında worn/unworn olmak
zorunda.** İki üretim yolu karşılaştırıldı:

- **A · regresyon + eşik** — VB tahmin edilir, eşiği geçtiyse "worn"
- **B · doğrudan sınıflandırıcı** — modele VB hiç öğretilmez, doğrudan sorulur

NASA, eşik 300 µm, 70 worn / 75 unworn.

### Neden doğruluk (accuracy) tek başına yetmiyor

"Hep çoğunluk sınıfını söyle" tabanı **%51,7 doğruluk** alıyor ama
`worn_recall = 0` — 70 aşınmış takımın hepsini kaçırıyor. Asıl bakılacak
metrikler `worn_recall` (aşınmışların kaçta kaçını yakaladık) ve
`missed_worn` (kaç aşınmış takım kaçtı).

Hata türleri simetrik değil: kaçırılan aşınma parçayı hurdaya çıkarır,
yanlış alarm sadece takım ömrü israf eder.

### Dengeli doğruluk (balanced accuracy)

| Yöntem | Girdi | vaka-dışı | koşul-dışı | malzeme-dışı |
|---|---|---:|---:|---:|
| naif (koşu no + eşik) | 1 | 0,776 | 0,749 | **0,693** |
| A · regresyon + eşik | sensör | 0,803 | 0,781 | 0,574 |
| A · regresyon + eşik | **parametre + süre** | **0,885** | **0,844** | 0,576 |
| A · regresyon + eşik | hepsi | 0,830 | 0,755 | 0,583 |
| B · sınıflandırıcı | sensör | 0,814 | 0,786 | 0,664 |
| B · sınıflandırıcı | parametre + süre | 0,835 | 0,780 | 0,561 |
| B · sınıflandırıcı | hepsi | 0,814 | 0,773 | 0,692 |

### Kaçırılan aşınmış takım sayısı — asıl maliyet

| Yöntem | Girdi | vaka-dışı | koşul-dışı | malzeme-dışı |
|---|---|---:|---:|---:|
| naif | 1 | 9 | 9 | 15 |
| A · regresyon + eşik | **parametre + süre** | **4** | **5** | 23 |
| A · regresyon + eşik | sensör | 8 | 12 | 26 |
| B · sınıflandırıcı | sensör | 13 | 16 | 34 |

### Üç bulgu

**1. Regresyon + eşik, doğrudan sınıflandırıcıdan iyi.** Beklenen sonuç değildi.
Sebebi muhtemelen şu: regresyon VB'nin tamamını öğrenerek daha çok bilgi
kullanıyor; sınıflandırıcı ise "295 µm" ile "5 µm" arasındaki farkı görmüyor,
ikisi de sadece "unworn".

**2. Görülmemiş malzemede hiçbir model naif tabanı geçemiyor.** Naif 0,693;
en iyi model 0,692. Bu, Model B-1'deki regresyon bulgusunun sınıflandırma
karşılığı ve **alüminyum uyarısının en net kanıtı**: sistem hiç görmediği bir
malzemede güvenilir değil.

**3. Karar eşiği henüz ayarlanmadı.** Hem A'da (VB eşiği) hem B'de (olasılık
eşiği) varsayılan değerler kullanılıyor. Kaçırılan aşınma yanlış alarmdan
pahalı olduğuna göre eşik güvenli tarafa kaydırılmalı — Faz 09'un işi.

## Faz 06 — sistemi ayağa kaldır (31 Ağustos 2026)

`python scripts/train_model.py` → `python scripts/predict.py --from-nasa`

Faz 04b "hangi girdi kümesi daha iyi" sorusunu ölçmüştü. Faz 06 o soruyu
kapatıp **tek bir teslim edilebilir model** üretiyor: kaydedilmiş ağırlık,
sabitlenmiş eşik, çıkarım hattı, kapsam kontrolü ve künye.

### Teslim edilen model

| | |
|---|---|
| Model | B-1, LightGBM (`make_gbm_small`), tohum 42 |
| Eğitim | **145 satırın tamamı** — katlama ayrılmıyor |
| Girdi | `sensor+param+time`, **149 öznitelik** |
| Aktif eşik | **222 µm** (`case` kalibrasyonu) |
| Ardışık onay | k = 1 |

**Neden 145 satırın tamamı:** çapraz doğrulama katlamaları performansı
*ölçmek* içindi. Ölçüm bittikten sonra veriyi eğitim dışında tutmanın faydası
yok — sadece model zayıflar. Bedeli şu: bu modelin doğruluğu artık ölçülemez.
Raporlanan sayılar Faz 04b'nin çapraz doğrulama sayılarıdır ve öyle kalmalıdır.

`train_model.py` her çalıştırmada örneklem içi MAE'yi de basıyor: **4,81 µm**.
Bu bir performans sayısı değil, uyarıdır — aynı modelin çapraz doğrulama MAE'si
iki mertebe büyük (Faz 04b tablosu, `sensör + parametre + süre` satırı). Otuz
kat fark, örneklem içi hatanın neden raporlanamayacağının doğrudan kanıtı.

### Neden `sensör + parametre + süre` varsayılan

Faz 04b'de `parametre + süre` **bilinen** koşullarda daha iyiydi (108/114 µm vs
138/165 µm). Buna rağmen varsayılan değil, çünkü teslim senaryosu bilinen koşul
değil: sahada malzeme alüminyum, eğitim verisinde alüminyum yok. **Her saha
tahmini malzeme-dışı sınavına denk geliyor.** Orada tablo tersine dönüyor:

| Girdi kümesi | malzeme-dışı MAE |
|---|---:|
| naif taban | 309 µm |
| parametre + süre | **388 µm** — naif tabanın altında |
| sensör + parametre + süre | **271 µm** — naif tabanın üstünde |

`parametre + süre` config'den seçilebilir kalıyor
(`serving.feature_set: "param+time"`) ve bilinen koşullar için en iyi seçenek
olarak belgelidir.

### Alarm eşiği — iki kalibrasyon, biri aktif

Eşik Faz 09'un aynı koduyla (`tcm.decision.calibrate_threshold`), yalnızca
eğitim verisinden ve katlama dışı tahminlerle seçilir. Nihai modelin
tahminlerinden seçilseydi model kendi cevaplarını bildiği için hata sıfıra
yakın görünür ve eşik yanlış yere otururdu.

| Kalibrasyon | Bölme | Katlama | Eşik | İç maliyet | Durum |
|---|---|---:|---:|---:|---|
| `case` | takım bazında | 15 | **222 µm** | 31 | **AKTİF** |
| `material` | dökme demir ↔ çelik | 2 | 156 µm | 61 | pakette, aktif değil |

`material` kalibrasyonu `min_groups=2` ile zorlanır; varsayılan 3 olsaydı iki
malzeme yetersiz kalır, bölme `case`'e düşer ve sınav "görülmemiş malzeme"den
"görülmemiş takım"a dönüşerek kolaylaşırdı. Künyede her iki kalibrasyonun
`requested_split` / `actual_split` / `fell_back` alanları kayıtlıdır (ikisi de
`fell_back: false`).

**`material` (156 µm) neden aktif değil:** ölçüldü ve operasyonel olarak
savunulamaz çıktı — 15/15 takımda alarm ömrün ilk yarısında, 9/15'inde ilk
geçişte çalıyor. Gerekçe ve tam tablo "Yol boyunca yanlış çıkan şeyler"
bölümünde.

### Kapsam dışı uyarısı

Model hiçbir durumda susmuyor; her tahminin yanına o tahminin kapsam içinde mi
olduğu yazılıyor. İki seviye var ve karıştırılmamalı:

- **Kapsam dışı (sert)** — görülmemiş malzeme kodu ya da görülmemiş kesme
  koşulu. Faz 04b'de ölçülmüş bir başarısızlık var; tahmin basılır ama
  güvenilmez.
- **Aralık uyarısı (yumuşak)** — malzeme ve koşul tanıdık ama sayısal bir
  parametre eğitim aralığının dışında (örneğin takım eğitimdeki en uzun
  takımdan uzun süredir kesiyor). Ekstrapolasyon; ağaç tabanlı model aralık
  dışında sabit tahmine takılır.

Eğitim kapsamı: malzeme `[1, 2]`, 8 kesme koşulu, `feed [0,25–0,5]`,
`doc [0,75–1,5]`, `cum_time [0–929]`.

Alüminyum senaryosunun benzetimi:

```bash
python scripts/predict.py --from-nasa --simulate-unseen-material
```

### Model paketi

| Dosya | Git | İçerik |
|---|---|---|
| `runs/model_b1/model_b1.joblib` | hayır | model + her şey (yeniden üretilebilir) |
| `reports/model_b1_package.json` | **evet** | öznitelik listesi, eşikler, kapsam, künye, joblib sha256 |
| `reports/model_b1_baselines.csv` | **evet** | 149 özniteliğin referans istatistikleri |

İkili dosya depoya girmiyor (yeniden üretilebilir); künye ve taban çizgileri
metin oldukları için giriyor. Böylece "hangi model, hangi eşikle, hangi kodla"
sorusu git geçmişinden yanıtlanabiliyor. `model_sha256` künye ile ikilinin
birbirine ait olduğunu doğruluyor.

**Taban çizgileri bir dönüşüm değil.** Modele giren değerler ham kalıyor.
Gradyan artırma ağaçları eşik tabanlı çalıştığı için girdiyi ölçeklemek
bulunan bölünmeleri değiştirmez; buna karşılık çıkarımda uygulanmayı unutmak
sessiz bir hata kaynağı olurdu. İstatistikler yalnızca kapsam kontrolü ve hata
ayıklama için saklanıyor.

### Eğitim/çıkarım tutarlılığı

Üretimdeki en sinsi hata sınıfı, özniteliklerin eğitimde bir kodla, çıkarımda
başka bir kodla hesaplanmasıdır — model çökmez, sessizce yanlış tahmin eder.
Bu projede risk somuttu: aynı döngü iki yerde kopyalanmıştı.

Çekirdek tek bir modüle alındı (`tcm.features.extract`) ve eğitim de çıkarım da
onu çağırıyor. Ortaklaştırılanlar: öznitelik hesabı, `cum_time`, `condition`
kimliği ve ham `mill.mat` alanlarının ortak şemaya çevrilmesi
(`tcm.datasets.nasa.run_table` — `VB` mm→µm çevrimi dahil).

`predict.py --from-nasa` öznitelikleri önbellekten okumaz, **ham sinyalden
yeniden üretir** — iki yolun gerçekten aynı olduğu ancak böyle gösterilebilir.
`tests/test_serving.py::TestTrainingInferenceParity` bunu sınıyor.

Refaktör doğrulaması: 145×154 öznitelik tablosu ham veriden yeniden üretildi ve
eskisiyle karşılaştırıldı — sütun sırası dahil birebir aynı, en büyük sayısal
fark 0.

### Örneklem içi uyarısı

`predict.py --from-nasa` nihai modelin eğitim verisiyle çalıştığı için 145/145
satır örneklem içi. Paket bunu satır bazında işaretliyor (`in_sample`) ve betik
uyarı basıyor: bu çalıştırma **hattın çalıştığını gösterir, performansını
ölçmez**.

## Faz 09 — karar mantığı (28 Ağustos 2026)

`python scripts/run_decision_rule.py`

Model bir VB sayısı üretiyor; "takımı değiştir" demek ayrı bir karar. Varsayılan
eşik (aşınma sınırının kendisi, 300 µm) optimal değil çünkü modelin hatası var
ve hata türlerinin maliyeti simetrik değil.

**Maliyet tanımı:** 1 kaçırılan aşınma = 5 yanlış alarm. Bu bir işletme kararı,
teknik değil; `config/default.yaml` içinde `decision.cost_missed` ile değişir.

**Sızıntıyı önleyen kurgu:** eşik, her dış katlamanın *içinde* ikinci bir çapraz
doğrulama ile ve yalnızca eğitim verisiyle seçilir. Test verisi eşik seçimine
hiçbir noktada karışmaz.

### Sonuç

> Aşağıdaki sayılar **31 Ağustos 2026'da düzeltilen kilitlenme hatasından
> sonraki** değerlerdir. Eski (hatalı) sayılar için "Yol boyunca yanlış çıkan
> şeyler" bölümüne bakın.

| Sınav | Kural | Kaçırılan | Yanlış alarm | Seçilen eşik | Maliyet |
|---|---|---:|---:|---:|---:|
| vaka-dışı | sabit (300 µm) | 4 | 13 | 300,0 | 33 |
| vaka-dışı | **ayarlı** | **0** | 21 | **272,2** | **21** |
| koşul-dışı | sabit (300 µm) | 5 | 20 | 300,0 | 45 |
| koşul-dışı | **ayarlı** | **1** | 28 | **257,6** | **33** |
| malzeme-dışı | sabit (300 µm) | 23 | 39 | 300,0 | 154 |
| malzeme-dışı | **ayarlı** | **22** | 44 | **237,0** | **154** |

**Toplam maliyet 232 → 208 (−%10,3).**

Üç okuma:

- **Bilinen takımda (vaka-dışı) kaçırılan aşınma sıfıra iniyor**, koşul-dışında
  bire. Karar kuralının değeri burada.
- **Görülmemiş malzemede karar kuralı hiçbir şey kazandırmıyor** (154 → 154).
  Eşiği düşürmek 1 kaçırma kurtarıyor, karşılığında 5 yanlış alarm getiriyor —
  5:1 oranında tam başabaş. Model o sınavda o kadar bozuk ki eşiği oynatmak
  fayda üretmiyor. Bu, alüminyum uyarısının bir başka biçimi.
- Seçilen eşikler her sınavda 300'ün altında (272 / 258 / 237 µm) — model
  sistematik olarak eksik tahmin ettiği için karar kuralı güvenli tarafa
  kayıyor. Fizikten beklenen davranış.

**Ardışık onay (histerezis) işe yaramadı:** iç seçim her katlamada k = 1 seçti.
Tahminler zaten düzgün ilerlediği için (`cum_time` monoton) gürültü bastırmaya
ihtiyaç yok.

## Yol boyunca yanlış çıkan şeyler

Bu bölüm bilerek tutuluyor. Düzeltilen hatalar, hiç yapılmamış gibi
davranılırsa tekrar yapılır — ve buradakilerin ikisi zaten tekrar edildi.

### 1. Alarm kilidinin takımlar arasına taşması (iki kez yapıldı)

`apply_consecutive` alarmı **kilitler**: bir kez çaldıktan sonra sönmez.
Tek bir takımın ömrü içinde bu doğrudur, çünkü aşınma geri dönmez. Ama birden
çok takımın tahminleri arka arkaya eklenmiş bir dizide uygulanırsa, **ilk
takımdaki alarm sonraki bütün takımları da alarmda gösterir**.

**Birinci kez — eşik seçiminde (Faz 09).** İç çapraz doğrulamanın birleştirilmiş
tahmin dizisine uygulanıyordu. Optimizasyon, erken alarmların her şeyi
zehirlemesinden kaçınmak için eşiği **300 → 421 µm**'ye, yani güvensiz tarafa
çıkarıyordu. Düzeltildi; `tests/test_decision.py::TestLatchingAcrossToolsRegression`
regresyon testi yazıldı.

**İkinci kez — dış değerlendirmede (Faz 09, Faz 06'da bulundu).** Aynı hata
`run_decision_rule.py` içindeki sonuç tablosunda hayatta kaldı: kilit
katlamanın tamamına uygulanıyordu. Etkisi:

| Sınav | Kaçırılan (hatalı) | Kaçırılan (doğru) | Yanlış alarm (hatalı→doğru) | Maliyet (hatalı→doğru) |
|---|---:|---:|---|---|
| vaka-dışı | 0 | 0 | 21 → 21 | 21 → 21 |
| koşul-dışı | 0 | **1** | 34 → 28 | 34 → 33 |
| malzeme-dışı | 11 | **22** | 48 → 44 | 103 → **154** |
| **TOPLAM** | **11** | **23** | 103 → 93 | **158 → 208** |

Görülmemiş malzemede kaçırılan aşınma **iki katına** çıktı. Hatanın yönü
tehlikeliydi: kilit taştığında satırlar "alarm verildi" sayılıyor, dolayısıyla
gerçekte kaçırılan aşınmalar sayılmıyordu — **sistem olduğundan güvenli
görünüyordu**.

Karar kuralının kazancı da −%22,5'ten −%10,3'e indi.

`vaka-dışı` sınavı etkilenmedi, çünkü o protokolde her katlama tek bir
takımdan oluşuyor; taşacak ikinci takım yok.

**Eşikler değişmedi** (272,2 / 257,6 / 237,0 µm) — onlar iç çapraz doğrulamadan
geliyor ve orası birinci düzeltmeyle zaten onarılmıştı. Değişen yalnızca
raporlanan performans.

**Neden birinci düzeltme ikincisini yakalamadı:** yazdığımız regresyon testi
kütüphane fonksiyonunu (`alarm_flags`) sınıyordu, betiğin *o fonksiyonu
çağırdığını* değil. Betik yanlış olan `apply_consecutive`'i çağırmaya devam
etti ve test yeşil kaldı. Ayrıca doğru fonksiyonun adı `_flags_by_group` idi —
alt çizgi onu "özel" gösteriyor, çağıran kodu genel olana itiyordu.

İki düzeltme yapıldı: fonksiyon `alarm_flags` adıyla açıkça genel hale
getirildi, ve `tests/test_decision_rule_script.py` betiği **doğrudan çağıran**
regresyon testleriyle eklendi. O dosyadaki ilk test, sentetik verinin iki
uygulamayı gerçekten ayırt ettiğini kanıtlıyor — aksi halde test hiçbir şeyi
korumazdı.

> **Ders:** kütüphaneyi test etmek, kütüphaneyi doğru kullandığınızı test etmez.

### 2. `conservative` eşik seçim kuralı (Faz 06'da kaldırıldı)

Faz 06'da eşik iki kurguyla kalibre edildi (`case` ve `material`) ve
`conservative` adında bir kural ikisinin **düşüğünü otomatik** seçiyordu.
Gerekçesi makuldü: kaçırılan aşınma yanlış alarmdan 5 kat pahalı, belirsizlikte
güvenli taraf düşük eşik.

Sonucu makul değildi. Kural her seferinde yalnızca **2 katlamayla** hesaplanan
gürültülü `material` kalibrasyonunu (156 µm) seçiyordu. O eşik ölçüldüğünde
(`scripts/threshold_sweep.py`):

| Eşik | Kaçırılan | Yanlış alarm | Ort. alarm konumu | İlk yarıda | **İlk geçişte** |
|---:|---:|---:|---:|---:|---:|
| **156** | 0 | 61 | ömrün %12'si | 14/15 | **9/15** |
| 190 | 4 | 58 | %15 | 13/15 | 8/15 |
| 222 | 9 | 53 | %16 | 11/15 | 7/15 |
| 237 | 9 | 50 | %18 | 11/15 | 6/15 |

156 µm'de **15 takımın 9'unda alarm daha ilk geçişte çalıyor** ve ortalama
alarm konumu takım ömrünün %12'si. Yani kural pratikte "her takımı takar takmaz
at" diyor — bunun için modele gerek yok.

**Kök neden maliyet fonksiyonunda:** yanlış alarm **geçiş başına** sayılıyor.
Ömrünün başında atılan bir takım, tabloda "1 yanlış alarm" olarak görünüyor;
gerçekte ise bütün bir takım ömrü çöpe gidiyor. Maliyet fonksiyonu erken
değiştirmenin bedelini ölçmediği için optimizasyon eşiği dibe itiyor.

Karar: `conservative` kaldırıldı, aktif eşik **`case` kalibrasyonu (222 µm)**.
`material` kalibrasyonu pakette ve künyede duruyor ama varsayılan değil.

> **Ders:** "belirsizlikte güvenli tarafı seç" kuralı, güvenli tarafın bedeli
> maliyet fonksiyonunda temsil edilmiyorsa otomatikleştirilemez.

**Gelecek çalışma:** maliyet modeline **takım bazında ömür israfı** terimi
eklenmeli (örneğin "alarm ömrün %p'sinde çaldıysa (1−p) takım maliyeti").
O terim eklendiğinde `material` kalibrasyonu yeniden değerlendirilmelidir —
şu anki reddi, kalibrasyonun kendisine değil, onu değerlendiren maliyet
fonksiyonunun eksikliğine dayanıyor.

### 3. Kabul kriterinin ulaşılamaz olması (Faz 04)

Faz 00'da konan "gecikme ≤ 5 geçiş" hedefi, etiketin kendi ölçüm gürültüsünün
altında doğruluk istiyordu. Ölçüldü, ulaşılamaz olduğu gösterildi ve
overshoot (µm) metriğiyle değiştirildi. Ayrıntı yukarıda, Faz 04 bölümünde.

### 4. PHM birleştirmenin reddi (Faz 04c)

Denendi, ölçüldü, üç ağırlıkta da işe yaramadı ve alarm davranışını bozdu.
Ayrıntı yukarıda, Faz 04c bölümünde. Olumsuz sonuç da sonuçtur.

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
