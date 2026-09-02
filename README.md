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
| 12 | Sabit 100/20/25 bölme deneyi | **tamamlandı** |

## Sonuçların kaynağı (provenance)

Rapordaki her sayı bir çalıştırmadan gelir. Hangi kodla, hangi ayarla ve hangi
kütüphane sürümleriyle üretildiği bilinmiyorsa o sayı doğrulanamaz — ve üç ay
sonra yeniden üretilemez.

Bu yüzden **sonuç üreten her betik** çıktısının yanına künye yazar
(`tcm.provenance`): git commit'i, çalışma ağacının kirli olup olmadığı, config
dosyasının sha256'sı, zaman damgası ve LightGBM / scikit-learn / NumPy /
pandas / SciPy / PyTorch sürümleri. Kaydedilen tablolar ayrıca bir `git_hash`
sütunu taşır.

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
| Faz 12 · sabit bölme | `reports/holdout_split_summary.csv`, `reports/holdout_tree_sweep.csv` | `python scripts/run_holdout_split.py --save` |

### Künye denetimi (31 Ağustos 2026)

Faz 04–04d sonuçları künye eklenmeden önce (28 Ağustos) üretilmişti. Hepsi
yeniden çalıştırılıp kayıtlı tablolarla karşılaştırıldı. **İki tabloda kayma
bulundu, ikisi de düzeltildi:**

| Bölüm | Denetim sonucu |
|---|---|
| Faz 04 · Model A | **Hata bulundu.** Kod, `run_time`/`cum_time` sütunlarını sessizce öznitelik olarak yutuyordu (168 → 170). README'deki sayılar doğruydu; kod düzeltildi ve tablo birebir geri geldi. |
| Faz 04b · Model B-1 | **Kayma bulundu.** README dokuz hücrede `skew`/`kurtosis` koruması öncesi değerleri taşıyordu. CSV ile eşitlendi. |
| Faz 04c · Model B-2 | temiz — sayılar değişmedi |
| Faz 04d · sınıflandırma | temiz — sayılar değişmedi |

İkisinin de ayrıntısı "Yol boyunca yanlış çıkan şeyler" bölümünde. Dört
betiğin dördüne de künye eklendi; bu sınıf hata bir daha sessizce geçemez.

İki uyarı daha:

- **Faz 09 sayıları 31 Ağustos 2026'da değişti** (kilitlenme hatası
  düzeltildi). Eski ve yeni değerlerin karşılaştırması "Yol boyunca yanlış
  çıkan şeyler" bölümünde.
- **Bazı künyeler "kirli" (dirty) işaretli.** O sayılar henüz commit
  edilmemiş bir çalışma ağacıyla üretildi ve o git hash'inden birebir
  yeniden üretilemezler. Betikler temiz ağaçta yeniden çalıştırılırsa künye
  temiz commit'e bağlanır.

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
| sadece parametre (süresiz) | 4 | 206,20 | 202,86 | 238,87 |

> **Düzeltildi (31 Ağustos 2026).** Bu tablo bir süre `145,38 / 173,26 /
> 221,34` gibi **başka** sayılar içeriyordu. Sebep teşhis edildi: o değerler,
> `skew`/`kurtosis` koruması eklenmeden **önceki** kodla üretilmişti.
>
> Koruma `cd9912b` (Faz 04b) commit'inde eklendi: neredeyse sabit bir sinyalde
> çarpıklık ve basıklık tanımsızdır — payda sıfıra gider, SciPy güvenilmez sayı
> döndürür. NASA'nın **`smcDC`** (DC iğ motor akımı) kanalı 145 koşunun
> 23'ünde bu durumda; koruma o hücrelerde 0 döndürüyor (2 öznitelik × 23 satır
> = 46 hücre).
>
> Bu, gözlenen desenin tamamını açıklıyor: değişen dokuz hücrenin hepsi
> **sensör özniteliği içeren** satırlarda, `naif taban` ve `parametre + süre`
> satırları ise değişmemiş — çünkü koruma yalnızca sensör özniteliklerine
> dokunuyor. Doğrulama: koruma koddan geçici olarak çıkarılıp tablo yeniden
> üretildiğinde eski dokuz değerin **dokuzu da virgülden sonrasına kadar**
> yeniden elde edildi.
>
> **Doğru olan yukarıdaki (korumalı) sayılardır.** Koruma öncesi değerler,
> sayısal olarak güvenilmez skew/kurtosis girdileriyle eğitilmiş modellerden
> geliyordu.

### Üç bulgu

**1. Kesme parametreleri TEK BAŞINA işe yaramıyor, ama süreyle birlikte her
şeyi geçiyor.**

| Girdi | vaka-dışı | koşul-dışı |
|---|---:|---:|
| sadece parametre (süresiz) | 206,20 | 202,86 |
| **parametre + süre** | **108,38** | **114,19** |

Kümülatif süreyi eklemek hatayı neredeyse yarıya indiriyor.

Fizik açık: kesme parametreleri aşınma **hızını** belirler, **miktarını** değil.
"Çelikte, 0,5 mm/dev ilerlemeyle" bilgisi takımın şu an ne kadar aşındığını
söylemez — ne kadar süredir kestiğini de bilmek gerekir. Hız × süre = aşınma.

Bu, istenen girdi tanımının (malzeme + ilerleme + kesme parametreleri)
**eksik** olduğu anlamına gelir: kümülatif kesme süresi olmadan çalışmaz.
Sahada bilinen bir değerdir (yeni takımda sayaç sıfırlanır).

**2. BİLİNEN koşullarda 5 girdilik parametre modeli, 144 öznitelikli sensör
modelini açık farkla geçiyor** — ve sensör eklemek onu bozuyor.

| Girdi | vaka-dışı | koşul-dışı |
|---|---:|---:|
| parametre + süre (5) | **108,38** | **114,19** |
| sadece sensör (144) | 140,19 | 166,57 |
| sensör + parametre + süre (149) | 138,47 | 164,27 |

Sensör eklemek parametre modelini 108,38 → 138,47 µm'ye **kötüleştiriyor**:
149 özniteliğin 144'ü gürültü taşıyor ve modelin dikkatini dağıtıyor.

Sebebi NASA'nın 250 Hz örneklemesi. PHM'de (50 kHz) sensörler naif tabanı
geçmişti; burada geçemiyor. Sensörün değeri, sensörün kalitesine bağlı.

Bu üstünlüğün **yalnızca bilinen koşullar için** geçerli olduğuna dikkat: bir
sonraki bulgu tabloyu tersine çeviriyor ve teslim edilen model o yüzden
parametre modeli değil (bkz. Faz 06).

**3. Malzeme değişince tablo tersine dönüyor.** Görülmemiş malzemede
parametre modeli çöküyor (388,16 — naif tabandan bile kötü), sensör modeli
en iyisi oluyor (216,85).

Sebep: parametre modeli dökme demirin aşınma hızını öğrenip çeliğe uyguluyor
ve yanılıyor. Sensör modeli hızı çıkarsamak yerine **durumu ölçtüğü** için
daha zarif bozuluyor.

**Bu doğrudan Tomtaş'ın alüminyum sorusuna denk geliyor:** hiç görülmemiş bir
malzemede parametre tabanlı model güvenilmez; orada yalnızca sensör işe yarar.

### Yan gözlemler

- `cum_time` öznitelik öneminde 1. sırada; `feed` ve `rpm` sıfır. (`rpm` NASA'da
  sabit olduğu için sıfır çıkması beklenen davranış.) —
  `reports/model_b1_importance.csv`

### Aşırı aşınmış koşular hatanın yarısını üretiyor

NASA'da VB 1530 µm'ye kadar çıkıyor — ISO sınırının beş katı. Bu koşular
aşınma tahmini için anlamlı bir çalışma bölgesi değil. VB ≤ 600 µm ile
sınırlandığında (126/145 koşu) sensör modelinin MAE'si:

| Sınav | Tam veri | VB ≤ 600 µm | Değişim |
|---|---:|---:|---:|
| vaka-dışı | 140,19 | **78,44** | −%44,0 |
| koşul-dışı | 166,57 | **88,10** | −%47,1 |
| malzeme-dışı | 216,85 | **109,18** | −%49,7 |

Hatanın yaklaşık yarısı, zaten çoktan hurdaya çıkmış takımlardan geliyor.
Kullanılabilir aralıkta model iki kat daha iyi. — `reports/model_b1_extras.csv`

### Sensörler malzemeyi ne kadar ele veriyor?

Sensör özniteliklerinden malzemeyi tahmin etmeyi denedik. **Bölme protokolü
sonucu belirliyor**, o yüzden ikisi de veriliyor (çoğunluk sınıfı tabanı 0,676):

| Bölme | Doğruluk | Tabanın üstünde |
|---|---:|---:|
| takım bazında | 0,890 | +0,214 |
| **koşul bazında** | **0,717** | **+0,041** |

**Asıl sayı koşul bazındaki.** Takım bazında bölme iyimser, çünkü `condition`
kimliği malzemeyi zaten içeriyor (malzeme + kesme derinliği + ilerleme):
dışarıda bırakılan takımın kardeşleri aynı koşulla eğitimde kalıyor ve model
"bu imza = bu koşul = bu malzeme" ezberleyebiliyor.

Görülmemiş bir kesme koşulunda sensörler malzemeyi tabanın yalnızca 4 puan
üstünde bilebiliyor. Bu, **alüminyum uyarısını zayıflatmıyor, güçlendiriyor**:
sistem hiç görmediği bir malzemeyi sinyalden tanıyıp kendini uyaramaz.
— `reports/model_b1_extras.csv`

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
en iyi model 0,692.

Dikkat: bu, **sınıflandırma katmanı** için geçerli. Regresyonda durum farklı —
Model B-1'de sensör içeren kümeler görülmemiş malzemede naif tabanı geçiyor
(216,9–271,5 vs 308,96), yalnızca `parametre + süre` altına düşüyor (388,2).
Yani sinyal orada hâlâ iş görüyor, ama ikili karara indirgendiğinde geriye
kalan bilgi naif tabanın üstüne çıkmaya yetmiyor.

Bu ayrım **alüminyum uyarısının en net kanıtı**: sistem hiç görmediği bir
malzemede karar üretecek kadar güvenilir değil.

**3. Karar eşiği henüz ayarlanmadı.** Hem A'da (VB eşiği) hem B'de (olasılık
eşiği) varsayılan değerler kullanılıyor. Kaçırılan aşınma yanlış alarmdan
pahalı olduğuna göre eşik güvenli tarafa kaydırılmalı — Faz 09'un işi.

## Faz 05 — derin öğrenme (31 Ağustos 2026)

`python scripts/run_model_deep.py --seeds 3 --save`

Gradyan artırmadan farkı: öznitelikleri biz tanımlamıyoruz. Orada "RMS hesapla,
3. mertebe enerjisini al" diyorduk; burada 1B-CNN + GRU ham sinyalden neye
bakacağını kendisi öğreniyor. Kesme parametreleri evrişimden geçmez, GRU
çıktısına eklenir.

Aynı üç sınav, aynı protokol, aynı metrikler — yoksa karşılaştırma anlamsız
olurdu.

### Neden üç tohum

145 örnekle derin ağ eğitildiğinde tek bir koşunun sonucu güvenilir değil:
ağırlık başlangıcı ve yığın sırası şansa bağlı. Model üç tohumla (42, 43, 44)
tekrarlanıp **ortalama ve saçılım** birlikte raporlanıyor.

Ölçüt şu: **saçılım, iki model arasındaki farktan büyükse "kazandı" demek
anlamsızdır** — sonuç modelden değil, rastgele başlangıçtan geliyor olabilir.

### Sonuç

MAE (µm), CNN+GRU üç tohumun ortalaması ± saçılım:

| Sınav | CNN + GRU | Gradyan artırma | Fark | Saçılım | Hüküm |
|---|---:|---:|---:|---:|---|
| vaka-dışı | **123,53** | 144,06 | 20,53 | ±2,12 | **GEÇTİ** |
| koşul-dışı | **137,79** | 159,74 | 21,95 | ±10,10 | **GEÇTİ** |
| malzeme-dışı | 252,26 | 257,65 | 5,38 | ±18,29 | **KARARSIZ** |

**CNN+GRU üç sınavın ikisinde gradyan artırmayı kesin olarak geçti.**

Malzeme-dışı sınavında fark (5,38 µm) saçılımın (±18,29) çok altında —
bu sınavda hangi modelin iyi olduğu **ölçülemiyor**. "Geçti" demek yanlış
olurdu; "geçemedi" demek de. Tablodaki hüküm bu yüzden KARARSIZ.

### Bu beklenen sonuç değildi

Kod yazılırken beklenti açıkça düşüktü: "145 örnek derin öğrenme için çok
küçük, ağ ezberler". Ölçüm bunu **iki sınavda çürüttü**. Muhtemel sebep,
modelin kasıtlı olarak küçük tutulması ve düzenlileştirmenin yüksek olması
(dropout 0,3, weight decay 1e-3, erken durdurma yok).

Yine de teslim edilen model **B-1 (gradyan artırma)**. Üç gerekçe:

1. **Malzeme-dışı sınavında üstünlük gösterilemedi** — ve teslim senaryosu
   tam olarak o sınav (sahada alüminyum var, eğitimde yok).
2. **Maliyet:** CNN+GRU vaka-dışı sınavında 2399 s, gradyan artırma 2,8 s.
   Yaklaşık 850 kat.
3. **Açıklanabilirlik:** gradyan artırma hangi özniteliği ne kadar kullandığını
   raporlayabiliyor; jüriye savunulacak bir sistemde bu ağır basıyor.

Bulgu raporlanacak: derin öğrenme bu veri ölçeğinde **çalıştı**, ama teslim
kararını değiştirecek yerde (görülmemiş malzeme) fark ölçülemedi.

### Künye

`reports/model_deep_summary.csv` + `reports/model_deep_summary.provenance.json`
— git `74d4de8` (temiz çalışma ağacı), 3 tohum, 120 epoch, PyTorch 2.13.0.

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

## Faz 12 — sabit 100/20/25 bölmesi (1 Eylül 2026)

`python scripts/run_holdout_split.py --detail`

Projenin geri kalanı çapraz doğrulama kullanıyor. Bu bölüm, **klasik
eğitim/doğrulama/test bölmesini** aynı veri üzerinde deneyip ne olduğunu
ölçüyor. Amaç yeni bir model üretmek değil; çapraz doğrulama tercihinin
gerekçesini tahmin olmaktan çıkarıp **ölçüme** dayandırmak.

Teslim edilen sistem bu deneyden etkilenmedi — Faz 06 paketi olduğu gibi
duruyor.

### Bölme

145 koşu, takım bazında bölündüğünde 100/20/25'i tam tutturuyor:

| Küme | Takımlar | Koşu |
|---|---|---:|
| Eğitim | 1, 2, 4, 7, 9, 10, 12, 13, 14, 15, 16 | 100 |
| Doğrulama | 3, 5 | 20 |
| Test | 8, 11 | 25 |

Doğrulama kümesi iki iş yapıyor, ikisi de test kümesine dokunmadan: **erken
durdurma** (kaç ağaç kurulacağı) ve **alarm eşiği kalibrasyonu**. Test yalnızca
en sonda, bir kez kullanılıyor.

Karşılaştırma için satır bazlı rastgele bölme de çalıştırıldı.

### Sonuç

| Bölme | Ağaç | Eşik (µm) | Test MAE | Test RMSE | Yakalama | Kaçırılan | Yanlış alarm |
|---|---:|---:|---:|---:|---:|---:|---:|
| takım bazlı | 10 | 402,0 | 164,33 | 192,24 | 0,545 | 5 | 0 |
| rastgele | 133 | 207,0 | **114,94** | **166,32** | **1,000** | 0 | 7 |

**Rastgele bölme %30 daha iyi görünüyor — ve sorun tam olarak budur.** O
kurguda 15 takımın 14'ü hem eğitimde hem testte. Aynı takımın komşu koşuları
neredeyse aynı sinyali taşıdığı için model, test satırlarına çok benzeyen
satırları eğitimde görmüş oluyor. Bu sayı genellemeyi değil ezberi ölçüyor.
Betik bu durumu otomatik tespit edip uyarı basıyor.

### Asıl bulgu: 20 satırlık doğrulama kümesi modeli bozdu

Ağaç sayısı taranınca doğrulama ve test **zıt yönleri** gösteriyor:

| Ağaç | Doğrulama MAE | Test MAE |
|---:|---:|---:|
| 10 | **151,84** | 164,33 |
| 50 | 166,75 | 120,86 |
| 100 | 185,45 | 110,84 |
| 300 | 207,13 | 107,43 |
| 600 | 212,85 | **105,04** |

Erken durdurma doğrulama hatasına baktığı için **10 ağaç** seçti — test için
mümkün olan en kötü seçim. 600 ağaçla test MAE'si 164 yerine 105 olacaktı.

Sebep: doğrulama kümesi yalnızca 2 takımdan (20 koşu) oluşuyor ve o iki takımın
aşınma davranışı test takımlarınınkine benzemiyor. **Sabit bölme bu veri
ölçeğinde sadece daha gürültülü bir tahmin vermiyor; aktif olarak yanlış model
seçiyor.**

### Yan etki: eşik güvensiz tarafa kaydı

Seçilen alarm eşiği **402 µm** — aşınma sınırının (300) *üstünde*. Faz 09'da
doğru kalibre edilmiş eşikler sınırın altında çıkmıştı (272 / 258 / 237), çünkü
model sistematik olarak eksik tahmin ediyor.

402'nin sebebi doğrulama kümesinde optimal olması: orada hiç aşınma
kaçırmadan yanlış alarmı 1'e indiriyor. Ama test takımlarında modelin
tahminleri daha aşağıda oturduğu için beş aşınmayı birden kaçırıyor.

### Precision 1,000 iyi haber değil

| Metrik | Değer |
|---|---:|
| Precision | 1,000 |
| Recall | 0,545 |
| F1 | 0,706 |
| Balanced accuracy | 0,773 |

Precision mükemmel çünkü eşik o kadar yüksek ki model neredeyse hiç alarm
vermiyor; verdiği az sayıda alarm doğru çıkıyor. Asıl sayı **recall 0,545** —
11 aşınmış koşunun 5'i kaçırıldı. Takım aşınmasında kaçırılan aşınma (FN),
yanlış alarmdan (FP) çok daha pahalıdır.

Satır satır döküm için `--detail` kullanın; test kümesindeki her koşunun
gerçek/tahmin değeri ve TP/FN/FP/TN durumu basılır.

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

### 3. Model A'nın zaman sütunlarını sessizce yutması (Faz 06 denetiminde bulundu)

Model A'nın sorusu şu: **sensörler, geçiş sayısından okunabilen eğilimin
ötesine ne katıyor?** Naif tabanı geçip geçmediği bu yüzden anlamlı.

`run_model_a.py` özniteliklerini "meta olmayan her sütun" diye seçiyordu:

```python
META_COLUMNS = {"cutter", "cut", "vb_um", "flute_spread_um"}
feature_columns = [c for c in data.columns if c not in META_COLUMNS]
```

Faz 04c'de, Model B-2'nin iki veri setini birleştirebilmesi için PHM
tablosuna `run_time` ve `cum_time` sütunları eklendi. Model A onları
**sessizce öznitelik olarak yuttu**: 168 → 170.

`cum_time` PHM'de geçiş sayısının neredeyse birebir monoton bir fonksiyonu —
yani naif tabanın girdisinin ta kendisi. Model A artık "sensörler ne katıyor"
sorusunu ölçmüyordu; tabanın bilgisini hazır alıp üstüne sensör ekliyordu.

Etkisi, MAE'yi **iyileştirdiği** için fark edilmesi zor:

| Model | 170 öznitelik (kirli) | 168 öznitelik (doğru) |
|---|---:|---:|
| 1 · GBM (ham) | 19,18 | **19,91** |
| 2 · GBM + monoton | 19,72 | **20,74** |
| 3 · GBM + normalize + monoton | 19,71 | **19,55** |

Hata sistemi olduğundan **iyi** gösteriyordu — ve tam da naif tabanla
karşılaştırmanın anlamını yok ederek.

**Nasıl yakalandı:** Faz 06 denetiminde üç betik yeniden çalıştırılıp kayıtlı
tablolarla karşılaştırıldı. Model A değişti, ötekiler değişmedi. İlk şüphe
`skew`/`kurtosis` korumasıydı (Faz 04b'deki kaymanın sebebi) ama koruma PHM
verisinde **hiç** devreye girmiyor — sıfır hücre. Kesin ipucu
`model_a_channels.csv`'nin değişmemiş olmasıydı: kanal alt kümeleri
`select_channels` ile kanal önekine göre seçiliyor, `run_time`/`cum_time`
hiçbir kanal önekiyle eşleşmediği için o tabloya hiç girmemişti. Ana tablo
değişip alt küme tablosu değişmiyorsa, değişen şey kanal özniteliği değildir.

**Düzeltme:** iki sütun `META_COLUMNS`'a eklendi. README'deki Model A
sayıları zaten doğruydu (168 öznitelikli koşudan geliyorlardı); değişen kod
onlardan uzaklaşmıştı. Kod düzeltilince tablo birebir geri geldi.

> **Ders:** "meta olmayan her sütun özniteliktir" kuralı, tabloya sonradan
> sütun ekleyen her değişiklikte sessizce bozulur. Öznitelik kümesi ya açıkça
> listelenmeli ya da dışlama listesi tablodan türetilmelidir.

### 4. Kabul kriterinin ulaşılamaz olması (Faz 04)

Faz 00'da konan "gecikme ≤ 5 geçiş" hedefi, etiketin kendi ölçüm gürültüsünün
altında doğruluk istiyordu. Ölçüldü, ulaşılamaz olduğu gösterildi ve
overshoot (µm) metriğiyle değiştirildi. Ayrıntı yukarıda, Faz 04 bölümünde.

### 5. PHM birleştirmenin reddi (Faz 04c)

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
