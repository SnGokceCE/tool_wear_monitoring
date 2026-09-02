# CNC Frezelemede Takım Aşınması Tahmini için Uçtan Uca Bir Makine Öğrenmesi Sistemi

**Staj Raporu**

Kurum: Tomtaş Havacılık
Depo: `github.com/SnGokceCE/tool_wear_monitoring`

---

## Özet

Bu çalışmada, CNC frezeleme işlemlerinde takım aşınmasını sensör verisinden tahmin eden ve aşınmış/sağlam kararı üreten bir sistem geliştirilmiştir. İki açık veri seti (PHM 2010 ve NASA Milling) üzerinde üç farklı model kurgusu, üç farklı genelleme protokolü altında ölçülmüştür.

Çalışmanın ana katkısı en yüksek doğruluk değeri değil, **ölçümün kendisinin güvenilirliğidir**. Alanda yaygın olan iyimser değerlendirme kurgularından kaçınmak için takım bazında bölme, eğilim çıkarılmış korelasyon analizi ve iç çapraz doğrulamayla eşik seçimi baştan kurulmuştur. Geliştirme boyunca sonuçları olduğundan iyi gösteren sekiz ayrı kusur tespit edilip düzeltilmiş, hepsi raporlanmıştır.

En önemli bulgu olumsuz bir bulgudur ve hiç görülmemiş bir malzemeye ilişkindir: **sınıflandırma katmanında hiçbir model naif taban çizgisini geçememekte, karar kuralının ayarlanması da kazanç sağlamamaktadır; regresyon katmanında sensör içeren modeller naif tabanı geçmekte ancak hata bilinen koşullara göre yaklaşık iki katına çıkmaktadır.** Ayrıca sistem, karşılaştığı malzemenin eğitim kapsamı dışında olduğunu sinyalden tespit edememektedir (%71,7, çoğunluk tabanı %67,6). Bu iki sonuç birlikte, sistemin alüminyum işlenen bir üretim ortamına doğrudan taşınamayacağını göstermektedir ve teslim edilen sisteme açık bir kapsam-dışı uyarı mekanizması olarak yansıtılmıştır.

---

## 1. Problem tanımı ve kapsam

### 1.1 Görev

Frezeleme işleminde kesici takım kademeli olarak aşınır. Aşınma miktarı, yan yüzeydeki aşınma bandının genişliği (VB, flank wear) ile ölçülür. Takım zamanında değiştirilmezse işlenen parçanın yüzey kalitesi bozulur ve parça hurdaya çıkar; erken değiştirilirse takım ömrü israf edilir.

Sistemden beklenen iki çıktı:

1. **VB tahmini** (regresyon, mikrometre cinsinden)
2. **Aşınmış / sağlam kararı** (sınıflandırma)

### 1.2 Kısıt: saha verisi yok

Projenin başlangıcında hedef, kurumun kendi tezgâhlarından veri toplayıp model eğitmekti. Bu mümkün olmadığı için çalışma tamamen açık veri setleri üzerine kurulmuştur. Bunun iki sonucu vardır:

- Sistem, hedef üretim ortamının malzemesi (alüminyum) ve tezgâhı üzerinde hiç eğitilmemiştir.
- Dolayısıyla asıl ölçülmesi gereken şey, modelin **görmediği koşullara genelleme yeteneğidir** — tek bir veri setindeki doğruluk değil.

Bu kısıt, çalışmanın değerlendirme tasarımını belirleyen ana etkendir.

---

## 2. Literatür ve konumlandırma

Alandaki çalışmalar üç kuşağa ayrılabilir:

1. **El yapımı öznitelik + klasik ML** (SVM, rastgele orman). Yorumlanabilir, az veriyle çalışır.
2. **Derin öğrenme** (CNN, LSTM/GRU). Ham sinyalden öznitelik öğrenir, daha çok veri ister.
3. **Transformer ve alan uyarlama.** Bu kuşak daha yüksek doğruluk için değil, **genelleme sorunu çözülmediği için** doğmuştur.

Üçüncü kuşağın varlık sebebi bu çalışmanın da çıkış noktasıdır: tek bir kesme koşulunda %99 doğruluk raporlayan modeller, malzeme veya tezgâh değiştiğinde çökmektedir. Literatürün açık yarası, laboratuvar koşullarından sahaya transferi doğrulanmış uçtan uca bir çerçevenin bulunmamasıdır.

Sinyal türleri bilgi/uygulanabilirlik ödünleşimiyle sıralanır: kesme kuvveti (en bilgilendirici, dinamometre gerektirdiği için sahada kullanılamaz) → titreşim (en iyi denge) → akustik emisyon → iğ motor akımı (en ucuz, CNC'den doğrudan okunabilir).

---

## 3. Veri setleri

| | PHM 2010 | NASA Milling |
|---|---|---|
| Etiketli örnek | 945 geçiş | 145 koşu |
| Bağımsız takım | 3 (c1, c4, c6) | 15 vaka |
| Örnekleme | 50 kHz | 250 Hz |
| Kanallar | kuvvet XYZ, titreşim XYZ, AE | titreşim ×2, AE ×2, iğ motor akımı (AC/DC) |
| Malzeme | paslanmaz çelik (tek) | dökme demir, çelik |
| Kesme koşulu | tek kombinasyon | 8 kombinasyon |
| Kuvvet kanalı | var | yok |

**PHM 2010'un sınırı:** tüm kesme parametreleri (10.400 dev/dk, 1555 mm/dk ilerleme, 0,2 mm derinlik, tek malzeme) sabittir. Bu değişkenlerin varyansı sıfır olduğu için modele girdi olarak eklenseler bile bilgi taşımazlar. Parametreye duyarlı bir sistem bu veri seti üzerine kurulamaz.

**Örneklem büyüklüğü yanılgısı:** PHM 2010'da 945 etiketli geçiş vardır, ancak bunlar yalnızca **3 bağımsız aşınma yörüngesinden** gelir. Bir takımın 315 geçişi bağımsız örnek değil, aynı takımın kademeli körelmesinin ardışık gözlemleridir. Takım bazında bölündüğünde eğitim kümesinde 2, test kümesinde 1 yörünge kalır. Bu, model seçimini belirleyen asıl kısıttır.

---

## 4. Yöntem

### 4.1 Öznitelik çıkarımı

Her kanal için 24 öznitelik hesaplanmıştır: 7 zaman alanı (RMS, standart sapma, tepe, tepeden tepeye, çarpıklık, basıklık, mutlak ortalama) ve 17 mertebe alanı özniteliği.

**Mertebe alanı kararı.** Genel amaçlı FFT bantları yerine öznitelikler iğ dönüşüne hizalanmış frekans bantlarından çıkarılmıştır:

```
İğ frekansı      : 10.400 / 60      ≈ 173,3 Hz
Örnek / devir    : 50.000 / 173,3   ≈ 288
Diş geçiş frek.  : 173,3 × 3 ağız   ≈ 520 Hz
```

Bu tasarım veriyle doğrulanmıştır: spektrumda 520 Hz (mertebe 3) ve harmoniği 1040 Hz (mertebe 6) hesaplanan yerlerde keskin tepeler olarak görülmekte, aşınmış takımda tüm bantta 1–2 kat yükselme gözlenmektedir.

Ayrıca mertebe **oranı** öznitelikleri (toplam enerjinin yüzde kaçı diş geçişinde) sensör kazancından bağımsızdır; bu, iki farklı tezgâhın verisini ortak uzayda karşılaştırabilmek için gereklidir.

### 4.2 Değerlendirme protokolleri

Kayan pencereleri rastgele bölmek, aynı takımın komşu geçişlerini hem eğitime hem teste dağıtır ve sonuçları yapay olarak şişirir. Bu çalışmada bölme **her zaman takım bazındadır** ve üç zorluk seviyesinde tanımlanmıştır:

| Protokol | Gizlenen | Cevapladığı soru |
|---|---|---|
| **vaka-dışı** | 1 takım (aynı koşulda kardeş takım eğitimde) | Aynı ayarlarla yeni bir takım takıldığında çalışıyor mu? |
| **koşul-dışı** | 1 kesme koşulunun tüm takımları | Denenmemiş bir parametre kombinasyonunda çalışıyor mu? |
| **malzeme-dışı** | 1 malzemenin tüm takımları | Hiç görmediği bir malzemede çalışıyor mu? |

Üçünün birlikte raporlanması bilinçlidir: yalnızca vaka-dışı raporlansaydı sistem yaklaşık iki kat daha iyi görünürdü.

Sabit bir eğitim/doğrulama/test bölmesi yerine çapraz doğrulama kullanılmasının gerekçesi Bölüm 5.7'de deneyle ölçülmüştür.

### 4.3 Metrikler

- **MAE / RMSE** — VB tahmininin doğruluğu.
- **Overshoot** — alarm anında takımın aşınma sınırını kaç µm aşmış olduğu. İşaret anlamlıdır: pozitif = geç alarm (parça riski), negatif = erken alarm (ömür israfı).
- **worn_recall / missed_worn** — sınıflandırmada kaçırılan aşınmış takım sayısı.
- **Asimetrik maliyet** — kaçırılan aşınma ile yanlış alarm eşit maliyetli değildir. Varsayılan oran 5:1 alınmıştır; bu teknik değil işletme kararıdır ve yapılandırmadan değiştirilebilir.

**Doğruluk tek başına yanıltıcıdır.** Hiçbir şey öğrenmeyen, daima çoğunluk sınıfını söyleyen bir model %51,7 doğruluk almakta, ancak 70 aşınmış takımın tamamını kaçırmaktadır.

### 4.4 Naif taban çizgisi

Sinyale hiç bakmadan, yalnızca geçiş sayısından VB tahmin eden bir taban çizgisi kurulmuştur. Aşınma zamanla monoton arttığı için bu şaşırtıcı derecede iyi çalışır ve **her modelin geçmesi gereken çizgidir**.

PHM 2010 üzerinde naif taban: **MAE 21,15 µm**, ortalama mutlak gecikme 34,7 geçiş, en kötü +47 geçiş (c6 katlaması).

### 4.5 Kritik keşif: ham korelasyon bir tuzaktır

Kuvvet öznitelikleri VB ile ρ ≈ 0,994 korelasyon vermektedir. Bu sonuç etkileyici değildir: VB geçiş sayısıyla monoton arttığı için, zamanla artan **her** öznitelik otomatik olarak ~1 korelasyon verir. Bu, naif tabanın zaten sahip olduğu bilgidir.

Geçiş sayısına bağlı eğilim her iki taraftan da çıkarıldığında tablo değişir:

| | Ham | Eğilim çıkarılmış |
|---|---:|---:|
| En güçlü öznitelik | 0,994 | **0,301** |
| Tutarlı yönlü öznitelik | 140/168 | **47/168** |

*(Kaynak: `reports/correlation_summary.csv`, `python scripts/explore_phm.py --save`)*

**0,30, sensörlerin naif tabana katabileceği bilginin üst sınırıdır.** Literatürde bu veri seti üzerinde raporlanan çok yüksek doğrulukların bir kısmının bu tuzaktan kaynaklandığından şüphelenmek için sebep vardır.

Ek bulgu: ham sıralamada kuvvet kanalları önde olsa da, eğilim çıkarıldığında ilk sıraları titreşim ve akustik emisyon öznitelikleri almaktadır. Kuvvet katkısının büyük bölümünün "zamanla artıyor" bileşeninden ibaret olması, sahada dinamometre bulunmamasının sanıldığı kadar büyük bir kayıp olmayabileceğini göstermektedir.

---

## 5. Modeller ve deneyler

Üç model kurgusu tanımlanmıştır. Adlandırma eğitim verisi kaynağını belirtir.

| | Eğitim verisi | Rolü |
|---|---|---|
| **Model A** | yalnızca PHM | Sensörden çıkarılabilecek bilginin üst sınırı; literatürle karşılaştırma zemini |
| **Model B-1** | yalnızca NASA | Kesme parametreleri gerçekten değiştiği için istenen tanımı karşılayan model |
| **Model B-2** | NASA + ağırlıklı PHM | Veri seti birleştirme denemesi |

Ana algoritma **LightGBM (GBDT)**'dir: 300 karar ağacı, ağaç başına 7 yaprak. Küçük ağaç seçimi bilinçlidir; bu veri ölçeğinde derin ağaçlar ezberler. Hiperparametreler muhafazakâr biçimde **sabitlenmiş, deneyerek seçilmemiştir** — test katlamalarında ayar denemek test bilgisini modele sızdırır.

### 5.1 Model A — PHM, sensör tavanı

Bölme: leave-one-cutter-out (3 katlama). Girdi: 168 öznitelik, 7 kanal.

| Model | MAE (µm) | RMSE (µm) | \|overshoot\| | en kötü overshoot |
|---|---:|---:|---:|---:|
| naif taban | 21,15 | 25,03 | 35,74 | +57,53 |
| **GBM (ham)** | **19,91** | 23,37 | **21,13** | **−2,52** |
| GBM + monoton düzleştirme | 20,74 | 23,65 | 21,13 | −2,52 |
| GBM + normalize + monoton | 19,55 | 23,89 | 34,48 | +67,84 |

**MAE'de kazanç sınırlıdır** (naif tabana göre %6). Bu beklenen bir sonuçtur: eğilim çıkarılmış korelasyon tavanı 0,30 ölçülmüştü, model o tavana yakın çalışmaktadır. Bu bir kod eksikliği değil, verinin taşıdığı bilginin sınırıdır.

**Asıl kazanç alarm davranışındadır.** Naif taban en kötü katlamada takımın sınırı 57,5 µm aşmasına izin vermektedir. GBM hiçbir katlamada geç alarm vermemekte, en kötü durumda 2,52 µm erken uyarmaktadır. Üretimde bu, hurda parça ile sağlam parça arasındaki farktır.

**Monoton düzleştirme beklentiyi karşılamadı.** Aşınmanın fiziksel olarak azalamayacağı kısıtı doğru olsa da, kümülatif maksimum uygulaması erken bir aşırı tahmini kilitlemekte ve düzeltilmesini engellemektedir (19,91 → 20,74).

**Kanal alt kümeleri:**

| Kanal kümesi | Öznitelik | MAE (µm) |
|---|---:|---:|
| tümü (7 kanal) | 168 | 20,74 |
| titreşim + AE | 96 | 23,82 |
| yalnızca kuvvet | 72 | 24,75 |
| yalnızca titreşim | 72 | 25,10 |
| yalnızca AE | 24 | 55,70 |

Titreşim + AE kombinasyonu, kuvvet kanalları olmadan tam kümeye en yakın sonucu vermektedir. Bu, sahada dinamometre kullanılamamasının etkisini niceliksel olarak ortaya koyar.

### 5.2 Model B-1 — NASA, üç protokol

MAE (µm):

| Girdi kümesi | Öznitelik | vaka-dışı | koşul-dışı | malzeme-dışı |
|---|---:|---:|---:|---:|
| naif taban | 1 | 145,99 | 164,59 | 308,96 |
| yalnızca sensör | 144 | 140,2 | 166,6 | **216,9** |
| yalnızca parametre + süre | 5 | **108,4** | **114,2** | 388,2 |
| yalnızca parametre | 4 | 206,2 | 202,9 | 238,9 |
| sensör + parametre + süre | 149 | 138,5 | 164,3 | 271,5 |

Bu tablo çalışmanın merkezindeki ödünleşimi göstermektedir:

- **Parametre + süre kümesi bilinen koşullarda açık ara kazanmaktadır** (naif tabana göre %26–31 iyileşme).
- Aynı küme **görülmemiş malzemede naif tabanın altına düşmektedir** (388,2 vs 308,96). Yani öğrendiği şey orada yardımcı olmamakta, aktif olarak yanıltmaktadır.

Sebep yorumlanabilirdir: 5 özniteliğin dördü kesme ayarı, biri kümülatif süredir. Model esas olarak "bu ayarlarda takımlar şu kadar sürede aşınır" ilişkisini öğrenmektedir. Ayarlar tanıdıkken bu ilişki güçlüdür; yeni bir malzemede geçersizdir.

Sensör kanallarının eklenmesi bu ezberi kırmakta, bilinen koşullardaki performansı düşürmekte ancak görülmemiş malzemedeki çöküşü engellemektedir.

### 5.3 Model B-2 — veri seti birleştirme denemesi

İki veri seti ortak bir öznitelik uzayında birleştirilmiştir: kuvvet (yalnızca PHM'de) ve motor akımı (yalnızca NASA'da) düşürülerek 96 ortak sensör özniteliği elde edilmiş, etiketler ortak birime çevrilmiş, PHM satırları eğitime dört farklı ağırlıkla eklenmiştir. PHM satırları hiçbir katlamada teste girmemiştir.

Ölçüt bilinçli olarak katı tutulmuştur: **tek bir ağırlık, üç protokolün hepsinde birden taban çizgisini geçmeli.** (Her protokol için ayrı ayrı en iyi ağırlığı seçmek, test kümesine bakarak seçim yapmak olurdu.)

MAE değişimi:

| PHM ağırlığı | vaka-dışı | koşul-dışı | malzeme-dışı | Geçen protokol |
|---|---:|---:|---:|---|
| w = 0,05 | +3,1% | +12,0% | −14,3% | 1/3 |
| w = 0,15 | +1,5% | +8,9% | −3,1% | 1/3 |
| w = 1,00 | −5,4% | −0,3% | +6,4% | 2/3 |

**Hiçbir ağırlık ölçütü sağlamamıştır.** Daha belirleyici olan ise alarm davranışıdır:

| | malzeme-dışı en kötü overshoot |
|---|---:|
| PHM eklenmemiş | **−60 µm** (erken, güvenli) |
| PHM eklenmiş (her ağırlıkta) | **+1230 µm** (geç, güvensiz) |

w = 0,05 MAE'de %14 kazandırırken, takımın sınırı dört katına çıkmasına izin vermektedir. Bu, tek metriğe bakarak karar vermenin somut tehlikesidir.

**Sonuç: birleştirme reddedilmiştir.** PHM'in 945 satırı tek bir kesme koşulundan geldiği için parametre etkisini öğretmeye katkısı sıfırdır; yalnızca sensör–aşınma ilişkisinin şeklini öğretebilirdi, ancak bu ilişki iki tezgâh arasında yeterince farklı çıkmıştır.

### 5.4 Aşınmış / sağlam sınıflandırması

İki yaklaşım karşılaştırılmıştır: (a) regresyon çıktısını eşikleme, (b) doğrudan sınıflandırıcı eğitme.

Dengeli doğruluk:

| Yöntem | Girdi | vaka-dışı | koşul-dışı | malzeme-dışı |
|---|---|---:|---:|---:|
| naif (geçiş no + eşik) | 1 | 0,776 | 0,749 | **0,693** |
| regresyon + eşik | parametre + süre | **0,885** | **0,844** | 0,576 |
| regresyon + eşik | sensör | 0,803 | 0,781 | 0,574 |
| regresyon + eşik | tümü | 0,830 | 0,755 | 0,583 |
| doğrudan sınıflandırıcı | sensör | 0,814 | 0,786 | 0,664 |
| doğrudan sınıflandırıcı | tümü | 0,814 | 0,773 | 0,692 |

**Regresyon + eşik, doğrudan sınıflandırıcıdan üstündür.** Muhtemel açıklama: sınıflandırıcı için 295 µm ile 5 µm aynı şeydir (ikisi de "sağlam"); eğitim sırasında aşınmanın ne kadar ilerlediği bilgisi atılmaktadır. Regresyon bu bilgiyi kullanmaktadır.

**Görülmemiş malzemede hiçbir model naif tabanı geçememektedir** (naif 0,693, en iyi model 0,692). Regresyon katmanında sensör içeren modeller naif tabanı hâlâ geçebiliyordu (Bölüm 5.2); çıktı ikili karara indirgendiğinde bu üstünlük kaybolmaktadır. Regresyondaki zayıflama, sınıflandırmada naif tabanın altına düşmeye dönüşmektedir.

### 5.5 Karar kuralı ve eşik ayarı

Varsayılan eşik (300 µm, aşınma sınırı) ayarlanmamış bir karar kuralıdır. Eşik, her katlamanın **içinde** ikinci bir çapraz doğrulama ile, yalnızca eğitim verisi kullanılarak seçilmiştir; test kümesi eşik seçimine hiç karışmamaktadır.

**Girdi kümesi: parametre + süre.** Bu bölüm karar katmanının kendisini ölçmektedir; model olarak Bölüm 5.2'nin bilinen koşullarda en iyi kümesi kullanılmıştır. Teslim edilen modelin (sensör + parametre + süre) eşik kalibrasyonu ayrıca yapılmıştır ve Bölüm 6.1'de verilmektedir (222 µm).

| Protokol | Kural | Kaçırılan | Yanlış alarm | Seçilen eşik | Maliyet |
|---|---|---:|---:|---:|---:|
| vaka-dışı | sabit (300) | 4 | 13 | 300,0 | 33 |
| vaka-dışı | **ayarlı** | **0** | 21 | 272,2 | **21** |
| koşul-dışı | sabit | 5 | 20 | 300,0 | 45 |
| koşul-dışı | **ayarlı** | **1** | 28 | 257,6 | **33** |
| malzeme-dışı | sabit | 23 | 39 | 300,0 | 154 |
| malzeme-dışı | **ayarlı** | 22 | 44 | 237,0 | 154 |

Eşik ayarı, kaçırılan aşınmayı vaka-dışı protokolde **sıfıra**, koşul-dışı protokolde **bire** indirmektedir. Seçilen eşiklerin üçü de sınırın altındadır (272 / 258 / 237) ve genelleme zorlaştıkça düşmektedir; bu, modelin sistematik olarak eksik tahmin ettiğinin ve karar kuralının bunu güvenli tarafa telafi ettiğinin göstergesidir.

**Görülmemiş malzemede karar kuralı hiçbir kazanç sağlamamaktadır** (maliyet 154 → 154). Eşiği düşürmek bir kaçırmayı kurtarmakta, karşılığında beş yanlış alarm getirmektedir; 5:1 maliyet oranında bu tam olarak başabaştır. Modelin o protokoldeki hatası, eşik oynatmakla telafi edilemeyecek kadar büyüktür: eşik mevcut bilgiyi kaçırma ile yanlış alarm arasında yeniden dağıtır, yeni bilgi üretmez. Bölüm 5.2 ve 5.4'te zayıflayarak ilerleyen sinyal, karar katmanında da güçlendirilememektedir.

**Ardışık onay (histerezis) işe yaramamıştır.** İç seçim her katlamada k = 1 vermiştir; tahminler zaten düzgün ilerlediği için gürültü bastırmaya ihtiyaç yoktur.

### 5.6 Derin öğrenme (1B-CNN + GRU)

Ham sinyalden öznitelik öğrenen bir mimari denenmiştir: 145 örnek × 6 kanal × 4500 zaman adımı, 120 epoch, küçük model ve yüksek düzenlileştirme.

**Tek tohumlu sonuca güvenilmemiştir.** Sinir ağlarında ağırlık başlangıcı ve örnek karıştırma sırası rastgeledir; küçük veride bu, model farkından büyük saçılım üretebilir. Deney üç tohumla tekrarlanmış ve şu kural kodlanmıştır:

> Tohumlar arası saçılım, iki model arasındaki farktan büyük veya eşitse sonuç **KARARSIZ**'dır; "kazandı" denemez.

| Protokol | GBM | CNN + GRU | Saçılım | Fark | Hüküm |
|---|---:|---:|---:|---:|---|
| vaka-dışı | 144,06 | **123,53** | ±2,12 | 20,53 | **GEÇTİ** |
| koşul-dışı | 159,74 | **137,79** | ±10,10 | 21,95 | **GEÇTİ** |
| malzeme-dışı | 257,65 | 252,26 | ±18,29 | 5,38 | **KARARSIZ** |

Tohum başına sonuçlar (µm):

| Protokol | Tohum 42 | Tohum 43 | Tohum 44 |
|---|---:|---:|---:|
| vaka-dışı | *kaydedilmemiştir* | | |
| koşul-dışı | 136,2 | 126,3 | 150,9 |
| malzeme-dışı | 249,9 | 231,1 | 275,8 |

Vaka-dışı protokolün tohum başına değerleri, tohum kaydı özelliği eklenmeden önceki bir çalıştırmadan geldiği için kayıtlı değildir; o satır için yalnızca ortalama ve saçılım (123,53 ± 2,12 µm) raporlanabilmektedir. Kayıtsız bir sayıyı rapora yazmamak, Bölüm 7'de tarif edilen ilkenin gereğidir.

Malzeme-dışı protokolde aynı model, aynı veri, yalnızca farklı başlangıç ağırlıkları ile sonuç GBM'in (257,65) iki yanına savrulmaktadır. O protokolde yalnızca 2 katlama bulunması bunun yapısal sebebidir.

**Ek gözlem: sonuçlar tohum sabitlense bile tam olarak yeniden üretilememektedir.** Aynı komut ikinci kez çalıştırıldığında koşul-dışı ortalaması 137,44'ten 137,79 µm'ye kaymıştır (malzeme-dışı birebir aynı kalmıştır). Sebep, CPU üzerinde çok iş parçacıklı kayan nokta toplama sırasının belirlenimci olmamasıdır; `torch.manual_seed` bunu kapsamaz. Kayma tohum saçılımının çok altındadır ve hükümleri değiştirmemektedir, ancak raporlanan derin öğrenme değerlerinin künyesi kayıtlı belirli bir çalıştırmaya aittir (`reports/model_deep_summary.provenance.json`). Gradyan artırma tarafında böyle bir kayma yoktur.

**Maliyet karşılaştırması:**

| | GBM | CNN + GRU |
|---|---:|---:|
| Eğitim süresi (koşul-dışı) | 6,7 s | 1102,1 s |
| Oran | — | **164×** |

Süreler makine yüküne bağlıdır ve çalıştırmadan çalıştırmaya değişir; mertebe (yüz kat üzeri) değişmemektedir.

**Karar: teslim edilen model GBM'dir.** Gerekçe: (a) derin modelin üstünlüğü teslim senaryosuna karşılık gelen protokolde ölçülememiştir; (b) MAE'deki kazanç alarm davranışına yansımamaktadır (overshoot 150,00 → 151,25); (c) 164 kat maliyet artışı bu kazanç karşılığında gerekçelendirilememektedir; (d) ağaç tabanlı model öznitelik önemleri üzerinden yorumlanabilirdir.

### 5.7 Sabit bölme denemesi — değerlendirme tasarımının sınanması

Bölüm 4.2'de bölmenin neden takım bazında ve çapraz doğrulamayla yapıldığı gerekçelendirilmişti. Bu bölüm o gerekçeyi **ölçüyor**: aynı veri üzerinde klasik eğitim/doğrulama/test bölmesi uygulanmış ve ne olduğu kaydedilmiştir. Amaç yeni bir model üretmek değildir; teslim edilen sistem bu denemeden etkilenmemiştir.

145 koşu, takım bazında bölündüğünde hedeflenen boyutları tam tutturmaktadır: eğitim 100 koşu (11 takım), doğrulama 20 koşu (takım 3 ve 5), test 25 koşu (takım 8 ve 11). Doğrulama kümesi iki iş yapmaktadır — erken durdurma ile ağaç sayısını, maliyet fonksiyonu ile alarm eşiğini belirlemek. Test kümesi yalnızca en sonda bir kez kullanılmıştır. Karşılaştırma için satır bazlı rastgele bölme de çalıştırılmıştır.

Aynı bölmede iki model çalıştırılmıştır: LightGBM ve Bölüm 5.6'daki 1B-CNN+GRU mimarisi. Derin model üç tohumla tekrarlanmıştır.

| Bölme | Model | Ağaç/epoch | Eşik (µm) | Test MAE | Test RMSE | Yakalama | Kaçırılan | Yanlış alarm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| takım bazlı | LightGBM | 10 | 402,0 | 164,33 | 192,24 | 0,545 | 5 | 0 |
| takım bazlı | **CNN+GRU** | 123 | **273,0** | **114,12** | **145,15** | **0,727** | **3** | 0 |
| rastgele | LightGBM | 133 | 207,0 | **114,94** | 166,32 | 1,000 | 0 | 7 |
| rastgele | CNN+GRU | 177 | 163,0 | 125,15 | 190,23 | 1,000 | 0 | 11 |

Derin modelin tohum başına test MAE'si (takım bazlı) 123,2 / 112,0 / 116,6 µm, ortalama ± saçılım **117,25 ± 4,58 µm**'dir. Saçılım, LightGBM ile arasındaki 50 µm'lik farkın çok altındadır; fark gürültüden kaynaklanmamaktadır.

**Rastgele bölme LightGBM'de %30 daha iyi görünmektedir ve sorun tam olarak budur.** O kurguda 15 takımın 14'ü hem eğitimde hem testtedir; aynı takımın komşu koşuları neredeyse aynı sinyali taşıdığı için model, test satırlarına çok benzeyen satırları eğitimde görmüş olmaktadır. Ölçülen şey genelleme değil ezberdir. Bu, Bölüm 4.2'de gerekçe olarak sunulan sızıntının niceliksel karşılığıdır.

**Asıl bulgu doğrulama kümesindedir.** Ağaç sayısı tarandığında doğrulama ve test hataları zıt yönlere gitmektedir:

| Ağaç | Doğrulama MAE | Test MAE |
|---:|---:|---:|
| 10 | **151,84** | 164,33 |
| 50 | 166,75 | 120,86 |
| 100 | 185,45 | 110,84 |
| 300 | 207,13 | 107,43 |
| 600 | 212,85 | **105,04** |

Erken durdurma doğrulama hatasını izlediği için 10 ağaç seçmiştir — test açısından mümkün olan en kötü seçim. 600 ağaçla test MAE'si 164 yerine 105 olacaktı. Sebep, doğrulama kümesinin yalnızca 2 takımdan (20 koşu) oluşması ve o takımların aşınma davranışının test takımlarını temsil etmemesidir.

Aynı yetersizlik eşik kalibrasyonunda da görülmektedir: seçilen eşik **402 µm**, yani aşınma sınırının (300 µm) **üstünde**. Bölüm 5.5'te çapraz doğrulamayla kalibre edilen eşikler sınırın altında çıkmıştı (272 / 258 / 237); model sistematik olarak eksik tahmin ettiği için doğru yön aşağıdır. 402 değeri doğrulama kümesinde optimaldir, ancak test takımlarında beş aşınmayı birden kaçırmaktadır.

Sınıflandırma tarafında her iki modelin de precision değeri 1,000'dir; bu bir başarı göstergesi değildir. Eşikler o kadar yüksektir ki modeller nadiren alarm vermekte, verdikleri az sayıda alarm doğru çıkmaktadır. Anlamlı olan recall'dır: LightGBM 11 aşınmış koşunun 5'ini, CNN+GRU 3'ünü kaçırmıştır (dengeli doğruluk 0,773 ve 0,864).

**CNN+GRU'nun öne geçmesi model üstünlüğü olarak okunmamalıdır.** Ağaç taraması, LightGBM'in 600 ağaçla 105,04 µm yaptığını — yani derin modeli de geçtiğini — göstermektedir. Fark modellerin kendisinden değil, aynı zayıf doğrulama kümesinin ikisini farklı derecede bozmasından kaynaklanmaktadır:

| Model | Doğrulamanın seçtiği | Sonuç |
|---|---|---|
| LightGBM | 10 ağaç | Felç; her girdiye 280–460 arası bir değer vermektedir |
| CNN+GRU | ~123 epoch (ortalama) | Makul; öğrenme tamamlanmıştır |

Aynı örüntü eşik kalibrasyonunda da görülmektedir: derin modelin eşiği 273 µm ile aşınma sınırının **altında** kalmakta (Bölüm 5.5'te beklenen doğru yön), LightGBM'inki 402 µm ile üstüne çıkmaktadır.

#### Test kümesinin satır satır dökümü

Toplu metrikler modelin **nerede** yanıldığını göstermemektedir. Aşağıdaki tablo aynı 25 test koşusunda iki modelin tahminlerini yan yana vermektedir (alarm eşikleri ayrı kalibre edildiği için farklıdır: LightGBM 402 µm, CNN+GRU 273 µm).

| takım | koşu | süre | gerçek | GBM tahmin | GBM | CNN tahmin | CNN |
|---:|---:|---:|---:|---:|---|---:|---|
| 8 | 1 | 0 | 0,0 | 278,4 | TN | 144,8 | TN |
| 8 | 2 | 3 | 180,0 | 278,4 | TN | 161,1 | TN |
| 8 | 3 | 9 | 300,0 | 291,0 | **FN** | 314,2 | **TP** |
| 8 | 5 | 18 | 440,0 | 292,8 | **FN** | 333,7 | **TP** |
| 8 | 6 | 30 | 620,0 | 378,2 | **FN** | 348,5 | **TP** |
| 11 | 1 | 1 | 0,0 | 269,9 | TN | 89,0 | TN |
| 11 | 5 | 40 | 80,0 | 296,7 | TN | 136,5 | TN |
| 11 | 9 | 105 | 160,0 | 315,7 | TN | 138,0 | TN |
| 11 | 13 | 273 | 260,0 | 338,1 | TN | 184,8 | TN |
| 11 | 15 | 330 | 310,0 | 352,7 | **FN** | 171,6 | **FN** |
| 11 | 16 | 393 | 370,0 | 340,8 | **FN** | 184,1 | **FN** |
| 11 | 18 | 465 | 420,0 | 426,0 | **TP** | 211,6 | **FN** |
| 11 | 19 | 545 | 470,0 | 457,8 | TP | 373,8 | TP |
| 11 | 21 | 724 | 650,0 | 415,0 | TP | 453,6 | TP |
| 11 | 23 | 929 | 760,0 | 354,8 | TP | 425,6 | TP |

*(Tablo 25 satırın temsili bir seçkisidir; tamamı `reports/holdout_detail.csv` içindedir.)*

Üç gözlem öne çıkmaktadır:

**Yeni takım tahmininde fark belirgindir.** Gerçek aşınmanın sıfır olduğu iki koşuda LightGBM 278,4 ve 269,9 µm demektedir; CNN+GRU 144,8 ve 89,0. LightGBM tüm girdilere dar bir bantta (280–460 µm) yanıt vermekte, yani ayırt etmemektedir.

**Takım 8'de yönler tamamen ayrışmaktadır.** LightGBM o takımın üç aşınmış koşusunun üçünü de kaçırmakta, CNN+GRU üçünü de yakalamaktadır.

**Ancak üstünlük tek yönlü değildir.** Takım 11'in 18. koşusunda gerçek aşınma 420 µm iken LightGBM 426 µm ile alarm vermekte (TP), CNN+GRU 211,6 µm ile kaçırmaktadır (FN). Toplamda LightGBM 6 TP / 5 FN, CNN+GRU 8 TP / 3 FN vermektedir.

**Sonuç:** bu veri ölçeğinde sabit bölme yalnızca daha gürültülü bir tahmin vermemekte, model seçimini ve eşik kalibrasyonunu aktif olarak yanlış yöne çekmektedir. Bozulmanın derecesi modele göre değişmekte, dolayısıyla bu kurguda ölçülen model karşılaştırmaları da güvenilir olmamaktadır. Çapraz doğrulama tercihi böylece varsayım olmaktan çıkıp ölçülmüş bir gerekçeye dayanmaktadır.

---

## 6. Teslim edilen sistem

### 6.1 Yapı

| Bileşen | Karar |
|---|---|
| Model | B-1, LightGBM, 145 satırın tamamıyla eğitilmiş |
| Girdi kümesi | sensör + parametre + süre (149 öznitelik) |
| Alarm eşiği | 222 µm (leave-one-tool-out iç ÇD ile kalibre) |
| Paket | `runs/model_b1/model_b1.joblib` + izlenen künye `reports/model_b1_package.json` |
| Çıkarım | `scripts/predict.py` — ham sinyalden VB + aşınmış/sağlam + kapsam durumu |

**Girdi kümesi seçiminin gerekçesi** Bölüm 5.2'deki ödünleşimdir. Parametre + süre kümesi bilinen koşullarda daha iyi olmasına rağmen seçilmemiştir; teslim edilen sistem eğitim verisinde bulunmayan bir malzemede çalışacağı için, o senaryoda naif tabanın altına düşen bir model tercih edilemez.

**Künye (provenance)** şunları içerir: öznitelik listesi ve sırası, kalibre edilmiş eşikler ve hangisinin hangi protokole ait olduğu, eğitim verisinin öznitelik başına referans istatistikleri, kapsam kümeleri, git commit hash'i, config dosyasının SHA-256 özeti, model dosyasının SHA-256 özeti, eğitim tarihi ve kütüphane sürümleri. Amaç, herhangi bir sonucu üreten kod durumunun kesin olarak belirlenebilmesidir.

### 6.2 Kapsam-dışı uyarısı

Sistem, eğitimde görülmemiş bir malzeme kodu veya kesme koşulu ile karşılaştığında tahmini yine üretir, ancak yanına açık bir uyarı basar. Bu, Bölüm 5.2 ve 5.4'teki bulgunun sistemin davranışına yansıtılmış halidir.

**Uyarı metadata'ya bakar, sinyale değil.** Sebebi ölçülmüştür: malzemenin sinyalden tahmin edilmesi denenmiş, dürüst protokolde (koşul bazında bölme) %71,7 doğruluk elde edilmiştir — çoğunluk tabanının (%67,6) yalnızca 4 puan üstünde. Yani sistem, kapsam dışında olduğunu sinyalden anlayamamaktadır ve kapsam kontrolü kullanıcının bildirdiği metadata'ya dayanmak zorundadır.

### 6.3 Reddedilen kalibrasyon: bir yan etki analizi

Eşik seçiminde başlangıçta "iki kalibrasyondan düşük olanı al" kuralı benimsenmişti. Malzeme kalibrasyonu 156 µm vermiş ve maliyet fonksiyonunu kazanmıştır (61 vs 98, sıfır kaçırma, hiç geç alarm yok). Buna rağmen reddedilmiştir:

| Eşik | Kaçırılan | Yanlış alarm | Ort. alarm konumu | İlk geçişte alarm |
|---:|---:|---:|---:|---:|
| 156 | 0 | 61 | ömrün %12,3'ü | **9/15 takım** |
| 190 | 4 | 58 | %14,7 | 8/15 |
| 222 | 9 | 53 | %16,0 | 7/15 |
| 237 | 9 | 50 | %17,7 | 6/15 |

156 µm eşiğinde 15 takımın 9'unda alarm **daha ilk geçişte** çalmaktadır; o takımlar için model hiçbir ölçüm yapmamakta, sinyal görmeden "değiştir" demektedir. Bu, "her takımı takar takmaz at" kuralına denktir ve bunun için modele gerek yoktur.

Sorunun kaynağı eşik seçimi değil, **maliyet fonksiyonunun kendisidir**: yanlış alarmı geçiş başına saymakta, dolayısıyla bir takımın ömrünün %88'ini atmanın bedelini yapısal olarak ölçememektedir. Optimizasyon bu boşluk yüzünden eşiği dibe itmektedir. Düzeltmesi eşiği elle seçmek değil, maliyet modeline takım bazında erken değiştirme terimi eklemektir; bu gelecek çalışma olarak not edilmiştir.

Aktif eşik, 15 katlamayla kalibre edildiği için daha güvenilir olan case kalibrasyonudur (222 µm). Bu bir uzlaşmadır ve raporlanmaktadır: görülmemiş malzeme için doğru eşik eldeki veriyle kalibre edilememektedir.

---

## 7. Doğrulama ve yol boyunca bulunan kusurlar

Çalışma boyunca sonuçları olduğundan iyi gösteren sekiz kusur tespit edilip düzeltilmiştir. Hepsi burada listelenmiştir; bu liste raporun bir eksiği değil, güvenilirlik iddiasının dayanağıdır.

**1. İmkânsız kabul kriteri.** Faz 00'da "alarm gecikmesi ≤ 5 geçiş" hedefi konulmuştu. Veri incelendiğinde bu hedefin etiket çözünürlüğü nedeniyle ulaşılamaz olduğu görülmüş, kriter yeniden tanımlanmıştır.

**2–3. Test kümesine bakarak seçim (iki kez).** Normalizasyon varyantlarının en iyisini seçmek ve B-2 hükmünü protokol başına en iyi ağırlıkla vermek, her ikisi de test bilgisini karar sürecine sızdırmaktadır. Fark edilip düzeltilmiş; raporda tüm varyant tablosu verilmekte, en iyisi seçilip sunulmamaktadır.

**4. Kendini doğrulayan çıktı metni.** Derin öğrenme betiği, sonuç ne olursa olsun ön yargıyı doğrulayan bir cümle basıyordu ("Beklenen sonuç. 145 örnek derin öğrenme için çok küçük..."), çünkü metin "3/3 kazanmadıysa" koşuluna bağlanmıştı. Kaldırılmıştır.

**5. İşaretli gecikme ortalaması.** Metrik özeti, 27–30 geçiş erken ve 47 geçiş geç alarmları birbirini götürecek şekilde ortalayarak −3,33 gibi yanıltıcı bir değer üretiyordu. Mutlak ortalama ve en kötü değer eklenmiştir.

**6. Alarm kilidinin katlama geneline uygulanması (iki yerde).** Alarm bir kez çaldığında sönmemesi tek bir takımın ömrü içinde doğrudur, ancak çapraz doğrulamada birden fazla takımın tahminleri ardışık birleştirildiğinde bir takımdaki erken alarm sonraki takımlara taşmaktaydı. Önce iç çapraz doğrulamada tespit edilip düzeltilmiş (eşik saçma biçimde 421 µm çıkmıştı), sonra aynı kusurun dış değerlendirmede de bulunduğu fark edilmiştir. Etkisi:

Etki, Bölüm 5.5'teki kurgunun aynısında ölçülmüştür (malzeme-dışı protokol, ayarlı eşik 237 µm, parametre + süre girdisi):

| Kilitleme | Kaçırılan | Yanlış alarm | Maliyet |
|---|---:|---:|---:|
| katlama geneli (hatalı) | 11 | 48 | 103 |
| takım bazında (doğru) | **22** | 44 | **154** |

Kaçırılan aşınma sayısı iki katına çıkmaktadır. Kusurun yönü tehlikelidir: kilit taştığında satırlar "alarm verildi" sayılmakta, dolayısıyla gerçekte kaçırılan aşınmalar sayılmamakta ve **sistem olduğundan güvenli görünmektedir**. Karar kuralının toplam kazancı da −%22,5'ten −%10,3'e inmiştir.

Vaka-dışı protokol etkilenmemiştir; o protokolde her katlama tek bir takımdan oluştuğu için taşacak ikinci takım yoktur. Seçilen eşikler de değişmemiştir (272,2 / 257,6 / 237,0), çünkü onlar iç çapraz doğrulamadan gelmekte ve orası ilk düzeltmeyle zaten onarılmıştı; değişen yalnızca raporlanan performanstır.

Her iki konum için regresyon testi yazılmıştır. İkinci kusurun hayatta kalma sebebi öğreticidir: ilk düzeltme için yazılan test kütüphane fonksiyonunu sınıyordu, betiğin *o fonksiyonu çağırdığını* değil. Betik yanlış olanı çağırmaya devam etti ve test yeşil kaldı. Yeni test betiği doğrudan çağırmaktadır.

**7. Rapor–kod ayrışması (iki yönlü).** Sonuç tabloları üretildikleri koda bağlanmadığı için zamanla ikisi sessizce ayrışmıştır. İki farklı yönde tespit edilmiştir:

- *Faz 04b:* Çarpıklık/basıklık için eklenen bir sayısal koruma (neredeyse sabit sinyalde bu ölçüler tanımsızdır) NASA'nın DC motor akımı kanalında 145 koşunun 23'ünde devreye girmekte, dolayısıyla sensör içeren tüm satırları etkilemekteydi. Rapordaki değerler koruma öncesine aitti; kayıtlı sonuç dosyası doğruydu.
- *Faz 04:* Model A betiği özniteliklerini "meta olmayan her sütun" kuralıyla seçiyordu. Sonraki bir fazda tabloya `cum_time` sütunu eklendiğinde Model A bunu sessizce yutmuştur (168 → 170 öznitelik). `cum_time` geçiş sayısının monoton fonksiyonu, yani naif tabanın girdisidir; Model A'nın "sensörler eğilimin ötesine ne katıyor?" sorusu bu sızıntıyla geçersizleşmekteydi. Burada rapor doğru, kod ayrışmıştı.
- *Faz 02:* Aynı "meta olmayan her sütun" kuralı keşif betiğinde de bulunmaktaydı ve aynı iki sütun korelasyon analizine de sızmıştı. Etkisi ham tabloda görülmektedir: en güçlü ham korelasyon 0,994 → 0,999, tutarlı yönlü öznitelik 140/168 → 142/170. Sızan sütunun bu analizin tam olarak **ayırmaya çalıştığı** eğilimin kendisi olması, kusuru özellikle ironik kılmaktadır. Rapordaki değerler yine doğruydu; düzeltmeden sonra birebir geri gelmişlerdir.

Faz 04 kusuru özellikle öğreticidir, çünkü **sonucu iyileştiriyordu** (MAE 19,91 → 19,18). Bir hata sonucu kötüleştirdiğinde hemen incelenir; iyileştirdiğinde fark edilmesi zordur. Sızıntı kusurlarının ortak imzası budur ve bu çalışmadan çıkan metodoloji kuralı şudur: **beklenenden iyi çıkan sonuç, bir şüphe sebebidir.**

Üç olayın ortak kökü, sonuç tablolarının üretildikleri kod durumuna bağlanmamış olmasıdır. İki karşı önlem alınmıştır: (a) sonuç üreten her betik artık çıktısının yanına künye yazmakta ve kaydedilen tablolar `git_hash` sütunu taşımakta; (b) `scripts/check_report_numbers.py` rapordaki tabloları ayrıştırıp kayıtlı sonuç dosyalarıyla karşılaştırmakta ve uyuşmazlıkta hata koduyla çıkmaktadır. İkincisi, kusuru bulan denetimin otomatik hale getirilmiş halidir.

**8. Gürültülü kalibrasyonu otomatik seçen eşik kuralı.** Bölüm 6.3'te ayrıntılandırılmıştır: "iki kalibrasyondan düşüğünü al" kuralı, yalnızca 2 katlamayla hesaplanan ve dolayısıyla en oynak olan kalibrasyonu sistematik olarak tercih etmektedir, çünkü gürültü hangi yöne saparsa sapsın aşağı sapan seçilmektedir.

**Doğrulanamayan sonuçlar.** Rapor için tutulan üç sayıdan ikisi betikle yeniden üretilebilmiş ve doğrulanmıştır; üçüncüsü (malzeme tahmini için kaydedilmiş %79,5 değeri) hiçbir protokolle yeniden üretilememiş ve orijinal deney kurgusu kayıtlı olmadığı için **rapordan çıkarılmıştır**. Yerine, protokolü açıkça tanımlanmış olan %71,7 değeri konulmuştur.

Aynı ilke raporun tamamına uygulanmıştır: kayıtlı bir sonuç dosyasına dayanmayan hiçbir sayı bırakılmamıştır. Bu gereklilik nedeniyle üç analiz sonradan betiklere taşınmıştır — eşik taramasındaki takım ömrü israfı sütunları, Faz 02 korelasyon özeti ve derin öğrenmede tohum başına MAE değerleri.

**Test kapsamı.** Proje 164 otomatik test içermektedir. Bunların bir kısmı yukarıdaki kusurlara karşı yazılmış regresyon testleridir; amaç aynı hatanın tekrar ortaya çıkmasını engellemektir. Buna ek olarak `scripts/check_report_numbers.py` ve `scripts/check_staj_raporu.py`, bu raporun ve depo belgelerinin sayılarını kayıtlı sonuç dosyalarıyla otomatik karşılaştırmaktadır.

**Otomatik denetimin sınırı.** Bu raporun bir taslağında, görülmemiş malzemede "üç katmanda da naif tabanın geçilemediği" ileri sürülmüştü. Sayıların hepsi doğruydu ve denetimden temiz geçiyordu; yanlış olan, o sayılardan çıkarılan yorumdu — regresyon katmanında sensör içeren modeller naif tabanı **geçmektedir**. Denetim bunu yakalayamazdı, çünkü yaptığı iş sayıların kaynağını doğrulamaktır, onlardan çıkarılan iddiaları değil. Sayı doğrulaması ile muhakeme doğrulaması ayrı işlerdir; ikincisi hâlâ okumayı gerektirmektedir.

---

## 8. Sınırlılıklar

**Veri ölçeği.** NASA Milling'de 145 etiketli koşu, koşul başına ~18 örnek bulunmaktadır. Malzeme-dışı protokolde yalnızca 2 katlama vardır; bu protokolden çıkan sonuçların hata payı geniştir ve derin öğrenme karşılaştırmasında bu doğrudan gözlenmiştir (±18,29 µm saçılım).

**Hedef malzeme kapsam dışıdır.** Kullanılan veri setlerinde alüminyum bulunmamaktadır. Sistemin hedef üretim ortamındaki performansı ölçülememiştir; ölçülen şey, benzer bir genelleme adımının (dökme demir → çelik) maliyetidir.

**Etiket değişkenliği.** PHM 2010'da üç ağız ayrı ölçülmekte ve ağızlar arası saçılım ortalama 12,87 µm (c6'da 20,5 µm) çıkmaktadır. Bu saçılım saf ölçüm gürültüsü değildir (ağızlar gerçekten farklı aşınır), ancak etiketin kendi değişkenliği bu mertebededir. Dolayısıyla bu değerin altındaki MAE farklarına dayanarak model sıralaması yapmak anlamlı olmayabilir.

**Nihai model performansı doğrudan ölçülemez.** Teslim edilen model 145 satırın tamamıyla eğitildiği için kendi eğitim verisi üzerinde ölçülemez. Raporlanan tüm performans değerleri çapraz doğrulama protokollerinden gelmektedir. Karşılaştırma için: aynı modelin örneklem içi MAE'si 4,81 µm, çapraz doğrulama MAE'si 138–271 µm'dir — yaklaşık 30 kat fark.

**Kesme parametreleri sabittir.** NASA'da iğ devri tüm koşularda 826 dev/dk'dır; devir etkisi ölçülememiştir.

---

## 9. Sonuç

Bu çalışmada CNC frezeleme için uçtan uca çalışan bir takım aşınması tahmin sistemi geliştirilmiş, teslim edilebilir bir paket ve çıkarım hattı olarak sonlandırılmıştır.

Niceliksel sonuçlar:

- Karar kuralı ayarlandığında kaçırılan aşınmış takım sayısı, aynı ayarlarla yeni bir takım senaryosunda (vaka-dışı) **sıfıra**, denenmemiş bir kesme koşulunda (koşul-dışı) **bire** inmektedir.
- Sensör verisi, geçiş sayısına dayalı naif tabanın ötesine sınırlı bilgi katmaktadır (eğilim çıkarılmış korelasyon tavanı 0,30); asıl kazanç doğrulukta değil, alarm zamanlamasının güvenli tarafa kaymasındadır.
- Derin öğrenme, bilinen koşullarda GBM'i tohum saçılımının üstünde bir farkla geçmektedir (%13,7), ancak 164 kat maliyetle ve teslim senaryosunda ölçülebilir üstünlük olmadan.

Metodolojik sonuçlar:

- Zorluk seviyesi arttıkça hata iki katına çıkmaktadır (144 → 160 → 258 µm). Tek bir kolay protokol raporlansaydı sistem iki kat daha iyi görünürdü.
- İki veri setinin birleştirilmesi denenmiş, ölçülmüş ve **reddedilmiştir**. Bu tür olumsuz sonuçların literatürde nadiren raporlanması, alandaki iyimserlik yanlılığının kaynaklarından biridir.
- Sonuç tablolarını üreten kod durumuna bağlamayan bir çalışma düzeni, zamanla rapor ile kod arasında sessiz ayrışma üretmektedir. Bu çalışmada iki yönde de gözlenmiş ve künye altyapısıyla kapatılmıştır.

En önemli sonuç olumsuzdur ve doğrudan uygulamayı ilgilendirir: **sistem, hiç görmediği bir malzemede güvenilir değildir ve bu durumu kendisi tespit edemez.**

Bu sonucun ağırlığı, tek bir ölçümden değil, sistemin **üç katmanında da izlenebilmesinden** gelmektedir:

| Katman | Görülmemiş malzemede sonuç |
|---|---|
| **Regresyon** (Bölüm 5.2) | Sensör içeren modeller naif tabanı geçmektedir (216,9–271,5 µm; naif taban 308,96 µm), ancak hata bilinen koşullara göre yaklaşık 1,7 kat artmaktadır (vaka-dışına göre 1,55× ve 1,96×). Parametre + süre kümesi naif tabanın **altına** düşmektedir (388,2). |
| **Sınıflandırma** (Bölüm 5.4) | Hiçbir model naif tabanı geçememektedir (naif 0,693; en iyi model 0,692). |
| **Karar kuralı** (Bölüm 5.5) | Eşik ayarı hiçbir kazanç sağlamamaktadır (maliyet 154 → 154). Bir kaçırma kurtarmak beş yanlış alarma mal olmakta; 5:1 oranında tam başabaş. |

Buradaki engel **mutlak değil, kademelidir** ve kademelenmenin yönü anlamlıdır. Regresyon katmanında sensör bilgisi hâlâ iş görmekte, model naif tabanın üstünde kalmakta ama belirgin biçimde zayıflamaktadır. Aynı çıktı ikili karara indirgendiğinde geriye kalan bilgi naif tabanın üstüne çıkmaya artık yetmemektedir. Karar eşiğini oynatmak da bu kaybı telafi etmemektedir; çünkü eşik yalnızca mevcut bilgiyi yeniden dağıtır, yeni bilgi üretmez.

Yani sorun bir model seçimi ya da eşik ayarı meselesi değildir: **görülmemiş malzemeye ilişkin bilgi sistemde zayıf biçimde vardır ve karara dönüşecek kadar güçlü değildir.** Alt katmanda zayıflayan sinyal üst katmanlarda güçlendirilememektedir.

Buna, sistemin kapsam dışında olduğunu sinyalden anlayamaması eklenmektedir (%71,7, çoğunluk tabanı %67,6): model yalnızca yanılmakla kalmamakta, yanıldığını da fark edememektedir. Kapsam kontrolünün metadata'ya dayandırılmasının sebebi budur.

Hedef üretim ortamında alüminyum işlendiği ve eldeki hiçbir veri seti alüminyum içermediği için, sistemin oraya doğrudan taşınması önerilmemektedir. Bölüm 10'daki ilk maddede tarif edilen kısmi ince ayar deneyi, bu engeli aşmanın ölçülebilir tek yoludur.

## 10. Gelecek çalışma

1. **Hedef malzemeden etiketli veri.** Sistemin yeni bir tezgâha kurulması için o tezgâhtan kaç etiketli takım gerektiğini ölçen kısmi ince ayar deneyi (2/4/6 takım ile ince ayar → kalan takımlarda ölçüm), pratik olarak en değerli sonraki adımdır.
2. **Maliyet modeline ömür israfı terimi.** Bölüm 6.3'te gösterildiği gibi, geçiş başına yanlış alarm sayımı erken değiştirmenin bedelini ölçememektedir. Takım bazında bir terim eklenmesi eşik seçimini fiziksel gerçeğe yaklaştıracaktır.
3. **Sensör azaltma çalışması.** Kanal alt kümesi tablosu (Bölüm 5.1) başlangıç niteliğindedir; sahada kurulabilir minimum sensör setinin belirlenmesi ayrı bir çalışma gerektirir.
4. **Alan uyarlama yöntemleri.** B-2'nin başarısızlığı basit örnek ağırlıklandırmanın yetersizliğini göstermektedir; alan uyarlama literatüründeki yöntemler bu problemi doğrudan hedeflemektedir.

---

## Ek A — Depo yapısı

```
config/default.yaml          tüm sabitler tek yerde
src/tcm/
  datasets/                  PHM 2010 ve NASA yükleyicileri
  features/                  zaman ve mertebe alanı öznitelik çıkarımı
  evaluation/                bölme protokolleri, metrikler, hüküm mantığı
  models/                    naif taban, GBM, CNN+GRU
  decision.py                eşik kalibrasyonu, alarm kuralı
  serving/                   model paketi, kapsam kontrolü
  provenance.py              git hash, config özeti, sürüm damgası
scripts/                     eğitim, değerlendirme ve çıkarım giriş noktaları
tests/                       164 test
reports/                     sonuç tabloları (künyeli)
```

## Ek B — Sonuçların yeniden üretilmesi

Her sonuç tablosu, üretildiği git commit hash'i ve komutla birlikte `README.md` içindeki provenance bölümünde listelenmiştir. Model paketi künyesi, teslim edilen modelin hangi kod durumundan üretildiğini SHA-256 özetleriyle sabitlemektedir.
