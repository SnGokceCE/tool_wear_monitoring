# Sıradaki iş

Durum: 31 Ağustos 2026. **Faz 06 kapandı.** Sıradaki iş rapor (Faz 11).

---

## AÇIK — 1. Faz 04, 04c ve 04d tabloları künye kontrolünden geçmedi

**Öncelik: yüksek. Rapora geçmeden önce yapılmalı.**

Faz 04b tablosunda dokuz hücrenin eski koddan kaldığı bulundu ve düzeltildi
(aşağıda, "tamamlananlar"). Aynı risk denetlenmemiş üç bölümde duruyor:

| Bölüm | Rapor dosyası | Betik | Künye |
|---|---|---|---|
| Faz 04 · Model A | `reports/model_a_*.csv` | `run_model_a.py` | **yok** |
| Faz 04c · Model B-2 | `reports/model_b2_summary.csv` | `run_model_b2.py` | **yok** |
| Faz 04d · sınıflandırma | `reports/classification_summary.csv` | `run_classification.py` | **yok** |

Faz 04b'deki kayma, `skew`/`kurtosis` korumasının `cd9912b`'de eklenmesinden
kaynaklanıyordu. **Faz 04 (Model A) o commit'ten öncedir** (`7e833be`,
28 Ağu 11:32) — yani README'deki Model A tablosu büyük olasılıkla koruma
öncesi sayılar içeriyor. Model A PHM verisini kullanıyor; korumanın PHM
kanallarında da devreye girip girmediği kontrol edilmeli (NASA'da yalnızca
`smcDC` kanalında, 23/145 satırda tetikleniyordu).

### Yapılacak

1. Üç betiği `--save` ile yeniden çalıştır.
2. Kayıtlı CSV'lerle README tablolarını karşılaştır.
3. Fark varsa: README'yi CSV'ye eşitle ve Faz 04b'dekine benzer bir düzeltme
   notu yaz.
4. Üç betiğe künye ekle (`tcm.provenance.run_stamp` / `format_stamp`) —
   `run_model_deep.py` ve `train_model.py` örnek alınabilir.

Hızlı kontrol:

```bash
python scripts/run_model_a.py --save
python scripts/run_model_b2.py --save
python scripts/run_classification.py --save
git diff --stat reports/
```

`git diff` boş çıkarsa tablolar zaten güncel demektir.

---

## AÇIK — 2. Doğrulanmamış üç sayı README'de duruyor

Faz 04b "Yan gözlemler" bölümünde `(doğrulanmamış)` diye işaretlendiler.
Elle yapılmış, betiğe girmemiş analizlerden geliyorlar ve koruma
düzeltmesinden önce üretildikleri için sayıları da güncel olmayabilir:

- VB ≤ 600 µm ile sınırlandığında sensör modeli MAE 140,45 → 78,46
- Sensörden malzeme tahmini %79,5 doğruluk (taban %67,6)
- Kesme parametreleri **tek başına** MAE 205 µm

Her biri ya `run_model_b1.py` içine alınıp yeniden üretilmeli ya da README'den
çıkarılmalı. Üçüncüsü kolay: tabloya bir `sadece parametre` satırı eklemek
yeterli.

---

## AÇIK — 3. Kod temizliği (sona bırakıldı)

- `src/tcm/decision.py` içindeki `_flags_by_group = alarm_flags` geriye dönük
  takma adı artık kullanılmıyor; kaldırılabilir.
- `src/tcm/datasets/nasa.py` içindeki `ensure_not_used_for_training()` artık
  anlamsız — NASA eğitimde kullanılıyor (`use_for_training: true`) ve fonksiyon
  hiçbir yerden çağrılmıyor.
- Faz 04/04b/04c/04d betikleri `META_COLUMNS` / `PARAMETER_COLUMNS` /
  `TIME_COLUMN` sabitlerini tekrar tekrar tanımlıyor;
  `tcm.serving.resolve_feature_columns` bunları zaten sağlıyor.

---

## Bu oturumda TAMAMLANANLAR — yeniden yapmayın

### Model B-1 sayı kayması — TEŞHİS EDİLDİ VE DÜZELTİLDİ

README'deki dokuz hücre kayıtlı CSV ile uyuşmuyordu (0,14–6,69 µm fark,
yalnızca sensör içeren satırlarda).

**Sebep:** `skew`/`kurtosis` koruması `cd9912b` (Faz 04b) commit'inde eklendi.
Neredeyse sabit sinyalde bu iki ölçü tanımsızdır; SciPy güvenilmez sayı
döndürür. NASA'nın `smcDC` (DC iğ motor akımı) kanalı 145 koşunun 23'ünde bu
durumda — 2 öznitelik × 23 satır = 46 hücre. Koruma yalnızca sensör
özniteliklerine dokunduğu için `naif taban` ve `parametre + süre` satırları
değişmemişti; desen buydu.

**Doğrulama:** koruma koddan geçici olarak çıkarılıp öznitelik tablosu yeniden
üretildi ve B-1 çapraz doğrulaması tekrarlandı — eski dokuz değerin **dokuzu
da virgülden sonrasına kadar** yeniden elde edildi.

**Karar:** korumalı (mevcut CSV) sayılar doğru. README tablosu ve satır içi
metinler eşitlendi, düzeltme notu yazıldı.

### Faz 05 — üç sınav da üç tohumla ölçüldü, README bölümü yazıldı

| Sınav | CNN+GRU | GBM | Fark | Saçılım | Hüküm |
|---|---:|---:|---:|---:|---|
| vaka-dışı | 123,53 | 144,06 | 20,53 | ±2,12 | GEÇTİ |
| koşul-dışı | 137,44 | 159,74 | 22,30 | ±10,17 | GEÇTİ |
| malzeme-dışı | 252,26 | 257,65 | 5,38 | ±18,29 | KARARSIZ |

Künye temiz (git `74d4de8`, `git_dirty: false`). Teslim modeli yine de B-1;
gerekçe README Faz 05 bölümünde.

### Latching hatası (dış değerlendirme) — DÜZELTİLDİ

`run_decision_rule.py` artık `alarm_flags(..., groups=case)` çağırıyor.
`tests/test_decision_rule_script.py` betiği doğrudan çağıran 6 regresyon
testiyle eklendi. Faz 09 tablosu yeniden hesaplandı: malzeme-dışı ayarlı eşik
**22 kaçırılan / 44 yanlış alarm / maliyet 154** (eskiden 11/48/103), toplam
maliyet 158 → 208. README'nin hem Faz 09 hem "Yol boyunca yanlış çıkan şeyler"
bölümü güncel.

### `conservative` eşik kuralı — KALDIRILDI

Aktif eşik **`case` kalibrasyonu, 222 µm**. `material` (156 µm) pakette ve
künyede duruyor ama varsayılan değil. Gerekçe ve ölçüm tablosu README'de.
Gelecek çalışma notu: maliyet fonksiyonuna takım bazında ömür israfı terimi.

### README — provenance, Faz 05, Faz 06, hatalar bölümleri YAZILDI

Durum tablosu güncel. Eksik bölüm kalmadı; açık olan tek şey yukarıdaki
1. ve 2. maddeler (denetlenmemiş tablolar ve doğrulanmamış üç sayı).
