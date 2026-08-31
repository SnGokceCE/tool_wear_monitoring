# Sıradaki iş

Durum: 31 Ağustos 2026, Faz 06 oturumunun sonu.

Bu dosya bir sonraki oturumun nereden devam edeceğini söyler. **Önce "Bu
oturumda tamamlananlar" bölümünü okuyun** — orada listelenen işler yeniden
yapılmamalı.

---

## AÇIK — 1. Model B-1 sayılarındaki açıklanmamış kayma

**Öncelik: yüksek.** Rapora giren sayıları doğrudan etkiliyor.

`scripts/run_model_b1.py` 31 Ağustos'ta yeniden çalıştırıldı ve
`reports/model_b1_summary.csv` dosyasını **birebir** yeniden üretti (koşu
deterministik, tohum 42 sabit). Ama README'de o güne kadar duran sayılar
farklıydı:

| Girdi kümesi | Sınav | Eski README | Yeni koşu | Fark |
|---|---|---:|---:|---:|
| sadece sensör | vaka-dışı | 145,38 | 140,19 | 5,19 |
| sadece sensör | koşul-dışı | 173,26 | 166,57 | **6,69** |
| sadece sensör | malzeme-dışı | 221,34 | 216,85 | 4,49 |
| sensör + parametre | vaka-dışı | 144,04 | 139,42 | 4,62 |
| sensör + parametre | koşul-dışı | 159,52 | 163,99 | 4,47 |
| sensör + parametre | malzeme-dışı | 226,15 | 226,66 | 0,51 |
| sensör + par. + süre | vaka-dışı | 138,61 | 138,47 | 0,14 |
| sensör + par. + süre | koşul-dışı | 164,82 | 164,27 | 0,55 |
| sensör + par. + süre | malzeme-dışı | 271,91 | 271,49 | 0,42 |

`naif taban` ve `parametre + süre` satırları **değişmedi**. Bu, teşhisin en
güçlü ipucu: değişen satırların hepsi **sensör özniteliği içeren** satırlar.
Sensör içermeyen iki satır sabit kalmış.

### Yapılacak

1. **İki koşunun öznitelik sütun listelerini karşılaştır.** `run_model_b1.py`
   sensör sütunlarını `data[c].notna().any()` süzgeciyle veriden türetiyor
   ([`scripts/run_model_b1.py`](scripts/run_model_b1.py) içindeki
   `sensor_columns`). Sütun sayısı ya da hangi sütunların NaN olduğu
   değiştiyse model farklı girdi görmüş olur. README "144 öznitelik" diyor —
   yeni koşuda gerçekten 144 mü, kontrol et:

   ```bash
   python -c "import pandas as pd; from tcm.serving import resolve_feature_columns; d=pd.read_csv('data/processed/nasa_run_features.csv'); print(len(resolve_feature_columns(d,'sensor+param+time')))"
   ```

2. **Öznitelik tablosunun git geçmişini kontrol et.**
   `data/processed/nasa_run_features.csv` git dışında, ama onu üreten kod
   değil. `git log -p -- src/tcm/features/` ile 28–31 Ağustos arasında
   `timedomain.py` / `spectral.py` içinde bir değişiklik olup olmadığına bak.
   Özellikle `time_domain_features` içindeki `shape_defined` koruması (sabit
   sinyalde skew/kurtosis 0 döndürme) şüpheli: sensör özniteliklerini
   etkiler, parametre sütunlarını etkilemez — gözlenen desenle birebir uyuyor.

3. Sebep bulunduğunda:
   - README'deki **Faz 04b tablosunun üstündeki UYARI notunu kaldır**.
   - Aynı bölümdeki satır içi sayıları düzelt (aşağıda madde 3).
   - `run_model_b1.py`'ye künye (`tcm.provenance.run_stamp`) ekle ki bir daha
     olmasın — Faz 04/04b/04c/04d betiklerinin hiçbirinde künye yok.

**Not:** Faz 06 teslim modeli bu kaymadan **etkilenmiyor**. Paket, mevcut
`nasa_run_features.csv` ile eğitildi ve künyesi
`reports/model_b1_package.json` içinde kayıtlı (config sha256 + git hash +
joblib sha256). Etkilenen şey yalnızca README'nin Faz 04b tablosu.

---

## AÇIK — 2. README'nin satır içi sayıları

Madde 1'e bağlı, ondan sonra yapılmalı.

`README.md` Faz 04b bölümünde iki cümle eski koşunun sayılarını içeriyor:

- "**sensör modelini eziyor** — ve sensör eklemek onu bozuyor (108,38 →
  **138,61**)" → yeni koşuda 138,47
- "sensör modeli en iyisi oluyor (**221,34**)" → yeni koşuda 216,85

Ayrıca doğrulanamayan bir iddia var: "*VB ≤ 600 µm ile sınırlandığında
(126/145 koşu) sensör modelinin MAE'si 140,45 → 78,46'ya düşüyor*". Bu analiz
`run_model_b1.py` içinde **yok**; elle yapılmış ve kaydedilmemiş. Ya betiğe
eklenip yeniden üretilmeli ya da README'den çıkarılmalı.

---

## AÇIK — 3. Faz 05: `vaka-dışı` sınavı üç tohumla ölçülmedi

`koşul-dışı` ve `malzeme-dışı` üç tohumla ölçüldü
(`reports/model_deep_summary.csv`). `vaka-dışı` satırı hâlâ eski tek tohumlu
koşudan devralınmış durumda:

| alan | değer |
|---|---|
| `karar` | `saçılım ölçülmedi` |
| `mae_std` | boş (NaN) |
| `tohum_sayisi` | **boş (NaN)** — `1` değil |
| `git_hash` | `önceki çalıştırma (künyesiz)` |

`tohum_sayisi` kasıtlı olarak boş bırakıldı: eski koşuda tohum sayısı
**kaydedilmemişti**. `--seeds` varsayılanı 1 olduğu için muhtemelen 1'dir ama
bu bir çıkarım, ölçüm değil; künyeye çıkarım yazılmadı.

**Bu oturumda üç tohumlu `vaka-dışı` koşusu arka planda başlatıldı ama
bitmedi** (~1 saat sürüyor, 15 katlama × 3 tohum). Yeniden çalıştırmak için:

```bash
python scripts/run_model_deep.py --protocols "vaka-dışı" --seeds 3 --epochs 120 --save
```

`--save` diğer iki sınavın satırlarını korur (`_merge_with_existing`).
Bittiğinde README'ye Faz 05 bölümü yazılmalı — şu an **README'de Faz 05
bölümü yok**, sadece durum tablosunda "tamamlandı" yazıyor.

Üç tohumla ölçülen iki sınavın sonucu:

| Sınav | CNN+GRU | GBM | Fark | Saçılım | Hüküm |
|---|---:|---:|---:|---:|---|
| koşul-dışı | 137,44 | 159,74 | 22,30 | ±10,17 | **GEÇTİ** |
| malzeme-dışı | 252,26 | 257,65 | 5,38 | ±18,29 | **KARARSIZ** |

---

## Bu oturumda TAMAMLANANLAR — yeniden yapmayın

### Latching hatası (dış değerlendirme) — DÜZELTİLDİ

> Bu madde "yapılacaklar" listesinde duruyordu; bu oturumda **bitirildi**.
> Aşağıdaki dört kanıtın dördü de doğrulandı.

1. **Kod düzeltildi.** [`scripts/run_decision_rule.py:147`](scripts/run_decision_rule.py)
   artık `alarm_flags(predicted, threshold, k, test["case"].to_numpy())`
   çağırıyor; eskiden `apply_consecutive(predicted >= threshold, k)` idi ve
   kilidi katlamanın tamamına uyguluyordu.
2. **Regresyon testi yazıldı.** `tests/test_decision_rule_script.py` — 6 test,
   betiği `importlib` ile **doğrudan çağırıyor**. Önceki test yalnızca
   kütüphaneyi sınadığı için hatayı yakalayamamıştı.
3. **Faz 09 tablosu yeniden hesaplandı.** `reports/decision_rule_summary.csv`
   yenilendi; malzeme-dışı ayarlı eşik **22 kaçırılan / 44 yanlış alarm /
   maliyet 154** (eskiden 11 / 48 / 103). Toplam maliyet 158 → 208, karar
   kuralının kazancı −%22,5 → −%10,3. README hem Faz 09 bölümünde hem "Yol
   boyunca yanlış çıkan şeyler" bölümünde eski/yeni karşılaştırmasıyla
   güncellendi.
4. **Başka çağrı yeri kalmadı.** `grep` ile doğrulandı: gruplamasız
   `apply_consecutive` yalnızca `alarm_flags`'in kendi içinde çağrılıyor.

**Kalan tek iş:** yukarıdaki madde 3'teki `vaka-dışı` derin öğrenme koşusu
bittiğinde Faz 09 tablosunun etkilenmediğini teyit et (etkilenmemeli — farklı
dosya, farklı betik).

### `conservative` eşik kuralı — KALDIRILDI

Aktif eşik artık **`case` kalibrasyonu, 222 µm**. `material` kalibrasyonu
(156 µm) pakette ve künyede duruyor ama varsayılan değil.
`config/default.yaml` içindeki `serving.calibration` artık yalnızca `"case"`
ve `"material"` kabul ediyor. Gerekçe README'nin "Yol boyunca yanlış çıkan
şeyler" bölümünde, tam ölçüm tablosuyla.

**Gelecek çalışma olarak not düşüldü:** maliyet fonksiyonuna takım bazında
ömür israfı terimi eklenmeli. Şu anki fonksiyon yanlış alarmı **geçiş başına**
sayıyor; ömrünün başında atılan bir takım "1 yanlış alarm" görünüyor. Bu
eksiklik giderildiğinde `material` kalibrasyonu yeniden değerlendirilmeli.

### README bölümleri — provenance ve Faz 06 YAZILDI

- "Sonuçların kaynağı (provenance)" bölümü yazıldı: her README tablosunun
  hangi rapor dosyasından ve hangi komuttan geldiği listeli.
- "Faz 06 — sistemi ayağa kaldır" bölümü yazıldı.
- "Yol boyunca yanlış çıkan şeyler" bölümü yazıldı (4 madde).
- Durum tablosunda Faz 05 ve Faz 06 "tamamlandı" yapıldı.
- **Eksik olan tek bölüm: Faz 05 sonuçları** (yukarıda madde 3).

### Faz 06 altyapısı — TAMAM

`scripts/train_model.py`, `scripts/predict.py`, `scripts/threshold_sweep.py`,
`src/tcm/serving/`, `src/tcm/provenance.py`, `src/tcm/features/extract.py`,
`src/tcm/evaluation/verdict.py`. 164 test geçiyor.

Doğrulanmış olanlar:
- Öznitelik tablosu refaktörü çıktıyı değiştirmedi (145×154, sütun sırası
  dahil birebir aynı, en büyük sayısal fark 0).
- Eğitim ve çıkarım aynı öznitelik kodunu kullanıyor
  (`tests/test_serving.py::TestTrainingInferenceParity`).
- Paket kaydet/yükle turu tahminleri koruyor.
- Kapsam dışı uyarısı çalışıyor (görülmemiş malzeme ve koşul).
