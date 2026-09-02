# Sıradaki iş

Durum: 1 Eylül 2026. **Faz 06, doğrulama denetimi ve rapor sayı denetimi kapandı.**

Rapor (`staj_raporu.md`) ve `README.md`, kayıtlı sonuç dosyalarıyla makineyle
doğrulanmış durumda. Kayıtsız sayı kalmadı.

```bash
python scripts/check_report_numbers.py   # README  - 198 sayı
python scripts/check_staj_raporu.py      # rapor   - 181 sayı
```

**Faz 12** (sabit 100/20/25 bölme deneyi) README'ye ve rapora (§5.7)
eklendi, ikisi de denetim kapsamında. Deney LightGBM ve CNN+GRU ile
ayrı ayrı çalıştırıldı. 175 test. Teslim edilen sistemi değiştirmez; çapraz doğrulama
tercihinin deneysel gerekçesidir.

---

## AÇIK — Kod temizliği

Hiçbiri sonucu etkilemiyor.

- `src/tcm/decision.py` — `_flags_by_group = alarm_flags` takma adı artık
  kullanılmıyor.
- `src/tcm/datasets/nasa.py` — `ensure_not_used_for_training()` anlamsız
  kaldı; NASA eğitimde kullanılıyor ve fonksiyon çağrılmıyor.
- Betikler `META_COLUMNS` / `PARAMETER_COLUMNS` / `TIME_COLUMN` sabitlerini
  ayrı ayrı tanımlıyor. **Dikkat:** Model A ve `explore_phm.py` bilinçli
  olarak `run_time`/`cum_time` dışlıyor (aşağıdaki kusura bakın); birleştirme
  yapılırsa bu ayrım korunmalı.
- `check_report_numbers.py` ile `check_staj_raporu.py` ayrı belgeler için
  ayrı tutuldu; ortak bir etiket profili mekanizmasıyla birleştirilebilirler.

## AÇIK — Faz 07, 08, 10

Yol haritasında var, teslim için zorunlu değil. Gerekçeleri README'de.

## AÇIK — Derin öğrenme yeniden üretilebilirliği

Aynı komut ikinci kez çalıştırıldığında koşul-dışı ortalaması 137,44'ten
137,79 µm'ye kaydı (malzeme-dışı birebir aynı kaldı). Sebep CPU'da çok iş
parçacıklı kayan nokta toplama sırasının belirlenimci olmaması; kayma tohum
saçılımının çok altında ve hükümleri değiştirmiyor. Tam belirlenimcilik
isteniyorsa `torch.use_deterministic_algorithms(True)` ve tek iş parçacığı
denenebilir — maliyeti yavaşlamadır.

---

## Bu oturumlarda TAMAMLANANLAR

### Rapor sayı denetimi

Raporun tamamı CSV'lerle karşılaştırıldı; **13 hücre bayat çıktı ve
düzeltildi**: Faz 09 tablosunun üç satırı (latching düzeltmesi öncesi),
Model B-1'in bir hücresi (shape guard öncesi), derin öğrenme koşul-dışı
satırı ve maliyet tablosu. Ayrıca §7'deki latching tablosu §5.5 ile aynı
kurguya çevrildi ve "kaçırılan aşınma sıfıra iniyor" cümlesi düzeltildi
(vaka-dışında sıfır, koşul-dışında bir).

### Kayıtsız sayı bırakılmadı

- Faz 02 korelasyon özeti → `reports/correlation_summary.csv`
  (`explore_phm.py --save`)
- Eşik taramasında takım ömrü israfı sütunları → `threshold_sweep.csv`
- Derin öğrenmede tohum başına MAE'ler → `model_deep_summary.csv`
- Yeniden üretilemeyen tek sayı (malzeme tahmini %79,5) rapordan çıkarıldı;
  yerine protokolü tanımlı %71,7 kondu.

### Aynı kusurun üç örneği bulundu ve düzeltildi

"Meta olmayan her sütun özniteliktir" kuralı, tabloya sonradan eklenen
`run_time`/`cum_time` sütunlarını üç yerde sessizce yuttu:

| Yer | Etki |
|---|---|
| Model A (`run_model_a.py`) | 168 → 170 öznitelik, MAE 19,91 → 19,18 |
| Keşif (`explore_phm.py`) | ham korelasyon 0,994 → 0,999, 140/168 → 142/170 |
| (Faz 04b'deki kayma farklı kökenli: `skew`/`kurtosis` koruması) |  |

Üçünde de **rapor doğruydu, kod ayrışmıştı**. Künye altyapısı ve iki denetim
betiği bu boşluğu kapattı.

### Daha önce kapatılanlar

Latching hatası (iç + dış) + regresyon testleri, `conservative` eşik kuralının
kaldırılması (aktif eşik `case`, 222 µm), Faz 05'in üç sınavda üç tohumla
ölçülmesi, Faz 06 teslim paketi ve çıkarım hattı.

164 test geçiyor.
