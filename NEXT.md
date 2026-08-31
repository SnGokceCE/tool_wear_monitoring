# Sıradaki iş

Durum: 31 Ağustos 2026. **Faz 06 ve doğrulama denetimi kapandı.**
Sıradaki iş **rapor (Faz 11)**.

README'de artık `(doğrulanmamış)` etiketli sayı yok; her tablo bir rapor
dosyasına ve bir git hash'ine bağlı.

---

## AÇIK — Kod temizliği (rapordan sonra da yapılabilir)

Hiçbiri sonucu etkilemiyor; sadece bakım borcu.

- `src/tcm/decision.py` — `_flags_by_group = alarm_flags` geriye dönük takma
  adı artık hiçbir yerden çağrılmıyor, kaldırılabilir.
- `src/tcm/datasets/nasa.py` — `ensure_not_used_for_training()` anlamsız
  kaldı: NASA artık eğitimde kullanılıyor (`use_for_training: true`) ve
  fonksiyon hiçbir yerden çağrılmıyor.
- Faz 04/04b/04c/04d betikleri `META_COLUMNS` / `PARAMETER_COLUMNS` /
  `TIME_COLUMN` sabitlerini ayrı ayrı tanımlıyor. `tcm.serving` içindeki
  `resolve_feature_columns` bunları zaten sağlıyor. **Dikkat:** Model A'nın
  `META_COLUMNS`'ı bilinçli olarak `run_time`/`cum_time` içeriyor (aşağıdaki
  hataya bakın); birleştirme yapılırsa bu ayrım korunmalı.

## AÇIK — Faz 07, 08, 10 hiç yapılmadı

Yol haritasında var, teslim için zorunlu değil:

- **Faz 07** çapraz veri seti genelleme sınavı — Faz 04c'de PHM birleştirme
  reddedildiği için kısmen cevaplanmış sayılabilir.
- **Faz 08** sensör azaltma çalışması — Faz 04'teki kanal alt kümesi tablosu
  bunun ön çalışması.
- **Faz 10** sağlamlaştırma ve paketleme — Faz 06 paketi bunun çoğunu
  karşılıyor.

---

## Bu oturumda TAMAMLANANLAR

### Künye denetimi — iki tabloda kayma bulundu, ikisi de düzeltildi

Faz 04–04d betikleri yeniden çalıştırılıp kayıtlı tablolarla karşılaştırıldı.

**Faz 04 · Model A — GERÇEK HATA.** `run_model_a.py` özniteliklerini "meta
olmayan her sütun" diye seçiyordu. Faz 04c'de PHM tablosuna `run_time` ve
`cum_time` eklenince Model A onları sessizce yuttu (168 → 170 öznitelik).
`cum_time` geçiş sayısının monoton fonksiyonu, yani naif tabanın girdisi —
Model A'nın "sensörler eğilimin ötesine ne katıyor" sorusu geçersizleşiyordu.
Hata MAE'yi **iyileştirdiği** için fark edilmesi zordu (19,91 → 19,18).
İki sütun `META_COLUMNS`'a eklendi; tablo README değerlerine birebir döndü.

**Faz 04b · Model B-1 — kayma.** README dokuz hücrede `skew`/`kurtosis`
koruması öncesi değerleri taşıyordu (`smcDC` kanalı, 23/145 satır). Sebep
teşhis edildi, tablo CSV'ye eşitlendi.

**Faz 04c ve 04d — temiz.** Sayılar değişmedi.

Dört betiğin dördüne de künye eklendi (`tcm.provenance`); kaydedilen tablolar
artık `git_hash` sütunu taşıyor.

### Üç doğrulanamayan sayı — ikisi üretildi, biri reddedildi

| İddia | Sonuç |
|---|---|
| Parametreler tek başına MAE 205 µm | **Üretildi:** 206,20 (vaka-dışı) / 202,86 (koşul-dışı). Ana tabloya `5 · sadece parametre (süresiz)` satırı olarak eklendi. |
| VB ≤ 600 → MAE 140,45 → 78,46 | **Üretildi:** 126/145 koşu, 140,19 → 78,44. Üç sınav için de betiğe alındı. |
| Sensörden malzeme tahmini %79,5 | **ÜRETİLEMEDİ.** Takım bazında 0,890, koşul bazında 0,717 — hiçbiri 0,795 değil. Protokolü kayıtlı olmadığı için hangi kurgudan geldiği bilinmiyor. Sayı README'den çıkarıldı. |

Malzeme tahmini analizi doğru protokolle yeniden yazıldı ve **bulgunun yönü
değişti**: koşul bazında (dürüst sınav) doğruluk 0,717, çoğunluk tabanının
yalnızca 4 puan üstünde. Takım bazındaki 0,890 iyimser, çünkü `condition`
kimliği malzemeyi zaten içeriyor ve kardeş takımlar eğitimde kalıyor.

Bu, alüminyum uyarısını **güçlendiriyor**: sistem hiç görmediği bir malzemeyi
sinyalden tanıyıp kendini uyaramaz. Yeni çıktı: `reports/model_b1_extras.csv`.

### Daha önce kapatılanlar

- Latching hatası (dış değerlendirme) düzeltildi + regresyon testi
  (`tests/test_decision_rule_script.py`), Faz 09 tablosu yeniden hesaplandı.
- `conservative` eşik kuralı kaldırıldı; aktif eşik `case` kalibrasyonu
  (222 µm). Gelecek çalışma: maliyet fonksiyonuna takım bazında ömür israfı
  terimi.
- Faz 05 üç sınavda da üç tohumla ölçüldü, README bölümü yazıldı.
- Faz 06 teslim paketi, çıkarım hattı, kapsam uyarısı, künye.

164 test geçiyor.
