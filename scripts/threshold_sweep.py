"""Alarm eşiği taraması - takasın şeklini görmek için (teşhis).

BU BETİK EŞİK SEÇMEZ. Eşik seçimi iç çapraz doğrulamada kalır
(``tcm.decision.calibrate_threshold``, Faz 09 mantığı). Buradaki tablo,
seçilen eşiğin etrafındaki takasın nasıl davrandığını göstermek içindir:
eşiği düşürdükçe kaçırılan aşınma azalır, yanlış alarm artar ve alarm daha
erken çalar. Bu üç eğrinin şekli, tek bir sayıdan daha çok şey anlatır.

Sonuçlara bakıp eşik seçmek sızıntıdır - tam olarak Faz 09'un kaçındığı şey.

ÖLÇÜ BİRİMİ NOTU
----------------
Zamanlama GEÇİŞ cinsinden veriliyor: alarm, gerçek aşınmanın sınırı aştığı
geçişten kaç geçiş önce/sonra çaldı.

  negatif -> erken alarm (güvenli, takım ömrü israfı)
  pozitif -> geç alarm   (tehlikeli, parça hurdaya gider)

Faz 04'te bu metriğin kesiciler arasında karşılaştırılamadığı tespit edilmişti
(eğime bağlı) ve yerine µm cinsinden overshoot kullanılmıştı. Burada yine
geçiş cinsinden veriliyor çünkü soru "operatör kaç parça önce uyarılır"
sorusu - ve o soru geçiş cinsindendir. µm karşılığı da tabloda yanında.

KİLİTLENME TAKIM BAZINDA
------------------------
Alarm bir kez çalınca sönmez (aşınma geri dönmez), ama bu kilit bir takımdan
diğerine TAŞMAMALIDIR. Geçiş sayısı da ancak tek bir takımın ömrü içinde
anlamlıdır. Bu yüzden her şey ``case`` bazında hesaplanır.

    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --thresholds 156,190,222,237 --save
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from tcm import load_config
from tcm.cli import setup_console
from tcm.models.gbm import make_gbm_small
from tcm.provenance import format_stamp, run_stamp
from tcm.serving import resolve_feature_columns

DEFAULT_THRESHOLDS = (156.0, 190.0, 222.0, 237.0)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--feature-set", default="sensor+param+time")
    parser.add_argument("--protocol", default="material",
                        help="gruplama sütunu (öntanımlı: material = malzeme-dışı)")
    parser.add_argument("--thresholds", default=None,
                        help="virgülle ayrılmış eşikler (µm)")
    parser.add_argument("--half-life-threshold", type=float, default=156.0,
                        help="ömrün ilk yarısı sayımının yapılacağı eşik")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    limit = float(config.get("nasa.wear_limit_um", 300))
    cost_missed = float(config.get("decision.cost_missed", 5.0))
    cost_false = float(config.get("decision.cost_false_alarm", 1.0))

    thresholds = (
        tuple(float(t) for t in args.thresholds.split(","))
        if args.thresholds else DEFAULT_THRESHOLDS
    )

    data = pd.read_csv(
        config.path("paths.data_processed") / "nasa_run_features.csv"
    )
    columns = resolve_feature_columns(data, args.feature_set)

    print("=" * 92)
    print("EŞİK TARAMASI - teşhis, SEÇİM DEĞİL")
    print("=" * 92)
    print(f"Model     : gradyan artırma, girdi = {args.feature_set} "
          f"({len(columns)} öznitelik)")
    print(f"Protokol  : {args.protocol}-dışı çapraz doğrulama")
    print(f"Aşınma sın: {limit:.0f} µm  |  maliyet {cost_missed:.0f}:{cost_false:.0f}")
    print(f"Eşikler   : {', '.join(f'{t:.0f}' for t in thresholds)} µm")

    predictions = _cross_validated_predictions(data, args.protocol, columns, seed)

    rows = [
        _evaluate(predictions, threshold, limit, cost_missed, cost_false)
        for threshold in thresholds
    ]
    table = pd.DataFrame(rows)

    _print_main_table(table, limit)
    _print_timing_table(table)
    _print_half_life(predictions, args.half_life_threshold, limit)
    _print_reading(table, limit)

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 92)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        out = target / "threshold_sweep.csv"
        table.assign(git_hash=stamp["git_hash"],
                     feature_set=args.feature_set,
                     protocol=args.protocol).to_csv(out, index=False)
        print(f"\nKaydedildi: {out}")

    return 0


def _cross_validated_predictions(
    data: pd.DataFrame, group_column: str, columns: list[str], seed: int
) -> pd.DataFrame:
    """Katlama dışı tahminler - her satır kendisini görmemiş modelden.

    Örneklem içi tahmin kullanılsaydı model neredeyse hatasız görünür
    (Faz 06'da ölçüldü: 4,8 µm) ve takas tablosu tamamen anlamsız çıkardı.
    """
    frames = []
    for held_out in sorted(data[group_column].unique()):
        train = data[data[group_column] != held_out]
        test = data[data[group_column] == held_out].sort_values(["case", "run"])

        model = make_gbm_small(random_state=seed)
        model.fit(train[columns], train["vb_um"])

        frames.append(pd.DataFrame({
            "case": test["case"].to_numpy(),
            "run": test["run"].to_numpy(),
            "fold": held_out,
            "vb_um": test["vb_um"].to_numpy(dtype=float),
            "vb_pred_um": np.asarray(model.predict(test[columns]), dtype=float),
        }))

    return pd.concat(frames, ignore_index=True).sort_values(["case", "run"])


def _tool_events(
    tool: pd.DataFrame, threshold: float, limit: float
) -> dict[str, object]:
    """Tek bir takımın ömründeki olaylar.

    Kilitlenme burada, takımın KENDİ içinde uygulanır - bu yüzden fonksiyon
    takım bazında çağrılır.
    """
    tool = tool.sort_values("run")
    truth = tool["vb_um"].to_numpy(dtype=float)
    pred = tool["vb_pred_um"].to_numpy(dtype=float)

    flags = np.maximum.accumulate(pred >= threshold)

    crossed = np.flatnonzero(truth >= limit)
    alarmed = np.flatnonzero(flags)

    true_idx = int(crossed[0]) if crossed.size else None
    alarm_idx = int(alarmed[0]) if alarmed.size else None

    worn = truth >= limit
    return {
        "case": float(tool["case"].iloc[0]),
        "n_runs": len(tool),
        "true_idx": true_idx,
        "alarm_idx": alarm_idx,
        "missed": int(np.sum(worn & ~flags)),
        "false_alarms": int(np.sum(~worn & flags)),
        # Alarm anındaki GERÇEK aşınmanın sınırı ne kadar aştığı (µm).
        "overshoot_um": (
            float(truth[alarm_idx] - limit) if alarm_idx is not None
            else float(truth[-1] - limit) if true_idx is not None else np.nan
        ),
        # Geçiş cinsinden gecikme; ikisi de tanımlıysa.
        "delay_cuts": (
            float(alarm_idx - true_idx)
            if (alarm_idx is not None and true_idx is not None) else np.nan
        ),
    }


def _evaluate(
    predictions: pd.DataFrame,
    threshold: float,
    limit: float,
    cost_missed: float,
    cost_false: float,
) -> dict[str, object]:
    events = pd.DataFrame([
        _tool_events(tool, threshold, limit)
        for _, tool in predictions.groupby("case")
    ])

    delays = events["delay_cuts"].dropna()
    early = delays[delays < 0]
    late = delays[delays > 0]

    missed = int(events["missed"].sum())
    false_alarms = int(events["false_alarms"].sum())

    return {
        "esik_um": threshold,
        "missed_worn": missed,
        "false_alarms": false_alarms,
        "maliyet": cost_missed * missed + cost_false * false_alarms,
        "alarm_veren_takim": int(events["alarm_idx"].notna().sum()),
        "n_takim": len(events),
        "gecikme_ort": float(delays.mean()) if len(delays) else np.nan,
        "gecikme_ortanca": float(delays.median()) if len(delays) else np.nan,
        "en_erken": float(delays.min()) if len(delays) else np.nan,
        "en_gec": float(delays.max()) if len(delays) else np.nan,
        "erken_takim": int(len(early)),
        "gec_takim": int(len(late)),
        "tam_zamanli": int((delays == 0).sum()),
        "overshoot_ort_um": float(events["overshoot_um"].mean()),
        "en_kotu_overshoot_um": float(events["overshoot_um"].max()),
    }


def _print_main_table(table: pd.DataFrame, limit: float) -> None:
    print("\n" + "=" * 92)
    print(f"KAÇIRILAN AŞINMA / YANLIŞ ALARM  (gerçek sınır {limit:.0f} µm)")
    print("=" * 92)
    view = table[[
        "esik_um", "missed_worn", "false_alarms", "maliyet",
        "alarm_veren_takim", "n_takim",
    ]].rename(columns={
        "esik_um": "eşik µm", "missed_worn": "kaçırılan",
        "false_alarms": "yanlış alarm", "maliyet": "maliyet",
        "alarm_veren_takim": "alarmlı takım", "n_takim": "takım",
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:8.0f}"))


def _print_timing_table(table: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("ALARM ZAMANLAMASI - geçiş cinsinden (negatif = erken, pozitif = geç)")
    print("=" * 92)
    view = table[[
        "esik_um", "gecikme_ort", "gecikme_ortanca", "en_erken", "en_gec",
        "erken_takim", "tam_zamanli", "gec_takim",
        "overshoot_ort_um", "en_kotu_overshoot_um",
    ]].rename(columns={
        "esik_um": "eşik µm", "gecikme_ort": "ort geçiş",
        "gecikme_ortanca": "ortanca", "en_erken": "en erken",
        "en_gec": "en geç", "erken_takim": "erken",
        "tam_zamanli": "tam", "gec_takim": "geç",
        "overshoot_ort_um": "ort overshoot µm",
        "en_kotu_overshoot_um": "en kötü µm",
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
    print("\n  'erken/tam/geç' sütunları TAKIM sayısıdır (eşiği gerçekten aşan")
    print("  takımlar arasında). Gecikme yalnızca hem alarmın hem gerçek")
    print("  geçişin olduğu takımlarda tanımlıdır.")


def _print_half_life(
    predictions: pd.DataFrame, threshold: float, limit: float
) -> None:
    print("\n" + "=" * 92)
    print(f"{threshold:.0f} µm EŞİĞİNDE ÖMRÜN İLK YARISINDA ÇALAN ALARMLAR")
    print("=" * 92)

    rows = []
    for _, tool in predictions.groupby("case"):
        event = _tool_events(tool, threshold, limit)
        if event["alarm_idx"] is None:
            position = np.nan
            first_half = False
        else:
            # 0-indeksli konum / toplam geçiş: 0.0 = ilk geçiş, 1.0 = son.
            position = event["alarm_idx"] / event["n_runs"]
            first_half = position < 0.5
        rows.append({**event, "konum": position, "ilk_yarida": first_half})

    events = pd.DataFrame(rows)
    count = int(events["ilk_yarida"].sum())
    alarmed = int(events["alarm_idx"].notna().sum())

    view = events[["case", "n_runs", "alarm_idx", "true_idx", "konum",
                   "ilk_yarida", "delay_cuts"]].rename(columns={
        "case": "takım", "n_runs": "geçiş", "alarm_idx": "alarm@",
        "true_idx": "gerçek@", "konum": "ömür oranı",
        "ilk_yarida": "ilk yarı", "delay_cuts": "gecikme",
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    print(f"\n  SAYIM: {count} / {len(events)} takımda alarm ömrün İLK YARISINDA çaldı.")
    print(f"  ({alarmed} takımda alarm hiç çaldı; {len(events) - alarmed} takımda hiç çalmadı.)")
    print("\n  Bu sayı takım ömrü israfının doğrudan ölçüsü: alarm ömrün ilk")
    print("  yarısında çalıyorsa takım kullanılabilir ömrünün yarısından fazlasını")
    print("  kullanmadan değiştirilir.")


def _print_reading(table: pd.DataFrame, limit: float) -> None:
    print("\n" + "=" * 92)
    print("OKUMA")
    print("=" * 92)

    low = table.iloc[0]
    high = table.iloc[-1]

    print(
        f"\nEşik {low['esik_um']:.0f} -> {high['esik_um']:.0f} µm arasında:\n"
        f"  kaçırılan aşınma  {low['missed_worn']:.0f} -> {high['missed_worn']:.0f}\n"
        f"  yanlış alarm      {low['false_alarms']:.0f} -> {high['false_alarms']:.0f}\n"
        f"  ortalama gecikme  {low['gecikme_ort']:+.1f} -> {high['gecikme_ort']:+.1f} geçiş"
    )
    print(
        "\nBU TABLO EŞİK SEÇMEK İÇİN KULLANILMAMALIDIR. Sayılar test\n"
        "katlamalarından geliyor; en iyi görünen eşiği seçmek test bilgisini\n"
        "karara sızdırır ve raporlanan başarıyı gerçek dışı yükseltir.\n"
        "Eşik iç çapraz doğrulamada seçilir (scripts/train_model.py)."
    )


if __name__ == "__main__":
    sys.exit(main())
