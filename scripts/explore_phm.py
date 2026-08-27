"""Faz 02 - PHM 2010 keşifsel analizi.

Model kurmadan önce verinin nasıl davrandığını görmek için. Şekilleri
``reports/figures`` altına yazar, bulguları ekrana basar.

    python scripts/explore_phm.py
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # başsız çalışma - pencere açmaya çalışmasın

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tcm import load_config
from tcm.cli import setup_console
from tcm.datasets import PHM2010
from tcm.features import spindle_frequency_hz, tooth_passing_frequency_hz, welch_spectrum
from tcm.features.build import load_or_build

CUTTER_COLOURS = {"c1": "#0D6E70", "c4": "#9E5410", "c6": "#8E2B26"}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true", help="öznitelikleri yeniden üret")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    figures = config.path("paths.reports") / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fs = float(config.get("phm2010.sampling_rate_hz"))
    rpm = float(config.get("phm2010.spindle_rpm"))
    flutes = int(config.get("phm2010.flutes"))
    limit = float(config.get("evaluation.wear_limit_um"))

    dataset = PHM2010(
        config.path("phm2010.root"),
        wear_aggregation=config.get("phm2010.wear_aggregation", "max"),
    )
    cutters = dataset.labelled_cutters()
    wear = {c: dataset.wear(c) for c in cutters}

    features = load_or_build(
        config.path("paths.data_processed") / "phm_cut_features.csv",
        dataset,
        cutters,
        sampling_rate_hz=fs,
        rpm=rpm,
        max_order=int(config.get("transfer.max_order", 8)),
        rebuild=args.rebuild,
    )

    print(f"\nİğ frekansı      : {spindle_frequency_hz(rpm):.1f} Hz  (mertebe 1)")
    print(f"Diş geçişi       : {tooth_passing_frequency_hz(rpm, flutes):.1f} Hz  (mertebe {flutes})")

    _plot_wear_curves(wear, limit, figures)
    _plot_flute_wear(dataset, cutters, figures)
    _report_wear_shape(wear, limit)

    _plot_feature_vs_wear(features, figures)
    _report_correlations(features)
    _report_detrended_correlations(features)

    _plot_spectra(dataset, wear, fs, rpm, flutes, figures)

    print(f"\nŞekiller: {figures}")
    return 0


# ------------------------------------------------------------------ aşınma

def _plot_wear_curves(wear, limit, figures):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cutter, frame in wear.items():
        ax.plot(frame["cut"], frame["vb_um"], label=cutter,
                color=CUTTER_COLOURS.get(cutter), linewidth=1.6)
    ax.axhline(limit, color="#444", linestyle="--", linewidth=1,
               label=f"eşik {limit:.0f} µm")
    ax.set_xlabel("kesme geçişi")
    ax.set_ylabel("VB (µm)")
    ax.set_title("Aşınma eğrileri - PHM 2010 etiketli kesiciler")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "vb_curves.png", dpi=140)
    plt.close(fig)


def _plot_flute_wear(dataset, cutters, figures):
    fig, axes = plt.subplots(1, len(cutters), figsize=(13, 4), sharey=True)
    for ax, cutter in zip(np.atleast_1d(axes), cutters):
        frame = dataset.wear(cutter)
        for column in [c for c in frame.columns if c.startswith("flute_")
                       and c != "flute_spread_um"]:
            ax.plot(frame["cut"], frame[column], linewidth=1.1, label=column)
        ax.set_title(f"{cutter} - ağız bazında")
        ax.set_xlabel("kesme geçişi")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("VB (µm)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "flute_wear.png", dpi=140)
    plt.close(fig)


def _report_wear_shape(wear, limit):
    print("\n--- Aşınma eğrisi biçimi ---")
    rows = []
    for cutter, frame in wear.items():
        vb = frame["vb_um"].to_numpy()
        crossing = np.flatnonzero(vb >= limit)
        # İlk / son üçte birdeki ortalama artış hızı: eğrinin bükümünü gösterir.
        thirds = np.array_split(np.diff(vb), 3)
        rows.append({
            "kesici": cutter,
            "VB ilk": round(float(vb[0]), 1),
            "VB son": round(float(vb[-1]), 1),
            "eşiği geçtiği geçiş": int(crossing[0]) + 1 if crossing.size else None,
            "artış ilk 1/3": round(float(thirds[0].mean()), 3),
            "artış orta 1/3": round(float(thirds[1].mean()), 3),
            "artış son 1/3": round(float(thirds[2].mean()), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ öznitelik

def _plot_feature_vs_wear(features, figures):
    channels = ["force_y_rms", "vib_x_rms", "ae_rms_rms"]
    cutters = sorted(features["cutter"].unique())

    fig, axes = plt.subplots(len(channels), len(cutters),
                             figsize=(13, 8), sharex=True)
    for row, channel in enumerate(channels):
        for col, cutter in enumerate(cutters):
            ax = axes[row, col]
            subset = features[features["cutter"] == cutter].sort_values("cut")
            ax.plot(subset["cut"], subset[channel], linewidth=1.0,
                    color=CUTTER_COLOURS.get(cutter))
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(cutter)
            if col == 0:
                ax.set_ylabel(channel, fontsize=9)

            twin = ax.twinx()
            twin.plot(subset["cut"], subset["vb_um"], color="#555",
                      linewidth=0.9, linestyle="--")
            twin.tick_params(labelsize=7)
            if col == len(cutters) - 1:
                twin.set_ylabel("VB (µm)", fontsize=8, color="#555")

    for ax in axes[-1]:
        ax.set_xlabel("kesme geçişi")
    fig.suptitle("Öznitelik (renkli) ve aşınma (kesikli gri) birlikte", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "feature_vs_wear.png", dpi=140)
    plt.close(fig)


def _report_correlations(features, top_n=12):
    """Özniteliklerin aşınmayla Spearman ilişkisi.

    Kritik olan tek bir kesicideki güçlü ilişki değil, ilişkinin ÜÇ kesicide
    de aynı yönde olması. Yön değiştiren öznitelik genellemez.
    """
    print("\n--- Öznitelik / aşınma ilişkisi (Spearman) ---")

    exclude = {"cutter", "cut", "vb_um", "flute_spread_um"}
    columns = [c for c in features.columns if c not in exclude]
    cutters = sorted(features["cutter"].unique())

    rows = []
    for column in columns:
        per_cutter = {}
        for cutter in cutters:
            subset = features[features["cutter"] == cutter]
            values = subset[column].to_numpy()
            if np.all(~np.isfinite(values)) or np.nanstd(values) == 0:
                per_cutter[cutter] = np.nan
                continue
            rho, _ = stats.spearmanr(values, subset["vb_um"].to_numpy(),
                                     nan_policy="omit")
            per_cutter[cutter] = rho

        values = np.array(list(per_cutter.values()), dtype=float)
        if np.any(~np.isfinite(values)):
            continue

        rows.append({
            "oznitelik": column,
            **{c: round(float(per_cutter[c]), 3) for c in cutters},
            "ort_mutlak": round(float(np.mean(np.abs(values))), 3),
            "ayni_yon": bool(np.all(values > 0) or np.all(values < 0)),
        })

    table = pd.DataFrame(rows).sort_values("ort_mutlak", ascending=False)

    consistent = table[table["ayni_yon"]]
    print(f"\nEn güçlü {top_n} öznitelik (üç kesicide de aynı yönde):")
    print(consistent.head(top_n).to_string(index=False))

    flipping = table[~table["ayni_yon"]].head(5)
    if not flipping.empty:
        print(f"\nYön değiştirenler ({len(table) - len(consistent)} adet, ilk 5) - "
              "bunlar genellemez:")
        print(flipping.to_string(index=False))

    print(f"\nToplam öznitelik: {len(table)}  |  tutarlı yönlü: {len(consistent)}")
    return table


def _report_detrended_correlations(features, top_n=12):
    """Öznitelik, geçiş sayısının ÖTESİNDE bilgi taşıyor mu?

    Ham Spearman burada yanıltıcıdır: VB geçiş sayısıyla monoton arttığı için
    zamanla artan her öznitelik otomatik olarak ~1 korelasyon verir. Bu, naif
    tabanın zaten bildiği bilgidir - modele hiçbir şey katmaz.

    Doğru soru: geçiş sayısına bağlı düzgün monoton eğilim ikisinden de
    çıkarıldığında, özniteliğin sapmaları aşınmanın sapmalarını takip ediyor mu?
    Yani sensör, saatin söylediğinden fazlasını söylüyor mu?
    """
    from sklearn.isotonic import IsotonicRegression

    print("\n--- Geçiş sayısı etkisi çıkarıldıktan sonra (kısmi ilişki) ---")

    exclude = {"cutter", "cut", "vb_um", "flute_spread_um"}
    columns = [c for c in features.columns if c not in exclude]
    cutters = sorted(features["cutter"].unique())

    def detrend(x, y):
        """y'den, x'e bağlı monoton eğilimi çıkarır."""
        model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        try:
            return y - model.fit(x, y).predict(x)
        except ValueError:
            return None

    rows = []
    for column in columns:
        per_cutter = {}
        for cutter in cutters:
            subset = features[features["cutter"] == cutter].sort_values("cut")
            cuts = subset["cut"].to_numpy(dtype=float)
            values = subset[column].to_numpy(dtype=float)
            wear = subset["vb_um"].to_numpy(dtype=float)

            if not np.all(np.isfinite(values)) or np.nanstd(values) == 0:
                per_cutter[cutter] = np.nan
                continue

            residual_feature = detrend(cuts, values)
            residual_wear = detrend(cuts, wear)
            if residual_feature is None or residual_wear is None:
                per_cutter[cutter] = np.nan
                continue
            if np.std(residual_feature) == 0 or np.std(residual_wear) == 0:
                per_cutter[cutter] = np.nan
                continue

            rho, _ = stats.spearmanr(residual_feature, residual_wear)
            per_cutter[cutter] = rho

        values = np.array(list(per_cutter.values()), dtype=float)
        if np.any(~np.isfinite(values)):
            continue

        rows.append({
            "oznitelik": column,
            **{c: round(float(per_cutter[c]), 3) for c in cutters},
            "ort_mutlak": round(float(np.mean(np.abs(values))), 3),
            "ayni_yon": bool(np.all(values > 0) or np.all(values < 0)),
        })

    table = pd.DataFrame(rows).sort_values("ort_mutlak", ascending=False)
    consistent = table[table["ayni_yon"]]

    print(f"\nEn güçlü {top_n} (üç kesicide de aynı yönde):")
    print(consistent.head(top_n).to_string(index=False))
    print(f"\nToplam: {len(table)}  |  tutarlı yönlü: {len(consistent)}")
    print(
        "\nBu sayılar ham Spearman'dan çok daha düşük çıkar ve olması gereken de budur:\n"
        "modelin naif tabana KATABİLECEĞİ bilginin üst sınırını gösterirler."
    )
    return table


# ------------------------------------------------------------------ spektrum

def _plot_spectra(dataset, wear, fs, rpm, flutes, figures):
    """Yeni ve aşınmış takımın spektrumu - diş geçişi görünüyor mu?"""
    cutter = "c1"
    frame = wear[cutter]
    early = int(frame["cut"].iloc[0])
    late = int(frame["cut"].iloc[-1])

    f0 = spindle_frequency_hz(rpm)
    fig, ax = plt.subplots(figsize=(10, 5))

    for cut_index, label, colour in ((early, f"geçiş {early} (yeni)", "#0D6E70"),
                                     (late, f"geçiş {late} (aşınmış)", "#8E2B26")):
        signal = dataset.load_cut(cutter, cut_index)["force_y"].to_numpy()
        freqs, psd = welch_spectrum(signal, fs)
        ax.semilogy(freqs, psd, linewidth=0.9, label=label, color=colour)

    for order in range(1, 9):
        ax.axvline(order * f0, color="#888", linewidth=0.6, linestyle=":")
        ax.text(order * f0, ax.get_ylim()[1], f"{order}×", fontsize=7,
                ha="center", va="bottom", color="#888")

    ax.axvline(tooth_passing_frequency_hz(rpm, flutes), color="#111",
               linewidth=1.0, linestyle="--", label=f"diş geçişi ({flutes}×)")
    ax.set_xlim(0, 9 * f0)
    ax.set_xlabel("frekans (Hz)")
    ax.set_ylabel("güç spektral yoğunluğu")
    ax.set_title(f"{cutter} force_y - yeni ve aşınmış takım spektrumu")
    ax.legend()
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(figures / "spectrum_early_late.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
