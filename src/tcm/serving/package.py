"""Teslim edilen model paketi - Faz 06.

Bir modeli "kaydetmek" ağırlıkları diske yazmak değildir. Altı ay sonra o
dosyadan tahmin alabilmek için şunların hepsi gerekir:

  öznitelik listesi     hangi sütunlar, HANGİ SIRADA. LightGBM sütunları
                        isimle değil konumla eşler; sıra kayarsa model
                        hata vermez, yanlış tahmin eder.
  öznitelik çıkarımı    hangi örnekleme hızı / devir / mertebe sayısıyla
                        hesaplandıkları. Bunlar değişirse aynı sinyal
                        farklı öznitelik üretir.
  taban çizgileri       her özniteliğin eğitimdeki tipik değer aralığı.
                        Bu projede DÖNÜŞÜM İÇİN KULLANILMIYOR (modele ham
                        değer giriyor); referans ve kapsam kontrolü için.
  alarm eşiği           model bir VB sayısı üretir; "takımı değiştir" demek
                        ayrı bir karardır ve eşik veriden kalibre edilir.
  kapsam                model hangi malzemeleri ve kesme koşullarını gördü.
                        Görmediği bir şey gelirse tahmin basılır AMA yanına
                        kapsam dışı uyarısı konur.
  künye                 hangi kod (git), hangi ayar (config), hangi veri.

Bu modül bunların hepsini tek bir nesnede toplar ve diske yazar.

KAPSAM UYARISI NEDEN VAR
------------------------
Bu projenin bilinen ve raporlanan sınırı: hiçbir açık veri setinde alüminyum
frezeleme aşınma verisi yok. Sistem sahada alüminyumda çalışacak ama
alüminyum görmeden eğitildi. Faz 04b'de ölçüldü: görülmemiş malzemede
parametre tabanlı model naif tabanın bile altına düşüyor.

Dolayısıyla model susup güvenle tahmin basmamalı - her tahminin yanında o
tahminin kapsam içinde mi dışında mı olduğu yazmalı. Bu, modelin doğruluğunu
artırmaz; operatörün tahmine ne kadar güveneceğini bilmesini sağlar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from tcm.decision import alarm_flags
from tcm.features.extract import (
    DERIVED_COLUMNS,
    add_derived_columns,
    sensor_columns_of,
)
from tcm.provenance import file_digest, relative_path as _display_path

MODEL_FILENAME = "model_b1.joblib"

# Kesme parametreleri ve kümülatif süre - Model B'nin varlık sebebi olan
# girdiler. Faz 04b'de ``cum_time`` öznitelik öneminde 1. sıraya çıkmıştı.
PARAMETER_COLUMNS = ["material", "feed", "doc", "rpm"]
TIME_COLUMN = "cum_time"
PARAM_TIME_COLUMNS = PARAMETER_COLUMNS + [TIME_COLUMN]

FEATURE_SET_NAMES = ("sensor+param+time", "param+time")

# Varsayılan: sensör dahil.
#
# Faz 04b'de "parametre + süre" BİLİNEN koşullarda daha iyiydi (108/114 µm vs
# 138/164 µm). Buna rağmen varsayılan değil, çünkü teslim senaryosu bilinen
# koşul değil: sahada malzeme alüminyum ve eğitim verisinde alüminyum yok.
# Yani her saha tahmini "malzeme-dışı" sınavına denk gelir. Orada:
#
#   parametre + süre          388 µm  <- naif tabanın (309) ALTINDA
#   sensör + parametre + süre 271 µm  <- naif tabanın üstünde
#
# Parametre modeli görülmemiş malzemede çöküyor çünkü dökme demirin aşınma
# HIZINI öğrenip başka malzemeye uyguluyor. Sensör modeli hızı çıkarsamak
# yerine durumu ÖLÇTÜĞÜ için daha zarif bozuluyor.
DEFAULT_FEATURE_SET = "sensor+param+time"


def resolve_feature_columns(data: pd.DataFrame, feature_set: str) -> list[str]:
    """Öznitelik kümesi adını somut sütun listesine çevirir.

    Sensör sütunları veriden türetilir; tamamen NaN olanlar atılır (Nyquist
    üstündeki mertebeler NASA'nın 250 Hz örneklemesinde ölçülemez).

    Sütun SIRASI burada sabitlenir ve pakete yazılır: çıkarımda başka bir sıra
    kullanılırsa model sessizce yanlış tahmin eder.
    """
    if feature_set not in FEATURE_SET_NAMES:
        raise ValueError(
            f"Bilinmeyen öznitelik kümesi: {feature_set!r}. "
            f"Seçenekler: {', '.join(FEATURE_SET_NAMES)}"
        )

    if feature_set == "param+time":
        return list(PARAM_TIME_COLUMNS)

    sensors = [
        column for column in sensor_columns_of(data.columns)
        if data[column].notna().any()
    ]
    return sensors + PARAM_TIME_COLUMNS


# ---------------------------------------------------------------- taban çizgileri

@dataclass
class FeatureBaselines:
    """Öznitelik başına eğitim referans istatistikleri.

    DİKKAT - bunlar bir DÖNÜŞÜM değildir. Modele giren değerler ham kalır.
    Gradyan artırma ağaçları eşik tabanlı çalışır; girdiyi ölçeklemek ağaçların
    bulduğu bölünmeleri değiştirmez, sadece eşiklerin sayısal değerini kaydırır.
    Yani standardizasyon bu modele bir şey katmaz, buna karşılık çıkarımda
    uygulanmayı unutulursa sessiz bir hata kaynağı olur.

    Saklanmalarının iki sebebi var:
      1. Kapsam kontrolü - gelen değer eğitim aralığının neresinde?
      2. Rapor ve hata ayıklama - sahada bir öznitelik eğitimdekinden 100 kat
         büyük geliyorsa sensör/bağlantı sorunu vardır, model sorunu değil.
    """

    stats: pd.DataFrame

    STAT_NAMES = ("min", "q05", "median", "mean", "q95", "max", "std")

    @classmethod
    def from_frame(cls, data: pd.DataFrame, columns: Sequence[str]) -> "FeatureBaselines":
        block = data[list(columns)].astype(float)
        stats = pd.DataFrame({
            "min": block.min(),
            "q05": block.quantile(0.05),
            "median": block.median(),
            "mean": block.mean(),
            "q95": block.quantile(0.95),
            "max": block.max(),
            "std": block.std(),
        })
        stats.index.name = "oznitelik"
        return cls(stats=stats)

    def to_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stats.to_csv(path)
        return path

    def summary(self) -> dict[str, Any]:
        """Künyeye yazılacak özet - 149 özniteliğin tamamı json'a sığdırılmaz."""
        return {
            "n_features": int(len(self.stats)),
            "stat_names": list(self.STAT_NAMES),
            "parameters": {
                column: {k: float(v) for k, v in self.stats.loc[column].items()}
                for column in PARAM_TIME_COLUMNS
                if column in self.stats.index
            },
        }


# ---------------------------------------------------------------- kapsam

@dataclass
class TrainingCoverage:
    """Modelin eğitimde gördüğü malzemeler, koşullar ve sayısal aralıklar."""

    materials: list[float]
    conditions: list[str]
    numeric_ranges: dict[str, tuple[float, float]]

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        numeric_columns: Iterable[str] = ("feed", "doc", "rpm", "cum_time"),
    ) -> "TrainingCoverage":
        ranges: dict[str, tuple[float, float]] = {}
        for column in numeric_columns:
            if column in data.columns:
                values = data[column].astype(float)
                ranges[column] = (float(values.min()), float(values.max()))

        return cls(
            materials=sorted(float(m) for m in data["material"].unique()),
            conditions=sorted(str(c) for c in data["condition"].unique()),
            numeric_ranges=ranges,
        )

    def check(self, features: pd.DataFrame) -> pd.DataFrame:
        """Satır başına kapsam değerlendirmesi.

        İki ayrı seviye döndürülür ve karıştırılmamalıdır:

        ``out_of_scope`` (SERT) - görülmemiş malzeme ya da görülmemiş kesme
            koşulu. Model bu durumda ölçülmüş bir başarısızlık sergiliyor
            (Faz 04b, malzeme-dışı sınavı). Tahmin basılır ama güvenilmez.

        ``advisory`` (YUMUŞAK) - sayısal bir parametre eğitim aralığının
            dışında, ama malzeme ve koşul tanıdık. Örneğin takım eğitimdeki
            en uzun takımdan daha uzun süredir kesiyor. Bu ekstrapolasyondur;
            ağaç tabanlı model aralık dışında SABİT tahmin verir (son yaprağa
            takılır), yani hata sessizce büyür.
        """
        known_materials = set(self.materials)
        known_conditions = set(self.conditions)

        rows = []
        for row in features.itertuples(index=False):
            hard: list[str] = []
            soft: list[str] = []

            material = float(getattr(row, "material", float("nan")))
            if not np.isnan(material) and material not in known_materials:
                hard.append(
                    f"görülmemiş malzeme (kod {material:g}; "
                    f"eğitimde {', '.join(f'{m:g}' for m in self.materials)})"
                )

            condition = getattr(row, "condition", None)
            if condition is not None and str(condition) not in known_conditions:
                hard.append(f"görülmemiş kesme koşulu ({condition})")

            for column, (low, high) in self.numeric_ranges.items():
                value = getattr(row, column, None)
                if value is None:
                    continue
                value = float(value)
                if np.isnan(value):
                    continue
                if value < low or value > high:
                    soft.append(
                        f"{column} eğitim aralığı dışında "
                        f"({value:g} ∉ [{low:g}, {high:g}])"
                    )

            rows.append({
                "out_of_scope": bool(hard),
                "out_of_scope_reason": "; ".join(hard),
                "advisory": "; ".join(soft),
            })

        return pd.DataFrame(rows, index=features.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "materials": self.materials,
            "conditions": self.conditions,
            "numeric_ranges": {k: list(v) for k, v in self.numeric_ranges.items()},
        }


# ---------------------------------------------------------------- paket

@dataclass
class ModelPackage:
    """Teslim edilen model ve onu kullanmak için gereken her şey."""

    model: Any
    feature_set: str
    feature_columns: list[str]
    baselines: FeatureBaselines
    coverage: TrainingCoverage
    thresholds: dict[str, float]
    threshold_details: dict[str, dict[str, Any]]
    active_threshold: str
    consecutive_k: int
    wear_limit_um: float
    cost_missed: float
    cost_false_alarm: float
    extraction: dict[str, float]
    provenance: dict[str, Any]
    n_train: int
    training_keys: list[tuple[float, float]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # ------------------------------------------------------------ eşik

    @property
    def threshold(self) -> float:
        """Etkin alarm eşiği (µm)."""
        return float(self.thresholds[self.active_threshold])

    # ------------------------------------------------------------ çıkarım

    def prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        """Çıkarım tablosunu doğrular ve türetilmiş sütunları tamamlar.

        ``cum_time`` / ``condition`` eksikse ``extract.add_derived_columns`` ile
        üretilir - eğitimdeki TANIMIN AYNISIYLA. Çağıranın kendi formülünü
        yazmasına izin verilmiyor: koşul kimliğinde tek bir biçim farkı bile
        (``2`` yerine ``2.0``) tanıdık bir koşulu görülmemiş gösterir.
        """
        result = features.copy()

        missing_derived = [c for c in DERIVED_COLUMNS if c not in result.columns]
        if missing_derived:
            required = {"case", "run_time", "material", "doc", "feed"}
            absent = required - set(result.columns)
            if absent:
                raise ValueError(
                    f"Türetilmiş sütunlar ({', '.join(missing_derived)}) eksik ve "
                    f"hesaplanamıyor; şu sütunlar da yok: {', '.join(sorted(absent))}"
                )
            result = add_derived_columns(result)

        missing = [c for c in self.feature_columns if c not in result.columns]
        if missing:
            raise ValueError(
                f"Çıkarım tablosunda {len(missing)} öznitelik eksik. "
                f"İlk birkaçı: {', '.join(missing[:5])}"
            )
        return result

    def predict(
        self,
        features: pd.DataFrame,
        *,
        group_column: str = "case",
        sort_column: str | None = "run",
    ) -> pd.DataFrame:
        """VB tahmini + alarm kararı + kapsam değerlendirmesi.

        Alarm kilitlenmesi ``group_column`` içinde ayrı işler: aşınma geri
        dönmediği için alarm da sönmez, ama bu kilit bir takımdan diğerine
        TAŞMAMALIDIR. (Bu hata Faz 09'da gerçekten yapıldı ve eşik seçimini
        bozmuştu; ``tcm.decision.alarm_flags`` o düzeltmeyi taşıyor.)
        """
        prepared = self.prepare(features)
        if sort_column is not None and sort_column in prepared.columns:
            prepared = prepared.sort_values([group_column, sort_column])

        predicted = np.asarray(
            self.model.predict(prepared[self.feature_columns]), dtype=float
        )

        groups = (
            prepared[group_column].to_numpy()
            if group_column in prepared.columns else None
        )
        latched = alarm_flags(
            predicted, self.threshold, self.consecutive_k, groups
        )

        scope = self.coverage.check(prepared)

        result = pd.DataFrame(index=prepared.index)
        for column in ("case", "run", "material", "condition", "cum_time"):
            if column in prepared.columns:
                result[column] = prepared[column].to_numpy()

        result["vb_pred_um"] = predicted
        result["esik_um"] = self.threshold
        result["raw_worn"] = predicted >= self.threshold
        result["worn"] = latched
        result["out_of_scope"] = scope["out_of_scope"].to_numpy()
        result["out_of_scope_reason"] = scope["out_of_scope_reason"].to_numpy()
        result["advisory"] = scope["advisory"].to_numpy()
        result["in_sample"] = self._in_sample_flags(prepared)

        if "vb_um" in prepared.columns:
            result["vb_true_um"] = prepared["vb_um"].to_numpy()

        return result.reset_index(drop=True)

    def _in_sample_flags(self, prepared: pd.DataFrame) -> np.ndarray:
        """Bu satır nihai modelin EĞİTİM verisinde var mıydı?

        Nihai model 145 satırın tamamıyla eğitildiği için, aynı veriyle alınan
        tahminler örneklem içidir ve model o satırların cevabını zaten görmüştür.
        Böyle bir tahminin hatası gerçek saha hatasını temsil ETMEZ - iyimserdir.
        Gerçek performans tahmini için çapraz doğrulama sayıları (Faz 04b)
        kullanılmalıdır.
        """
        if not self.training_keys or not {"case", "run"} <= set(prepared.columns):
            return np.zeros(len(prepared), dtype=bool)

        known = {(float(c), float(r)) for c, r in self.training_keys}
        return np.array(
            [
                (float(c), float(r)) in known
                for c, r in zip(prepared["case"], prepared["run"])
            ],
            dtype=bool,
        )

    # ------------------------------------------------------------ disk

    def save(
        self,
        directory: Path | str,
        manifest_path: Path | str | None = None,
        baselines_path: Path | str | None = None,
    ) -> dict[str, Path]:
        """Paketi diske yazar.

        İkili dosya (``joblib``) yeniden üretilebilir olduğu için depoya
        girmez; künye (json) ve taban çizgileri (csv) metin olduğu için girer.
        Böylece "hangi model, hangi eşikle, hangi kodla" sorusu git geçmişinden
        yanıtlanabilir - ikili dosyayı depoda taşımadan.
        """
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / MODEL_FILENAME
        joblib.dump(self, model_path)

        written = {"model": model_path}

        if baselines_path is not None:
            written["baselines"] = self.baselines.to_csv(baselines_path)

        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            manifest = self.manifest(model_path)
            manifest["files"]["model_dir"] = _display_path(directory)
            if "baselines" in written:
                manifest["files"]["baselines_csv"] = _display_path(written["baselines"])

            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            written["manifest"] = manifest_path

        return written

    def manifest(self, model_path: Path | str | None = None) -> dict[str, Any]:
        """Depoya girecek künye. İkili dosya hariç her şey buradadır.

        ``model_sha256`` kritik: künye ile ikili dosyanın birbirine ait olduğunu
        doğrular. İkili yeniden üretildiğinde özet değişir; künyedekiyle
        tutmuyorsa elinizdeki model künyedeki model değildir.
        """
        return {
            "model": "B-1 (LightGBM) - NASA Milling",
            "created_at": self.created_at,
            "feature_set": self.feature_set,
            "n_features": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
            "n_train": self.n_train,
            "training_cases": sorted({float(c) for c, _ in self.training_keys}),
            "decision": {
                "active_threshold": self.active_threshold,
                "threshold_um": self.threshold,
                "consecutive_k": self.consecutive_k,
                "wear_limit_um": self.wear_limit_um,
                "cost_missed": self.cost_missed,
                "cost_false_alarm": self.cost_false_alarm,
                "thresholds_um": {k: float(v) for k, v in self.thresholds.items()},
                "calibration": self.threshold_details,
            },
            "coverage": self.coverage.to_dict(),
            "baselines": self.baselines.summary(),
            "extraction": dict(self.extraction),
            "provenance": self.provenance,
            "files": {
                "model_joblib": MODEL_FILENAME,
                "model_sha256": (
                    file_digest(model_path) if model_path is not None else "bilinmiyor"
                ),
            },
        }

    @classmethod
    def load(cls, directory: Path | str) -> "ModelPackage":
        """Paketi diskten okur."""
        import joblib

        directory = Path(directory)
        path = directory / MODEL_FILENAME if directory.is_dir() else directory
        if not path.exists():
            raise FileNotFoundError(
                f"Model paketi bulunamadı: {path}\n"
                "Üretmek için: python scripts/train_model.py"
            )
        return joblib.load(path)

    def describe(self) -> str:
        """Konsola basılacak insan okunur özet."""
        thresholds = "  ".join(
            f"{name}={value:.1f}" for name, value in self.thresholds.items()
        )
        return (
            f"Model      : B-1 (LightGBM), {self.n_train} satırın tamamıyla eğitildi\n"
            f"Girdi      : {self.feature_set} ({len(self.feature_columns)} öznitelik)\n"
            f"Eşikler    : {thresholds}  (etkin: {self.active_threshold} = "
            f"{self.threshold:.1f} µm)\n"
            f"Ardışık k  : {self.consecutive_k}\n"
            f"Kapsam     : malzeme {self.coverage.materials}, "
            f"{len(self.coverage.conditions)} kesme koşulu"
        )


