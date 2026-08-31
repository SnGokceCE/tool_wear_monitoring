"""Teslim paketinin ve çıkarım hattının testleri (Faz 06).

Buradaki testlerin çoğu "kod çalışıyor mu" değil, "eğitim ile çıkarım aynı
şeyi mi yapıyor" sorusunu soruyor. Bu ayrım önemli: eğitim/servis sapması
hata vermez, sadece yanlış tahmin ettirir. Ancak açıkça test edilirse yakalanır.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tcm.datasets.nasa import run_table
from tcm.decision import calibrate_threshold
from tcm.features.build import build_nasa_features
from tcm.features.extract import add_derived_columns, assemble_feature_table
from tcm.models.gbm import make_gbm_small
from tcm.provenance import run_stamp
from tcm.serving import (
    FeatureBaselines,
    ModelPackage,
    TrainingCoverage,
    resolve_feature_columns,
)

SAMPLING_RATE_HZ = 250.0
RPM = 826.0
MAX_ORDER = 8

CHANNELS = ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle")


class FakeNASA:
    """``mill.mat`` yerine geçen sentetik veri kümesi.

    Gerçek veriye bağlı olmayan testler için: ``data/raw`` her makinede yok,
    ama eğitim/çıkarım tutarlılığı her makinede sınanabilmeli.

    Sinyaller tohumu koşu indeksinden türetilir - böylece aynı koşu her
    çağrıda aynı sinyali verir ve tahminler tekrarlanabilir olur.
    """

    def __init__(self, n_cases: int = 6, runs_per_case: int = 8, n_samples: int = 1024):
        self.n_samples = n_samples
        rows = []
        entry = 0
        for case in range(1, n_cases + 1):
            # İki malzeme, iki ilerleme, iki kesme derinliği -> 8 koşul mümkün.
            material = 1.0 if case % 2 else 2.0
            feed = 0.25 if case % 3 else 0.5
            doc = 0.75 if case % 4 else 1.5
            for run in range(1, runs_per_case + 1):
                rows.append({
                    "entry": entry,
                    "case": float(case),
                    "run": float(run),
                    # Aşınma zamanla monoton artar - fiziğe uygun sentetik etiket.
                    "VB": 0.05 * run + 0.02 * case,
                    "time": 10.0 + run,
                    "DOC": doc,
                    "feed": feed,
                    "material": material,
                    "has_label": True,
                })
                entry += 1
        self._meta = pd.DataFrame(rows)

    def metadata(self) -> pd.DataFrame:
        return self._meta.copy()

    def signals(self, entry_index: int) -> pd.DataFrame:
        rng = np.random.default_rng(1000 + int(entry_index))
        t = np.arange(self.n_samples) / SAMPLING_RATE_HZ
        # Aşınma arttıkça genlik artsın: sinyalin bilgi taşıması için.
        gain = 1.0 + 0.05 * int(entry_index)
        base = gain * np.sin(2 * np.pi * (RPM / 60.0) * 3 * t)
        return pd.DataFrame({
            channel: (base + rng.normal(0, 0.1, self.n_samples)).astype(np.float32)
            for channel in CHANNELS
        })


@pytest.fixture(scope="module")
def dataset() -> FakeNASA:
    return FakeNASA()


@pytest.fixture(scope="module")
def training_table(dataset: FakeNASA) -> pd.DataFrame:
    """Eğitim yolundan üretilen öznitelik tablosu."""
    return build_nasa_features(
        dataset,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        rpm=RPM,
        max_order=MAX_ORDER,
        show_progress=False,
    )


# --------------------------------------------------------------------------
# 1. Eğitim / çıkarım tutarlılığı - bu dosyanın asıl sebebi
# --------------------------------------------------------------------------

class TestTrainingInferenceParity:
    def test_inference_path_reproduces_training_features(self, dataset, training_table):
        """Çıkarım yolu, eğitim tablosunun AYNISINI üretmeli.

        Eğitim ``build_nasa_features`` çağırıyor; çıkarım (predict.py)
        ``run_table`` + ``assemble_feature_table`` çağırıyor. İkisi aynı
        çekirdeğe inmezse öznitelikler sessizce kayar - model hata vermeden
        yanlış tahmin eder.
        """
        runs = run_table(dataset.metadata())
        inferred = assemble_feature_table(
            runs,
            lambda entry: dataset.signals(int(entry.entry)),
            sampling_rate_hz=SAMPLING_RATE_HZ,
            rpm=RPM,
            max_order=MAX_ORDER,
            show_progress=False,
        )

        assert list(inferred.columns) == list(training_table.columns), \
            "sütun listesi veya SIRASI ayrıştı - LightGBM sütunları konumla eşler"
        pd.testing.assert_frame_equal(inferred, training_table)

    def test_derived_columns_are_defined_in_one_place(self, training_table):
        """``cum_time`` ve ``condition`` yeniden hesaplanınca değişmemeli."""
        recomputed = add_derived_columns(
            training_table.drop(columns=["cum_time", "condition"])
        )
        pd.testing.assert_series_equal(
            recomputed["cum_time"], training_table["cum_time"]
        )
        pd.testing.assert_series_equal(
            recomputed["condition"], training_table["condition"]
        )

    def test_cum_time_resets_per_tool(self, training_table):
        """Kümülatif süre her takımda sıfırdan başlar - sahada sayaç sıfırlanır."""
        first = training_table.sort_values(["case", "run"]).groupby("case").first()
        assert (first["cum_time"] == first["run_time"]).all()

    def test_vb_is_converted_to_micrometres(self, dataset):
        """mm -> µm çevrimi tek yerde; unutulursa model 1000 kat yanlış ölçekte."""
        runs = run_table(dataset.metadata())
        expected = dataset.metadata()["VB"].to_numpy() * 1000.0
        np.testing.assert_allclose(np.sort(runs["vb_um"]), np.sort(expected))


# --------------------------------------------------------------------------
# 2. Öznitelik kümesi çözümü
# --------------------------------------------------------------------------

class TestFeatureSets:
    def test_param_time_is_five_columns(self, training_table):
        columns = resolve_feature_columns(training_table, "param+time")
        assert columns == ["material", "feed", "doc", "rpm", "cum_time"]

    def test_sensor_set_includes_parameters_at_the_end(self, training_table):
        columns = resolve_feature_columns(training_table, "sensor+param+time")
        assert columns[-5:] == ["material", "feed", "doc", "rpm", "cum_time"]
        assert len(columns) > 100

    def test_all_nan_columns_are_dropped(self, training_table):
        """Nyquist üstündeki mertebeler NASA'da ölçülemez - modele girmemeli."""
        columns = resolve_feature_columns(training_table, "sensor+param+time")
        assert all(training_table[c].notna().any() for c in columns)

    def test_unknown_feature_set_is_rejected(self, training_table):
        with pytest.raises(ValueError, match="Bilinmeyen öznitelik kümesi"):
            resolve_feature_columns(training_table, "hepsi")


# --------------------------------------------------------------------------
# 3. Kapsam kontrolü - alüminyum uyarısının mekanizması
# --------------------------------------------------------------------------

class TestCoverage:
    @pytest.fixture
    def coverage(self, training_table) -> TrainingCoverage:
        return TrainingCoverage.from_frame(training_table)

    def test_training_rows_are_in_scope(self, coverage, training_table):
        result = coverage.check(training_table)
        assert not result["out_of_scope"].any()
        assert (result["out_of_scope_reason"] == "").all()

    def test_unseen_material_is_flagged(self, coverage, training_table):
        """Alüminyum senaryosu: eğitimde olmayan malzeme kodu."""
        row = training_table.head(1).copy()
        row["material"] = 9.0
        row = add_derived_columns(row)

        result = coverage.check(row)
        assert bool(result["out_of_scope"].iloc[0])
        assert "görülmemiş malzeme" in result["out_of_scope_reason"].iloc[0]

    def test_unseen_condition_is_flagged(self, coverage, training_table):
        """Malzeme tanıdık ama kesme koşulu yeni."""
        row = training_table.head(1).copy()
        row["feed"] = 99.0
        row = add_derived_columns(row)

        result = coverage.check(row)
        assert bool(result["out_of_scope"].iloc[0])
        assert "görülmemiş kesme koşulu" in result["out_of_scope_reason"].iloc[0]

    def test_numeric_extrapolation_is_advisory_not_out_of_scope(
        self, coverage, training_table
    ):
        """Takım eğitimdekinden uzun süredir kesiyor: uyarı evet, kapsam dışı hayır.

        Ayrım kasıtlı. Görülmemiş malzeme ölçülmüş bir başarısızlık; aralık
        dışı bir süre ise yalnızca ekstrapolasyon - model sabit tahmine takılır
        ama tamamen geçersiz değildir.
        """
        row = training_table.head(1).copy()
        row["cum_time"] = training_table["cum_time"].max() * 10

        result = coverage.check(row)
        assert not bool(result["out_of_scope"].iloc[0])
        assert "cum_time" in result["advisory"].iloc[0]


# --------------------------------------------------------------------------
# 4. Eşik kalibrasyonu
# --------------------------------------------------------------------------

class TestThresholdCalibration:
    def test_threshold_stays_within_search_span(self, training_table):
        limit = float(training_table["vb_um"].median())
        result = calibrate_threshold(
            training_table, resolve_feature_columns(training_table, "param+time"),
            lambda: make_gbm_small(random_state=42),
            group_column="case", wear_limit_um=limit, search_span=0.5,
        )
        assert limit * 0.5 <= result.threshold <= limit * 1.5

    def test_material_split_needs_min_groups_two(self, training_table):
        """İki malzeme varsa min_groups=3 ``case``'e düşer, 2 düşmez.

        Bu ayar teslim senaryosunun ta kendisini belirliyor: ``case``'e
        düşülürse sınav "görülmemiş takım"a dönüşür, kolaylaşır ve eşik
        olduğundan gevşek seçilir.
        """
        limit = float(training_table["vb_um"].median())
        columns = resolve_feature_columns(training_table, "param+time")
        factory = lambda: make_gbm_small(random_state=42)

        fell_back = calibrate_threshold(
            training_table, columns, factory, group_column="material",
            wear_limit_um=limit, min_groups=3,
        )
        kept = calibrate_threshold(
            training_table, columns, factory, group_column="material",
            wear_limit_um=limit, min_groups=2,
        )

        assert fell_back.fell_back and fell_back.split_column == "case"
        assert not kept.fell_back and kept.split_column == "material"
        assert kept.n_folds == 2

    def test_calibration_is_recorded_for_the_manifest(self, training_table):
        limit = float(training_table["vb_um"].median())
        result = calibrate_threshold(
            training_table, resolve_feature_columns(training_table, "param+time"),
            lambda: make_gbm_small(random_state=42),
            group_column="material", wear_limit_um=limit, min_groups=2,
        )
        payload = result.to_dict()
        assert payload["requested_split"] == "material"
        assert payload["actual_split"] == "material"
        assert payload["fell_back"] is False
        assert payload["n_folds"] == 2


# --------------------------------------------------------------------------
# 5. Paketin kaydedilip yüklenmesi
# --------------------------------------------------------------------------

def _build_package(training_table: pd.DataFrame, feature_set: str = "param+time"):
    columns = resolve_feature_columns(training_table, feature_set)
    limit = float(training_table["vb_um"].median())

    model = make_gbm_small(random_state=42)
    model.fit(training_table[columns], training_table["vb_um"])

    return ModelPackage(
        model=model,
        feature_set=feature_set,
        feature_columns=columns,
        baselines=FeatureBaselines.from_frame(training_table, columns),
        coverage=TrainingCoverage.from_frame(training_table),
        thresholds={"case": limit * 0.9, "material": limit * 0.7},
        threshold_details={"case": {}, "material": {}},
        active_threshold="material",
        consecutive_k=1,
        wear_limit_um=limit,
        cost_missed=5.0,
        cost_false_alarm=1.0,
        extraction={
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "rpm": RPM,
            "max_order": MAX_ORDER,
            "keep": 0.5,
            "drop_cases": [],
        },
        provenance={**run_stamp(), "random_seed": 42},
        n_train=len(training_table),
        training_keys=[
            (float(c), float(r))
            for c, r in zip(training_table["case"], training_table["run"])
        ],
    )


class TestPackageRoundTrip:
    def test_save_and_load_preserves_everything_needed(self, training_table, tmp_path):
        package = _build_package(training_table)
        written = package.save(
            tmp_path / "model",
            manifest_path=tmp_path / "manifest.json",
            baselines_path=tmp_path / "baselines.csv",
        )
        loaded = ModelPackage.load(tmp_path / "model")

        assert loaded.feature_columns == package.feature_columns
        assert loaded.feature_set == package.feature_set
        assert loaded.thresholds == package.thresholds
        assert loaded.active_threshold == package.active_threshold
        assert loaded.threshold == pytest.approx(package.threshold)
        assert loaded.consecutive_k == package.consecutive_k
        assert loaded.extraction == package.extraction
        assert loaded.provenance["git_hash"] == package.provenance["git_hash"]
        assert loaded.n_train == package.n_train
        assert loaded.coverage.materials == package.coverage.materials
        assert set(written) == {"model", "manifest", "baselines"}

    def test_predictions_survive_the_round_trip(self, training_table, tmp_path):
        package = _build_package(training_table)
        before = package.predict(training_table)

        package.save(tmp_path / "model")
        after = ModelPackage.load(tmp_path / "model").predict(training_table)

        np.testing.assert_allclose(before["vb_pred_um"], after["vb_pred_um"])
        np.testing.assert_array_equal(before["worn"], after["worn"])

    def test_manifest_has_the_fields_the_report_needs(self, training_table, tmp_path):
        package = _build_package(training_table)
        package.save(
            tmp_path / "model",
            manifest_path=tmp_path / "manifest.json",
            baselines_path=tmp_path / "baselines.csv",
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["feature_columns"] == package.feature_columns
        assert manifest["decision"]["threshold_um"] == pytest.approx(package.threshold)
        assert manifest["decision"]["thresholds_um"].keys() == {"case", "material"}
        assert manifest["coverage"]["materials"] == package.coverage.materials
        assert manifest["provenance"]["git_hash"]
        assert manifest["provenance"]["random_seed"] == 42
        assert manifest["extraction"]["sampling_rate_hz"] == SAMPLING_RATE_HZ
        assert len(manifest["files"]["model_sha256"]) == 64

    def test_manifest_digest_matches_the_binary(self, training_table, tmp_path):
        """Künye ile ikili dosyanın birbirine ait olduğu doğrulanabilmeli."""
        from tcm.provenance import file_digest

        package = _build_package(training_table)
        written = package.save(
            tmp_path / "model", manifest_path=tmp_path / "manifest.json"
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["files"]["model_sha256"] == file_digest(written["model"])

    def test_baselines_csv_covers_every_feature(self, training_table, tmp_path):
        package = _build_package(training_table)
        package.save(tmp_path / "model", baselines_path=tmp_path / "baselines.csv")
        stats = pd.read_csv(tmp_path / "baselines.csv", index_col="oznitelik")
        assert sorted(stats.index) == sorted(package.feature_columns)

    def test_missing_package_gives_a_useful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_model.py"):
            ModelPackage.load(tmp_path / "yok")


# --------------------------------------------------------------------------
# 6. Çıkarım davranışı
# --------------------------------------------------------------------------

class TestPredict:
    def test_predictions_match_the_raw_model(self, training_table):
        """Paketin sardığı model, doğrudan çağrılan modelle aynı sayıyı vermeli."""
        package = _build_package(training_table)
        result = package.predict(training_table)

        expected = package.model.predict(
            training_table.sort_values(["case", "run"])[package.feature_columns]
        )
        np.testing.assert_allclose(result["vb_pred_um"], expected)

    def test_alarm_uses_the_active_threshold(self, training_table):
        package = _build_package(training_table)
        result = package.predict(training_table)
        np.testing.assert_array_equal(
            result["raw_worn"], result["vb_pred_um"] >= package.threshold
        )

    def test_alarm_latches_within_a_tool_but_not_across_tools(self, training_table):
        """Alarm bir kez çalınca sönmez - ama sonraki takıma taşmaz.

        Bu, Faz 09'da gerçekten yapılan ve eşik seçimini bozan hatanın
        çıkarım tarafındaki karşılığı.
        """
        package = _build_package(training_table)
        result = package.predict(training_table)

        for _, group in result.groupby("case"):
            flags = group.sort_values("run")["worn"].to_numpy()
            assert list(flags) == list(np.maximum.accumulate(flags)), \
                "alarm takım içinde sönmemeli"

        # İlk takımda alarm varsa, bu sonraki her takımı da alarma sokmamalı.
        per_tool = result.groupby("case")["worn"].first()
        assert not per_tool.all() or result["raw_worn"].all(), \
            "kilitlenme takımlar arasına taşmış olabilir"

    def test_training_rows_are_marked_in_sample(self, training_table):
        """Nihai model tüm veriyle eğitildi: bu tahminler iyimser, işaretlenmeli."""
        package = _build_package(training_table)
        result = package.predict(training_table)
        assert result["in_sample"].all()

    def test_unseen_rows_are_not_marked_in_sample(self, training_table):
        package = _build_package(training_table)
        fresh = training_table.head(3).copy()
        fresh["case"] = 99.0
        fresh["run"] = [1.0, 2.0, 3.0]

        result = package.predict(add_derived_columns(fresh))
        assert not result["in_sample"].any()

    def test_out_of_scope_rows_still_get_a_prediction(self, training_table):
        """Kapsam dışında tahmin BASILIR - ama yanında uyarısıyla.

        Susmak yanlış olurdu: operatörün bir sayıya ihtiyacı var. Uyarısız
        basmak da yanlış olurdu: o sayıya ne kadar güveneceğini bilemez.
        """
        package = _build_package(training_table)
        row = training_table.head(1).copy()
        row["material"] = 9.0

        result = package.predict(add_derived_columns(row))
        assert bool(result["out_of_scope"].iloc[0])
        assert np.isfinite(result["vb_pred_um"].iloc[0])
        assert "görülmemiş malzeme" in result["out_of_scope_reason"].iloc[0]

    def test_missing_feature_column_is_rejected_loudly(self, training_table):
        """Eksik sütunla sessizce tahmin üretmektense hata vermek yeğdir."""
        package = _build_package(training_table)
        broken = training_table.drop(columns=["cum_time"])
        broken = broken.drop(columns=["run_time"])

        with pytest.raises(ValueError):
            package.predict(broken)
