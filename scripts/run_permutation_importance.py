"""Kesme parametrelerinin permütasyon önemi - iki model için (Faz 12).

NEDEN VAR
---------
LightGBM hangi özniteliği ne kadar kullandığını kendi raporlayabiliyor
(``feature_importances_``, dallanma sayısı). CNN+GRU'da böyle bir sayaç yok:
ağa ham sinyal giriyor ve karar binlerce ağırlığa dağılmış durumda.

Ama derin modele giren **5 kesme parametresi** isimli skaler değerler. Onlar
üzerinde permütasyon önemi hesaplanabilir:

    1. Test kümesinde bir parametrenin değerlerini karıştır
    2. Tahmini tekrarla, MAE ne kadar arttı bak
    3. Artış = modelin o parametreye bağımlılığı

Model YENİDEN EĞİTİLMİYOR; yalnızca tahmin tekrarlanıyor. Bu yüzden ölçüm
"model o bilgi olmadan ne yapardı" değil, "model o bilgiyi ne kadar
kullanıyor" sorusunun cevabı.

Aynı hesap LightGBM için de yapılıyor - böylece iki model karşılaştırılabilir
hale geliyor ve LightGBM'in dallanma tabanlı önemiyle çapraz kontrol edilmiş
oluyor.

BEKLENEN KONTROL
----------------
``rpm`` NASA'da sabit (826). Sabit bir sütunu karıştırmak hiçbir şeyi
değiştirmez, dolayısıyla önemi SIFIR çıkmalı. Çıkmazsa yöntemde hata vardır.

    python scripts/run_permutation_importance.py
    python scripts/run_permutation_importance.py --repeats 20 --save
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from tcm import PROJECT_ROOT, load_config
from tcm.cli import setup_console
from tcm.models.deep import (
    CNNGRUWearModel,
    SignalStandardiser,
    predict,
    require_torch,
    train_model,
)
from tcm.models.gbm import SMALL_DATA_PARAMS
from tcm.provenance import format_stamp, run_stamp
from tcm.serving import resolve_feature_columns


def _load_split_module():
    """Bölme tanımını kardeş betikten alır - aynı bölme kullanılmalı."""
    path = PROJECT_ROOT / "scripts" / "run_holdout_split.py"
    spec = importlib.util.spec_from_file_location("_holdout_split", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10,
                        help="her parametre kaç kez karıştırılacak")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    require_torch()
    split_module = _load_split_module()

    config = load_config(args.config)
    seed = int(config.get("random_seed", 42))
    parameters_names = split_module.DEEP_PARAMETERS

    data = pd.read_csv(
        config.path("paths.data_processed") / "nasa_run_features.csv"
    ).sort_values(["case", "run"]).reset_index(drop=True)
    parts = split_module._split_by_tool(data)
    train, validation, test = parts["eğitim"], parts["doğrulama"], parts["test"]

    print("=" * 84)
    print("PERMÜTASYON ÖNEMİ - kesme parametreleri")
    print("=" * 84)
    print(f"Bölme    : eğitim {len(train)} | doğrulama {len(validation)} | "
          f"test {len(test)} koşu")
    print(f"Parametre: {', '.join(parameters_names)}")
    print(f"Karıştırma tekrarı: {args.repeats}")
    print("\nÖnem = bir parametre karıştırıldığında test MAE'sinin ARTIŞI (µm).")
    print("Yüksek artış = model o parametreye çok bağımlı.\n")

    rows = []
    rows += _gbm_importance(data, train, validation, test, parameters_names,
                            seed, args.repeats, config)
    rows += _deep_importance(config, data, parts, parameters_names, seed,
                             args.seeds, args.epochs, args.repeats,
                             split_module)

    table = pd.DataFrame(rows)
    _report(table, parameters_names)

    stamp = run_stamp(args.config) if args.config else run_stamp()
    print("\n" + "-" * 84)
    print(format_stamp(stamp))

    if args.save:
        target = config.path("paths.reports")
        target.mkdir(parents=True, exist_ok=True)
        out = target / "permutation_importance.csv"
        table.assign(git_hash=stamp["git_hash"]).to_csv(out, index=False)
        print(f"\nKaydedildi: {out}")

    return 0


def _permuted_mae(predict_fn, values, truth, column, repeats, rng):
    """Bir sütunu ``repeats`` kez karıştırıp ortalama MAE döndürür."""
    scores = []
    for _ in range(repeats):
        shuffled = values.copy()
        shuffled[:, column] = rng.permutation(shuffled[:, column])
        scores.append(float(np.mean(np.abs(predict_fn(shuffled) - truth))))
    return float(np.mean(scores)), float(np.std(scores))


def _gbm_importance(data, train, validation, test, names, seed, repeats, config):
    """LightGBM - dallanma tabanlı önemle çapraz kontrol için."""
    columns = resolve_feature_columns(data, "sensor+param+time")
    model = LGBMRegressor(**{**SMALL_DATA_PARAMS, "n_estimators": 2000},
                          random_state=seed)
    model.fit(
        train[columns], train["vb_um"],
        eval_set=[(validation[columns], validation["vb_um"])],
        eval_metric="l1",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    values = test[columns].to_numpy(dtype=float)
    truth = test["vb_um"].to_numpy(dtype=float)
    baseline = float(np.mean(np.abs(model.predict(values) - truth)))

    rng = np.random.default_rng(seed)
    rows = []
    for name in names:
        index = columns.index(name)
        mean, spread = _permuted_mae(model.predict, values, truth, index,
                                     repeats, rng)
        rows.append({"model": "LightGBM", "parametre": name,
                     "taban_mae": baseline, "karisik_mae": mean,
                     "onem": mean - baseline, "sacilim": spread})
    return rows


def _deep_importance(config, data, parts, names, seed, n_seeds, epochs,
                     repeats, split_module):
    """CNN+GRU - tohum başına eğitilip önemler ortalanıyor."""
    signals = split_module._load_signals(config, data)
    idx = {k: v.index.to_numpy() for k, v in parts.items()}

    raw = data[names].to_numpy(dtype=np.float32)
    targets = data["vb_um"].to_numpy(dtype=np.float32)

    standardiser = SignalStandardiser()
    x = {k: standardiser.fit_transform(signals[v]) if k == "eğitim"
         else standardiser.transform(signals[v]) for k, v in idx.items()}

    mean = raw[idx["eğitim"]].mean(axis=0)
    std = raw[idx["eğitim"]].std(axis=0)
    std[std < 1e-9] = 1.0
    p = {k: (raw[v] - mean) / std for k, v in idx.items()}
    y = {k: targets[v] for k, v in idx.items()}

    truth = y["test"].astype(float)
    per_seed = {name: [] for name in names}
    baselines = []

    for offset in range(n_seeds):
        current = seed + offset
        model = CNNGRUWearModel(n_channels=signals.shape[1],
                                n_parameters=len(names))
        model = train_model(
            model, x["eğitim"], p["eğitim"], y["eğitim"],
            epochs=epochs, seed=current,
            validation=(x["doğrulama"], p["doğrulama"], y["doğrulama"]),
            patience=25,
        )

        def predict_fn(params, _model=model):
            return predict(_model, x["test"], params.astype(np.float32))

        baseline = float(np.mean(np.abs(predict_fn(p["test"]) - truth)))
        baselines.append(baseline)
        print(f"  tohum {current}: taban test MAE {baseline:7.2f} µm")

        rng = np.random.default_rng(current)
        for index, name in enumerate(names):
            value, _ = _permuted_mae(predict_fn, p["test"].astype(float),
                                     truth, index, repeats, rng)
            per_seed[name].append(value - baseline)

    return [{
        "model": "CNN+GRU", "parametre": name,
        "taban_mae": float(np.mean(baselines)),
        "karisik_mae": float(np.mean(baselines)) + float(np.mean(values)),
        "onem": float(np.mean(values)),
        "sacilim": float(np.std(values)),
    } for name, values in per_seed.items()]


def _report(table: pd.DataFrame, names) -> None:
    print("\n" + "=" * 84)
    print("SONUÇ - parametre karıştırılınca test MAE'si ne kadar arttı (µm)")
    print("=" * 84)

    pivot = table.pivot(index="parametre", columns="model", values="onem")
    pivot = pivot.reindex(names)
    print(pivot.to_string(float_format=lambda v: f"{v:9.2f}"))

    print("\nTaban MAE (karıştırma yok):")
    for model, group in table.groupby("model"):
        print(f"  {model:10s} {group['taban_mae'].iloc[0]:7.2f} µm")

    rpm = table[table["parametre"] == "rpm"]
    print("\nKONTROL - rpm NASA'da sabit, karıştırmak hiçbir şeyi değiştirmemeli:")
    for row in rpm.itertuples(index=False):
        flag = "✓" if abs(row.onem) < 0.5 else "✗ BEKLENMEDİK"
        print(f"  {row.model:10s} önem {row.onem:+7.3f} µm   {flag}")

    print("\nOKUMA")
    print("-" * 84)
    for model, group in table.groupby("model"):
        top = group.sort_values("onem", ascending=False).iloc[0]
        share = 100 * top.onem / group["taban_mae"].iloc[0]
        print(f"  {model:10s} en bağımlı olduğu parametre: {top.parametre} "
              f"(+{top.onem:.2f} µm, tabanın %{share:.0f}'i)")


if __name__ == "__main__":
    sys.exit(main())
