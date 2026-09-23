from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.normalization import fit_training_normalizers, save_normalizers
from green_dc_vpp.preprocessing import add_time_features

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2019-07-01")
VALIDATION_END = pd.Timestamp("2019-10-01")
YEAR_END = pd.Timestamp("2020-01-01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_publication_dataset(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(source, parse_dates=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
    df["split"] = np.select(
        [ts < TRAIN_END, (ts >= TRAIN_END) & (ts < VALIDATION_END)],
        ["train", "validation"],
        default="test",
    )
    train = df["split"].eq("train")

    # Forecasts are causal persistence/smoothing estimates. At t they use only
    # observations through t-1, including across chronological split boundaries.
    forecast_map = {
        "P_total_dc_kW": "Pdc_forecast_kW",
        "Ppv_kW": "Ppv_forecast_kW",
        "price_buy_EUR_per_kWh": "price_forecast_EUR_per_kWh",
    }
    audit_rows: list[dict[str, object]] = []
    for source_col, output_col in forecast_map.items():
        train_default = float(pd.to_numeric(df.loc[train, source_col], errors="coerce").median())
        causal = pd.to_numeric(df[source_col], errors="coerce").shift(1).rolling(4, min_periods=1).mean()
        df[output_col] = causal.fillna(train_default).clip(lower=0.0)
        audit_rows.append(
            {
                "parameter": f"{output_col}_initial_fallback",
                "source": f"training median of {source_col}; subsequent values use causal t-1 rolling mean",
                "training_period": "2019-01-01/2019-06-30",
                "value": train_default,
                "test_data_used": False,
                "code_location": "scripts/32_build_publication_dataset.py:build_publication_dataset",
                "classification": "B_empirically_estimated",
            }
        )

    net = pd.to_numeric(df["P_import_without_battery_kW"], errors="coerce").clip(lower=0.0)
    contract_limit = float(net.loc[train].quantile(0.90))
    df["grid_import_contract_limit_kW"] = contract_limit
    audit_rows.append(
        {
            "parameter": "grid_import_contract_limit_kW",
            "source": "90th percentile of training P_import_without_battery_kW",
            "training_period": "2019-01-01/2019-06-30",
            "value": contract_limit,
            "test_data_used": False,
            "code_location": "scripts/32_build_publication_dataset.py:build_publication_dataset",
            "classification": "B_empirically_estimated",
        }
    )

    fallback_ref = float(net.loc[train].median())
    ref = net.shift(1).rolling(96, min_periods=1).mean().fillna(fallback_ref).clip(0.0, contract_limit)

    day = ts.dt.floor("D")
    day_signals = pd.DataFrame(
        {
            "day": day,
            "price": pd.to_numeric(df["price_market_EUR_per_MWh"], errors="coerce"),
            "grid_forecast": pd.to_numeric(df["grid_load_forecast_MW"], errors="coerce"),
            "split": df["split"],
        }
    ).groupby("day", sort=True).agg(
        day_price_max=("price", "max"),
        day_grid_forecast_max=("grid_forecast", "max"),
        split=("split", "first"),
    )
    training_days = day_signals["split"].eq("train")
    price_threshold = float(day_signals.loc[training_days, "day_price_max"].quantile(0.85))
    grid_threshold = float(day_signals.loc[training_days, "day_grid_forecast_max"].quantile(0.85))
    stress = day_signals[
        day_signals["day_price_max"].ge(price_threshold)
        | day_signals["day_grid_forecast_max"].ge(grid_threshold)
    ].index
    active_days = set(d for d in stress if d.weekday() in (1, 3, 4))
    active = day.isin(active_days) & ts.dt.hour.ge(16) & ts.dt.hour.lt(18)
    df["DR_active"] = active.astype(int)

    forecast_import = (df["Pdc_forecast_kW"] - df["Ppv_forecast_kW"]).clip(lower=0.0)
    reduction = np.minimum(0.35 * forecast_import, 0.60 * 750.0)
    dr_limit = (forecast_import - reduction).clip(lower=0.0)
    df["DR_import_limit_kW"] = dr_limit.where(active, np.nan)
    df["DR_import_limit_for_model_kW"] = dr_limit.where(active, 1e9)
    df["DR_required_reduction_kW"] = reduction.where(active, 0.0)
    df["Pgrid_ref_kW"] = ref
    df.loc[active, "Pgrid_ref_kW"] = np.minimum(df.loc[active, "Pgrid_ref_kW"], dr_limit.loc[active])

    for name, source_desc, value in [
        ("DR_price_day_threshold", "85th percentile of training daily day-ahead price maximum", price_threshold),
        ("DR_grid_forecast_day_threshold", "85th percentile of training daily grid-load forecast maximum", grid_threshold),
        ("Pgrid_ref_initial_fallback", "training median of no-battery grid import", fallback_ref),
    ]:
        audit_rows.append(
            {
                "parameter": name,
                "source": source_desc,
                "training_period": "2019-01-01/2019-06-30",
                "value": value,
                "test_data_used": False,
                "code_location": "scripts/32_build_publication_dataset.py:build_publication_dataset",
                "classification": "B_empirically_estimated",
            }
        )

    fixed = {
        "forecast_rolling_window_intervals": 4.0,
        "reference_rolling_window_intervals": 96.0,
        "DR_start_hour_UTC": 16.0,
        "DR_duration_hours": 2.0,
        "DR_battery_power_assumption_kW": 750.0,
    }
    for name, value in fixed.items():
        audit_rows.append(
            {
                "parameter": name,
                "source": "fixed benchmark design assumption",
                "training_period": "not fitted",
                "value": value,
                "test_data_used": False,
                "code_location": "scripts/32_build_publication_dataset.py:build_publication_dataset",
                "classification": "A_physical_or_fixed_design",
            }
        )

    df = add_time_features(df)
    normalizers = fit_training_normalizers(df, contract_limit)
    for name, value in normalizers.items():
        audit_rows.append(
            {
                "parameter": f"observation_normalizer.{name}",
                "source": "training rows only",
                "training_period": "2019-01-01/2019-06-30",
                "value": value,
                "test_data_used": False,
                "code_location": "src/green_dc_vpp/normalization.py:fit_training_normalizers",
                "classification": "B_empirically_estimated",
            }
        )
    return df, pd.DataFrame(audit_rows), normalizers


def validate(df: pd.DataFrame) -> dict[str, object]:
    ts = pd.to_datetime(df["timestamp_utc"])
    expected = pd.date_range(TRAIN_START, YEAR_END, freq="15min", inclusive="left")
    balance_error = (
        df["P_total_dc_kW"] - df["P_IT_kW"] - df["P_cooling_kW"] - df["P_other_kW"]
    ).abs().max()
    checks = {
        "rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "missing_timestamps": int(len(expected.difference(pd.DatetimeIndex(ts)))),
        "maximum_facility_balance_error_kW": float(balance_error),
        "train_rows": int(df["split"].eq("train").sum()),
        "validation_rows": int(df["split"].eq("validation").sum()),
        "test_rows": int(df["split"].eq("test").sum()),
    }
    if checks["rows"] != 35040 or checks["duplicate_timestamps"] or checks["missing_timestamps"]:
        raise ValueError(f"Publication dataset validation failed: {checks}")
    if balance_error > 1e-6:
        raise ValueError(f"Facility balance error is {balance_error} kW")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/green_dc_vpp_2019_15min.csv")
    parser.add_argument("--output-dir", default="results/publication_v1/dataset")
    args = parser.parse_args()
    source = (ROOT / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    output = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "audit").mkdir(parents=True, exist_ok=True)
    (output.parent / "config").mkdir(parents=True, exist_ok=True)

    df, audit, normalizers = build_publication_dataset(source)
    checks = validate(df)
    parquet = output / "green_dc_vpp_publication_v1_2019_15min.parquet"
    csv = output / "green_dc_vpp_publication_v1_2019_15min.csv.gz"
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False, compression="gzip")
    audit.to_csv(output.parent / "audit" / "preprocessing_parameters.csv", index=False)
    save_normalizers(normalizers, output.parent / "config" / "training_normalization_parameters.json")

    metadata = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "dataset_parquet": str(parquet.relative_to(ROOT)),
        "dataset_parquet_sha256": sha256(parquet),
        "dataset_csv_gz_sha256": sha256(csv),
        "validation": checks,
        "split_definitions": {
            "train": "2019-01-01 through 2019-06-30",
            "validation": "2019-07-01 through 2019-09-30",
            "test": "2019-10-01 through 2019-12-31",
        },
        "python": sys.version,
        "platform": platform.platform(),
    }
    metadata_path = output / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = output.parent / "DATASET_MANIFEST.md"
    manifest.write_text(
        "# Publication Dataset Manifest\n\n"
        f"- Dataset: `{parquet.relative_to(ROOT)}`\n"
        f"- SHA256: `{metadata['dataset_parquet_sha256']}`\n"
        f"- Source layer: `{metadata['source']}` (`{metadata['source_sha256']}`)\n"
        "- Rows: 35,040; interval: 15 minutes; year: 2019.\n"
        "- Train: Jan-Jun; validation: Jul-Sep; test: Oct-Dec.\n"
        "- Empirical preprocessing parameters are fitted on training rows only.\n"
        "- Forecast features are causal and use observations no later than t-1.\n"
        "- Full signal provenance remains in the dataset source/provenance columns and project documentation.\n\n"
        f"Validation: `{json.dumps(checks, sort_keys=True)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
