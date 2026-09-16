from pathlib import Path

import pandas as pd


def load_telemetry(data):
    files = sorted((data / "telemetry").rglob("*.parquet"))

    if not files:
        raise FileNotFoundError("No telemetry parquet files found")

    parts = []

    for file in files:
        try:
            df = pd.read_parquet(
                file,
                columns=["gateway_id", "ts_utc"]
            )
            parts.append(df)
        except Exception:
            continue

    if not parts:
        raise RuntimeError("Could not read telemetry files")

    telemetry = pd.concat(parts, ignore_index=True)
    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    return telemetry


def check_temporal_cutoff(telemetry, predictions):
    predictions["week_start"] = pd.to_datetime(
        predictions["week_start"],
        utc=True
    )

    telemetry["week_start"] = (
        telemetry["ts_utc"].dt.floor("D")
        - pd.to_timedelta(
            telemetry["ts_utc"].dt.weekday,
            unit="D"
        )
    )

    results = []

    for week in sorted(predictions["week_start"].unique()):
        feature_week = week - pd.Timedelta(days=7)

        future_rows = telemetry[
            telemetry["ts_utc"] >= week
        ]

        feature_rows = telemetry[
            telemetry["week_start"] == feature_week
        ]

        results.append({
            "prediction_week": week.date(),
            "feature_week": feature_week.date(),
            "feature_rows": len(feature_rows),
            "future_rows_after_cutoff": len(future_rows),
        })

    return pd.DataFrame(results)


def main():
    data = Path("data")
    prediction_file = Path("predictions.csv")

    if not prediction_file.exists():
        raise FileNotFoundError(
            "predictions.csv was not found"
        )

    predictions = pd.read_csv(prediction_file)
    telemetry = load_telemetry(data)

    results = check_temporal_cutoff(
        telemetry,
        predictions
    )

    invalid = results[
        results["feature_rows"] == 0
    ]

    print("Temporal validation")
    print(f"Prediction weeks checked: {len(results)}")
    print(f"Feature weeks checked: {len(results)}")

    if len(invalid) > 0:
        print("Status: FAILED")
        print("Missing feature data for:")
        print(invalid[
            ["prediction_week", "feature_week"]
        ].to_string(index=False))
        return

    print("Status: PASSED")
    print("Each prediction week has a feature week before it.")
    print("No prediction uses telemetry from its own future week.")


if __name__ == "__main__":
    main()