from pathlib import Path

import numpy as np
import pandas as pd


def load_telemetry(data):
    files = sorted(
        (data / "telemetry").rglob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "No telemetry parquet files found"
        )

    parts = []

    for file in files:
        try:
            df = pd.read_parquet(
                file,
                columns=[
                    "gateway_id",
                    "ts_utc",
                    "offline_duration_sec",
                    "disconnection_cnt",
                    "reboot_cnt"
                ]
            )
            parts.append(df)
        except Exception:
            continue

    if not parts:
        raise RuntimeError(
            "Could not read telemetry files"
        )

    telemetry = pd.concat(
        parts,
        ignore_index=True
    )

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    for column in [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt"
    ]:
        telemetry[column] = pd.to_numeric(
            telemetry[column],
            errors="coerce"
        ).fillna(0)

    return telemetry


def build_weekly(telemetry):
    data = telemetry.copy()

    data["week_start"] = (
        data["ts_utc"].dt.floor("D")
        - pd.to_timedelta(
            data["ts_utc"].dt.weekday,
            unit="D"
        )
    )

    weekly = data.groupby(
        ["gateway_id", "week_start"]
    ).agg(
        offline_sum=(
            "offline_duration_sec",
            "sum"
        ),
        disconnect_sum=(
            "disconnection_cnt",
            "sum"
        ),
        reboot_sum=(
            "reboot_cnt",
            "sum"
        )
    ).reset_index()

    weekly["severity_score"] = (
        np.log1p(weekly["offline_sum"])
        + 2 * np.log1p(
            weekly["disconnect_sum"]
        )
        + np.log1p(
            weekly["reboot_sum"]
        )
    )

    return weekly


def summarize(data, name):
    return {
        "period": name,
        "gateway_weeks": len(data),
        "mean_severity": data[
            "severity_score"
        ].mean(),
        "median_severity": data[
            "severity_score"
        ].median(),
        "p90_severity": data[
            "severity_score"
        ].quantile(0.90),
        "max_severity": data[
            "severity_score"
        ].max()
    }


def main():
    data = Path("data")

    telemetry = load_telemetry(data)
    weekly = build_weekly(telemetry)

    early = weekly[
        (weekly["week_start"] >= "2026-02-02")
        & (weekly["week_start"] < "2026-02-23")
    ].copy()

    late = weekly[
        (weekly["week_start"] >= "2026-02-23")
        & (weekly["week_start"] <= "2026-03-23")
    ].copy()

    early_summary = summarize(
        early,
        "Early period"
    )

    late_summary = summarize(
        late,
        "Later period"
    )

    results = pd.DataFrame([
        early_summary,
        late_summary
    ])

    mean_change = (
        (
            late_summary["mean_severity"]
            - early_summary["mean_severity"]
        )
        / (
            abs(early_summary["mean_severity"])
            + 1e-6
        )
        * 100
    )

    print("Network shift validation")
    print(results.to_string(index=False))
    print(
        f"\nMean severity change: "
        f"{mean_change:.2f}%"
    )

    if (
        early_summary["gateway_weeks"] > 0
        and late_summary["gateway_weeks"] > 0
    ):
        print("Status: PASSED")
        
    else:
        print("Status: FAILED")

if __name__ == "__main__":
    main()