import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


weeks = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D",
    tz="UTC"
)

review_date = pd.Timestamp(
    "2026-02-16",
    tz="UTC"
)


def norm_id(x):
    if pd.isna(x):
        return None

    return (
        str(x)
        .replace(":", "")
        .replace("-", "")
        .replace(" ", "")
        .upper()
    )


def load_data(data_path):
    files = sorted(
        (data_path / "telemetry").rglob("*.parquet")
    )

    if not files:
        raise FileNotFoundError("No telemetry files found")

    cols = [
        "gateway_id",
        "ts_utc",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt"
    ]

    parts = []

    for file in files:
        try:
            df = pd.read_parquet(file, columns=cols)
            parts.append(df)
        except Exception:
            pass

    if not parts:
        raise RuntimeError("Could not read telemetry files")

    df = pd.concat(parts, ignore_index=True)

    df["gateway_norm"] = df["gateway_id"].apply(norm_id)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    for col in cols[2:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    return df


def make_weekly(df):
    df = df.copy()

    df["week_start"] = (
        df["ts_utc"].dt.floor("D")
        - pd.to_timedelta(
            df["ts_utc"].dt.weekday,
            unit="D"
        )
    )

    g = df.groupby(
        ["gateway_norm", "week_start"],
        sort=False
    )

    weekly = g.agg(
        offline_mean=("offline_duration_sec", "mean"),
        offline_std=("offline_duration_sec", "std"),
        offline_max=("offline_duration_sec", "max"),
        offline_sum=("offline_duration_sec", "sum"),
        disconnect_mean=("disconnection_cnt", "mean"),
        disconnect_std=("disconnection_cnt", "std"),
        disconnect_max=("disconnection_cnt", "max"),
        disconnect_sum=("disconnection_cnt", "sum"),
        reboot_mean=("reboot_cnt", "mean"),
        reboot_std=("reboot_cnt", "std"),
        reboot_max=("reboot_cnt", "max"),
        reboot_sum=("reboot_cnt", "sum"),
        observations=("offline_duration_sec", "size"),
        active_days=(
            "ts_utc",
            lambda x: x.dt.floor("D").nunique()
        ),
        days_with_disconnect=(
            "disconnection_cnt",
            lambda x: (
                (x > 0)
                .groupby(
                    df.loc[
                        x.index,
                        "ts_utc"
                    ].dt.floor("D")
                )
                .any()
                .sum()
            )
        ),
        days_with_reboot=(
            "reboot_cnt",
            lambda x: (
                (x > 0)
                .groupby(
                    df.loc[
                        x.index,
                        "ts_utc"
                    ].dt.floor("D")
                )
                .any()
                .sum()
            )
        )
    ).reset_index()

    weekly["offline_std"] = weekly["offline_std"].fillna(0)
    weekly["disconnect_std"] = weekly["disconnect_std"].fillna(0)
    weekly["reboot_std"] = weekly["reboot_std"].fillna(0)

    weekly = weekly.sort_values(
        ["gateway_norm", "week_start"]
    )

    for col in [
        "offline_mean",
        "disconnect_mean",
        "reboot_mean"
    ]:
        weekly[f"{col}_prev1"] = (
            weekly.groupby("gateway_norm")[col].shift(1)
        )

        weekly[f"{col}_trend"] = (
            weekly[col]
            - weekly[f"{col}_prev1"]
        )

        weekly[f"{col}_trend_ratio"] = (
            weekly[col]
            / (
                weekly[f"{col}_prev1"].abs()
                + 1e-6
            )
        )

        weekly[f"{col}_4w_mean"] = (
            weekly.groupby("gateway_norm")[col]
            .transform(
                lambda x: (
                    x.shift(1)
                    .rolling(4, min_periods=2)
                    .mean()
                )
            )
        )

    weekly["severity"] = (
        np.log1p(weekly["offline_sum"])
        + 2 * np.log1p(weekly["disconnect_sum"])
        + np.log1p(weekly["reboot_sum"])
        + 0.5 * weekly["days_with_disconnect"]
        + 0.25 * weekly["days_with_reboot"]
    )

    weekly["severity_prev"] = (
        weekly.groupby("gateway_norm")["severity"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(4, min_periods=2)
                .mean()
            )
        )
    )

    weekly["severity_delta"] = (
        weekly["severity"]
        - weekly["severity_prev"]
    )

    return weekly


def get_features(df):
    skip = {
        "gateway_norm",
        "week_start",
        "severity"
    }

    return [
        col
        for col in df.columns
        if col not in skip
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_review(data_path):
    review = pd.read_excel(
        data_path / "engineer_review_2026-02.xlsx"
    )

    review["gateway_norm"] = (
        review["gateway_id"].apply(norm_id)
    )

    review["label"] = (
        review["Kategorie"] == "Schlecht"
    ).astype(int)

    return review[
        ["gateway_norm", "label"]
    ].drop_duplicates("gateway_norm")


def train(train, features):
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        train[features],
        train["label"]
    )

    return model


def risk_score(df):
    def rank(x):
        return x.rank(
            pct=True,
            method="average"
        )

    score = (
        0.40 * rank(np.log1p(df["offline_sum"]))
        + 0.35 * rank(np.log1p(df["disconnect_sum"]))
        + 0.10 * rank(np.log1p(df["reboot_sum"]))
        + 0.10 * rank(df["days_with_disconnect"])
        + 0.05 * rank(df["days_with_reboot"])
    )

    delta = df["severity_delta"].fillna(0)

    score = (
        0.85 * score
        + 0.15 * rank(delta)
    )

    return score


def load_master(data_path):
    master = pd.read_csv(
        data_path / "gateway_master.csv",
        encoding="latin1"
    )

    master["gateway_norm"] = (
        master["gateway_id"].apply(norm_id)
    )

    return master[
        ["gateway_norm", "gateway_id"]
    ].drop_duplicates("gateway_norm")
    
def get_reason(current):
    disconnect_rank = current["disconnect_sum"].rank(pct=True)
    offline_rank = current["offline_sum"].rank(pct=True)
    reboot_rank = current["reboot_sum"].rank(pct=True)

    reasons = []

    for i in current.index:
        values = {
            "disconnections": disconnect_rank.loc[i],
            "offline": offline_rank.loc[i],
            "reboots": reboot_rank.loc[i]
        }

        strongest = max(values, key=values.get)

        if strongest == "disconnections":
            reasons.append("high recent disconnections")
        elif strongest == "offline":
            reasons.append("high recent offline duration")
        else:
            reasons.append("frequent recent reboots")

    return reasons


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data")
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("predictions_ml_v2.csv")
    )

    args = parser.parse_args()

    print("Loading telemetry...")

    data = load_data(args.data)

    print(f"Telemetry rows: {len(data):,}")

    weekly = make_weekly(data)
    features = get_features(weekly)

    print(f"Gateway-week rows: {len(weekly):,}")
    print(f"Features: {len(features)}")

    review = load_review(args.data)
    master = load_master(args.data)

    results = []

    for week in weeks:
        feature_week = week - pd.Timedelta(days=7)

        current = weekly[
            weekly["week_start"] == feature_week
        ].copy()

        if current.empty:
            raise RuntimeError(
                f"No data for {week.date()}"
            )

        if week < review_date:
            current["score"] = risk_score(current)
            method = "telemetry_only"

        else:
            train_data = weekly.merge(
                review,
                on="gateway_norm",
                how="inner"
            )

            train_data = train_data[
                train_data["week_start"] < week
            ].copy()

            if train_data["label"].nunique() < 2:
                current["score"] = risk_score(current)
                method = "telemetry_fallback"

            else:
                model = train(
                    train_data,
                    features
                )

                ml_score = model.predict_proba(
                    current[features]
                )[:, 1]

                telemetry_score = risk_score(
                    current
                )

                current["score"] = (
                    0.85 * ml_score
                    + 0.15 * telemetry_score
                )

                method = (
                    "review_supervised_plus_telemetry"
                )

        current = current.merge(
            master,
            on="gateway_norm",
            how="left"
        )

        current["gateway_id"] = (
            current["gateway_id"]
            .fillna(current["gateway_norm"])
        )

        current["reason"] = get_reason(current)

        top = current.sort_values(
            ["score", "gateway_norm"],
            ascending=[False, True]
        ).head(15).copy()

        top["week_start"] = week.strftime(
            "%Y-%m-%d"
        )

        top["rank"] = range(
            1,
            len(top) + 1
        )

        top["score"] = top["score"].round(6)

        top["method"] = method

        results.append(
            top[
                [
                    "week_start",
                    "rank",
                    "gateway_id",
                    "score",
                    "reason",
                    "method"
                ]
            ]
        )

        print(
            f"{week.date()}: "
            f"{method}, selected={len(top)}"
        )

    result = pd.concat(
        results,
        ignore_index=True
    )

    result[
        [
            "week_start",
            "rank",
            "gateway_id",
            "score",
            "reason"
        ]
    ].to_csv(
        args.out,
        index=False
    )

    print(
        f"\nWrote {args.out} — "
        f"{len(result)} rows"
    )


if __name__ == "__main__":
    main()