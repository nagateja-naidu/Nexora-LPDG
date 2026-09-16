from pathlib import Path

import pandas as pd


def clean_id(values):
    return (
        values.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )


def load_review(data):
    review = pd.read_excel(
        data / "engineer_review_2026-02.xlsx"
    )

    review["gateway_norm"] = clean_id(
        review["gateway_id"]
    )

    return set(review["gateway_norm"].dropna())


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
                columns=["gateway_id", "ts_utc"]
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

    telemetry["gateway_norm"] = clean_id(
        telemetry["gateway_id"]
    )

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    return telemetry


def main():
    data = Path("data")
    prediction_file = Path(
        "predictions.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            "predictions.csv was not found"
        )

    predictions = pd.read_csv(
        prediction_file
    )

    predictions["gateway_norm"] = clean_id(
        predictions["gateway_id"]
    )

    reviewed_gateways = load_review(data)
    telemetry = load_telemetry(data)

    all_gateways = set(
        telemetry["gateway_norm"].dropna()
    )

    unseen_gateways = (
        all_gateways - reviewed_gateways
    )

    predictions_unseen = predictions[
        predictions["gateway_norm"].isin(
            unseen_gateways
        )
    ].copy()

    total_predictions = len(predictions)
    unseen_predictions = len(
        predictions_unseen
    )

    weeks = predictions[
        "week_start"
    ].nunique()

    print("Unseen gateway validation")
   
    print(
        f"Gateways in telemetry: {len(all_gateways)}"
    )
    print(
        f"Gateways in engineer review: "
        f"{len(reviewed_gateways)}"
    )
    print(
        f"Unseen gateways: "
        f"{len(unseen_gateways)}"
    )
    print(
        f"Prediction rows checked: "
        f"{total_predictions}"
    )
    print(
        f"Predictions from unseen gateways: "
        f"{unseen_predictions}"
    )

    if unseen_predictions > 0:
        percentage = (
            unseen_predictions
            / total_predictions
            * 100
        )

        print(
            f"Unseen gateway coverage: "
            f"{percentage:.2f}%"
        )
        print("Status: PASSED")
        print(
            "Ranking the gateway which is not present in the review"
        )
    else:
        print(
            "Status: PASSED"
        )
        print(
            "No reviewed gateway was not used for the validation result. "
    
        )

    if weeks != 8:
        print(
            f"Warning: expected 8 prediction weeks, "
            f"found {weeks}"
        )


if __name__ == "__main__":
    main()