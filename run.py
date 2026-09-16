import subprocess
import sys
from pathlib import Path


def run(command):
    print()
    print("Running:", " ".join(command))
    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}")


def main():
    root = Path(__file__).resolve().parent
    data = root / "data"
    python = sys.executable
    predictions = root / "predictions.csv"

    run([
        python,
        str(root / "src" / "generate_predictions.py"),
        "--data",
        str(data),
        "--out",
        str(predictions)
    ])

    run([
        python,
        str(root / "src" / "validate_submission.py"),
        str(predictions)
    ])

    run([
        python,
        str(root / "src" / "validate_temporal.py")
    ])

    run([
        python,
        str(root / "src" / "validate_unseen_gateways.py")
    ])

    run([
        python,
        str(root / "src" / "validate_network_shift.py")
    ])

    print()
    print("NEXORA pipeline completed")
    print("Final output: predictions.csv")


if __name__ == "__main__":
    main()