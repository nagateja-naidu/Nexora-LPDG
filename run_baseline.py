import subprocess
import sys
from pathlib import Path


def run(command):
    print()
    print("Running:", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}"
        )


def main():
    root = Path(__file__).resolve().parent
    python = sys.executable
    data = root / "data"

    run([
        python,
        str(root / "src" / "baseline_3sigma.py"),
        "--data",
        str(data)
    ])


if __name__ == "__main__":
    main()