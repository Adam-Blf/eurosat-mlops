"""python -m eurosat_mlops.report writes reports/rapport-eurosat-mlops.pdf."""

from __future__ import annotations

import argparse
from pathlib import Path

from eurosat_mlops.report.build import OUTPUT, build_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT, help="PDF path to write")
    print(build_pdf(parser.parse_args().out))


if __name__ == "__main__":
    main()
