from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from report_likelihood import PROJECTS, build_report

OUT_PATH = Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/combined_likelihood_report.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-project likelihood reports into one CSV, sorted by BICc."
    )
    parser.add_argument(
        "--projects", nargs="+", choices=list(PROJECTS), default=list(PROJECTS),
        help="which projects to combine (default: all registered projects). "
             "Note: BICc is only directly comparable across projects fit to the "
             "same observation scale -- e.g. 4PL_pure_rlu/5PL_pure_rlu (raw rlu) "
             "are comparable to each other but not to 4PL/5PL (rlu_norm) without "
             "a Jacobian correction.",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    reports = []
    for project in args.projects:
        report = build_report(project)
        report.insert(0, "project", project)
        reports.append(report)

    # pd.concat aligns on column name and fills any column missing from one
    # side (e.g. s_random/s_mab_virus/s_goes_down, which only 5PL has) with
    # NaN for the other project's rows.
    combined = pd.concat(reports, ignore_index=True, sort=False)

    front = ["project", "model"]
    toggle_cols = [c for c in combined.columns if c not in front + ["AIC", "BICc"]]
    combined = combined[front + toggle_cols + ["AIC", "BICc"]]
    combined = combined.sort_values("BICc", na_position="last").reset_index(drop=True)

    print(combined.to_string(index=False))
    combined.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
