from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from report_likelihood import PROJECTS, build_report

OUT_PATH = Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/combined_likelihood_report.csv")


def project_datafile(project: str) -> str:
    """The observation datafile a project is fit to, read from its template
    (model_files/m0.mlxtran, else the lowest-numbered m*.mlxtran). AIC/BICc are
    only comparable across projects sharing a datafile *and* observation
    structure, so surfacing this as a column keeps the combined report honest.
    """
    mf = PROJECTS[project] / "model_files"
    candidates = [mf / "m0.mlxtran"] + sorted(
        (p for p in mf.glob("m*.mlxtran") if re.fullmatch(r"m\d+\.mlxtran", p.name)),
        key=lambda p: int(p.stem[1:]),
    )
    for tmpl in candidates:
        if not tmpl.exists():
            continue
        m = re.search(r"file\s*=\s*\{path='([^']*)'\}", tmpl.read_text())
        if m:
            return Path(m.group(1)).name
    return ""


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
        try:
            report = build_report(project)
        except SystemExit as e:
            # No completed models yet for this project -- skip it rather than
            # aborting the whole combine (common mid-run, e.g. one arm of a
            # 4PL/5PL pair has started fitting and the other hasn't).
            print(f"skipping {project}: {e}")
            continue
        report.insert(0, "project", project)
        report.insert(1, "datafile", project_datafile(project))
        reports.append(report)

    if not reports:
        raise SystemExit("no projects had completed models")

    # pd.concat aligns on column name and fills any column missing from one
    # side (e.g. s_random/s_mab_virus/s_goes_down, which only 5PL has) with
    # NaN for the other project's rows.
    combined = pd.concat(reports, ignore_index=True, sort=False)

    front = ["project", "datafile", "model"]
    toggle_cols = [c for c in combined.columns if c not in front + ["AIC", "BICc"]]
    combined = combined[front + toggle_cols + ["AIC", "BICc"]]
    combined = combined.sort_values("BICc", na_position="last").reset_index(drop=True)

    print(combined.to_string(index=False))
    combined.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
