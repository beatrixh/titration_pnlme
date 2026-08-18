from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import pandas as pd

PROJECTS = {
    "4PL": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/4PL_edge_effects"),
    "5PL": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/5PL_edge_effects"),
    "4PL_pure_rlu": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/4PL_pure_rlu"),
    "5PL_pure_rlu": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/5PL_pure_rlu"),
    "4PL_pure_rlu_v2": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/4PL_pure_rlu_v2"),
    "5PL_pure_rlu_v2": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/5PL_pure_rlu_v2"),
    "4PL_pure_rlu_larger_data": Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/4PL_pure_rlu_larger_data"),
}

EFFECT_LABEL = {
    "Random effect": "random",
    "mab_virus fixed effect": "mab_virus",
    "goes_down fixed effect": "goes_down",
    "run_id fixed effect": "run_id",
}


def read_model_tracker(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active

    header = [c.value for c in ws[1]]
    model_cols = {
        idx: name for idx, name in enumerate(header, start=1)
        if name and idx > 2
    }

    rows: dict[str, dict[str, str]] = {name: {} for name in model_cols.values()}
    current_param = None
    for row in ws.iter_rows(min_row=3):
        if row[0].value:
            current_param = row[0].value
        effect = row[1].value
        if effect is None or current_param is None:
            continue
        toggle_name = f"{current_param}_{EFFECT_LABEL.get(effect, effect)}"
        for col_idx, model_name in model_cols.items():
            rows[model_name][toggle_name] = row[col_idx - 1].value

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    return df.reset_index()


def discover_models(models_dir: Path) -> list[str]:
    names = [
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "LogLikelihood" / "logLikelihood.txt").exists()
    ]
    return sorted(names, key=lambda n: int(n[1:]) if n[1:].isdigit() else 10**9)


def read_criteria(model_name: str, models_dir: Path) -> dict:
    path = models_dir / model_name / "LogLikelihood" / "logLikelihood.txt"
    if not path.exists():
        return {"model": model_name, "AIC": None, "BICc": None}
    values = pd.read_csv(path).set_index("criteria")["importanceSampling"]
    return {
        "model": model_name,
        "AIC": values.get("AIC"),
        "BICc": values.get("BICc"),
    }


def build_report(project: str) -> pd.DataFrame:
    root = PROJECTS[project]
    models_dir = root / "model_files"
    tracker_path = root / "model_tracker.xlsx"

    model_names = discover_models(models_dir)
    if not model_names:
        raise SystemExit(f"no completed models (with LogLikelihood/logLikelihood.txt) found in {models_dir}")

    toggles = read_model_tracker(tracker_path)
    criteria = pd.DataFrame([read_criteria(m, models_dir) for m in model_names])

    missing = sorted(set(model_names) - set(toggles["model"]))
    if missing:
        print(f"warning: model(s) not found in tracker: {', '.join(missing)}")

    result = criteria.merge(toggles, on="model", how="left")
    toggle_cols = [c for c in toggles.columns if c != "model"]
    result = result[["model"] + toggle_cols + ["AIC", "BICc"]]
    return result.sort_values("BICc", na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report AIC/BICc across all completed models for a project.")
    parser.add_argument("project", choices=list(PROJECTS), help="which project to report on")
    args = parser.parse_args()

    result = build_report(args.project)
    print(result.to_string(index=False))

    out_path = PROJECTS[args.project] / f"{args.project.lower()}_likelihood_report.csv"
    result.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
