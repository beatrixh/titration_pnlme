from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS, build_report
from generate_pdf_model_report import (
    logistic_4pl, logistic_5pl, icp_4pl, icp_5pl, plot_design_matrix, plot_param_boxplots,
)

# Any registered project except the rlu_norm-scale edge_effects ones (short
# keys "4PL"/"5PL") is a pure_rlu-family project this script can handle --
# so 5PL_pure_rlu_v2 etc. become available automatically once registered in
# report_likelihood.PROJECTS, with no change needed here.
PURE_RLU_PROJECTS = [p for p in PROJECTS if p not in ("4PL", "5PL")]

PLATE_ROWS = 9
PLATE_COLS = 13


def dist_edge(plate_row: pd.Series, plate_col: pd.Series) -> pd.Series:
    return np.minimum(
        np.minimum(plate_row, PLATE_ROWS - plate_row),
        np.minimum(plate_col, PLATE_COLS - plate_col),
    )


def project_family(project: str) -> str:
    """Strip a leading 4PL_/5PL_ so e.g. '4PL_pure_rlu_v2' and
    '5PL_pure_rlu_v2' group together, but not with plain 'pure_rlu' (v1)."""
    for prefix in ("4PL_", "5PL_"):
        if project.startswith(prefix):
            return project[len(prefix):]
    return project


def resolve_data_csv(fitted_mlxtran: Path) -> Path:
    """Read the data file path directly out of a completed model's own saved
    .mlxtran, rather than maintaining a separate hardcoded constant per
    project -- avoids exactly the kind of drift that caused the original
    wrong-input-file mixup this project already ran into once."""
    text = fitted_mlxtran.read_text()
    m = re.search(r"file=\{path='([^']*)'\}", text)
    if not m:
        raise SystemExit(f"could not find data file path in {fitted_mlxtran}")
    raw_path = m.group(1)
    if re.match(r"^([A-Za-z]:[\\/]|/)", raw_path):
        return Path(raw_path)
    return (fitted_mlxtran.parent / raw_path).resolve()


# ---------------------------------------------------------------------------
# Data prep
#
# These projects fit raw `rlu` directly (no pre-normalized rlu_norm/cc/vc
# renormalization step like the edge_effects projects). U is derived rather
# than directly estimated: U = mean_vc * (1 - edge), where
# edge = alpha*exp(-k*dist_edge). The 'v2' variants add an estimated
# U_offset individual parameter, subtracted as U = mean_vc*(1-edge) -
# exp(U_offset) (detected automatically from whether estimatedIndividual
# Parameters.txt has a U_offset_mode column -- has_u_offset below). 5PL
# additionally has a shape parameter s, applied as an exponent on the
# logistic term: f = virus_cc*(L + (U-L)/(1+exp(...))**s).
# ---------------------------------------------------------------------------

def build_corrected_data(data: pd.DataFrame, params: pd.DataFrame, pl: str, has_u_offset: bool) -> pd.DataFrame:
    """Per-observation plate-edge correction.

    For each 'sample'/'vc' row (virus_cc==1), compute the model's own
    predicted f_edge at that row's actual (edge, virus_cc) using that
    individual's fitted L/m/e/(s)/alpha/k/(U_offset), take the residual
    against the observed rlu, then add that residual onto the edge=0
    ('canonical', plate-center) prediction. This is an exact per-row
    detrend of the plate-edge nuisance effect (not a same-scale
    approximation), since edge enters the model both as an overall
    multiplier and inside U, so a naive rlu / (1 - edge) correction
    doesn't exactly invert it.

    'cc' rows (virus_cc==0, background wells) pass through unmodified --
    they aren't part of the fitted dose-response curve.
    """
    cols = ["id", "L_mode", "m_mode", "e_mode", "alpha_mode", "k_mode"]
    if pl == "5PL":
        cols.append("s_mode")
    if has_u_offset:
        cols.append("U_offset_mode")
    merged = data.merge(params[cols], left_on="monolix_id", right_on="id", how="inner")

    merged["dist_edge"] = dist_edge(merged.plate_row, merged.plate_col)
    base = 1 / (1 + np.exp(merged.m_mode * (np.log(merged.concentration) + np.log(merged.e_mode))))
    D = base ** merged.s_mode if pl == "5PL" else base
    edge = merged.alpha_mode * np.exp(-merged.k_mode * merged.dist_edge)

    offset = np.exp(merged.U_offset_mode) if has_u_offset else 0
    U_row = merged.mean_vc * (1 - edge) - offset
    U_center = merged.mean_vc - offset

    f_edge_pred = merged.virus_cc * (1 - edge) * (merged.L_mode + D * (U_row - merged.L_mode))
    f_center_pred = merged.L_mode + D * (U_center - merged.L_mode)
    residual = merged.rlu - f_edge_pred

    merged["rlu_edge_corrected"] = np.where(
        merged.specrole == "cc", merged.rlu, f_center_pred + residual
    )
    keep = [
        "run_id", "mab_virus", "plate_row", "plate_col", "specrole",
        "concentration", "rlu", "rlu_edge_corrected",
    ]
    return merged[keep]


def calc_params_ics(data: pd.DataFrame, params: pd.DataFrame, pl: str, has_u_offset: bool) -> pd.DataFrame:
    """Canonical (plate-center, edge=0) per-mab_virus curve + IC50/IC80.

    Per the requested convention: normalize between this curve's own fitted
    U (= mean_vc, i.e. the upper asymptote with no edge effect, minus
    exp(U_offset) for the v2 variants) and L, then neutralization =
    1 - normalized(rlu); IC50/80 = concentration where neutralization hits
    0.5/0.8. Normalizing a curve by its own asymptotes makes U=1, L=0
    exactly, so icp_4pl/icp_5pl (imported unchanged from
    generate_pdf_model_report) apply directly. L_raw/U_raw are kept around
    for annotating the real fitted magnitudes and for normalizing the
    scatter data in plot_fit_grid.
    """
    mean_vc_by_run = data[["run_id", "mean_vc"]].drop_duplicates()
    p = params.merge(mean_vc_by_run, on="run_id", how="left")

    grouped = p.groupby("mab_virus")
    values = {
        "mab_virus": p.mab_virus,
        "e": grouped.e_mode.transform(lambda x: np.exp(np.mean(np.log(x)))),
        "m": grouped.m_mode.transform("mean"),
        "L_raw": grouped.L_mode.transform("mean"),
    }
    if has_u_offset:
        values["U_raw"] = grouped.mean_vc.transform("mean") - np.exp(
            grouped.U_offset_mode.transform("mean")
        )
    else:
        values["U_raw"] = grouped.mean_vc.transform("mean")
    if pl == "5PL":
        values["s"] = grouped.s_mode.transform("mean")

    mab_virus_params = pd.DataFrame(values).drop_duplicates().reset_index(drop=True)
    mab_virus_params["U"] = 1.0
    mab_virus_params["L"] = 0.0

    if pl == "5PL":
        mab_virus_params["ic50"] = mab_virus_params.apply(
            lambda row: icp_5pl(0.5, row.U, row.L, row.m, row.e, row.s), axis=1
        )
        mab_virus_params["ic80"] = mab_virus_params.apply(
            lambda row: icp_5pl(0.8, row.U, row.L, row.m, row.e, row.s), axis=1
        )
    else:
        mab_virus_params["ic50"] = mab_virus_params.apply(
            lambda row: icp_4pl(0.5, row.U, row.L, row.m, row.e), axis=1
        )
        mab_virus_params["ic80"] = mab_virus_params.apply(
            lambda row: icp_4pl(0.8, row.U, row.L, row.m, row.e), axis=1
        )
    return mab_virus_params


def summarize_raw_params(params: pd.DataFrame, pl: str) -> pd.DataFrame:
    """Per-mab_virus fitted parameter values, uncorrected -- used for the boxplot."""
    p = params.copy()
    p["e_mode_geommean"] = p.groupby("mab_virus").e_mode.transform(lambda x: np.exp(np.mean(np.log(x))))
    cols = ["mab_virus", "L_mode", "m_mode", "e_mode_geommean"]
    if pl == "5PL":
        cols.append("s_mode")
    return p[cols].drop_duplicates()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_toggle_table(report_row: pd.Series, pl: str, has_u_offset: bool) -> pd.DataFrame:
    params = (["U_offset"] if has_u_offset else []) + ["L", "m", "e"] + (["s"] if pl == "5PL" else []) + ["alpha", "k"]
    rows = []
    for p in params:
        row = {
            "parameter": p,
            "random effect": report_row.get(f"{p}_random", ""),
            "goes_down effect": report_row.get(f"{p}_goes_down", ""),
        }
        if f"{p}_mab_virus" in report_row.index:
            row["mab_virus effect"] = report_row.get(f"{p}_mab_virus", "")
        if f"{p}_run_id" in report_row.index:
            row["run_id effect"] = report_row.get(f"{p}_run_id", "")
        rows.append(row)

    df = pd.DataFrame(rows).fillna("")
    col_order = ["parameter", "random effect", "mab_virus effect", "run_id effect", "goes_down effect"]
    return df[[c for c in col_order if c in df.columns]]


def plot_info_page(model_label: str, report_row: pd.Series, pl: str, has_u_offset: bool) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, model_label, ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.915, f"BICc = {report_row['BICc']:.2f}    AIC = {report_row['AIC']:.2f}", ha="center", fontsize=12)

    table_df = build_toggle_table(report_row, pl, has_u_offset)
    ax_table = fig.add_axes((0.08, 0.55, 0.84, 0.3))
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_df.drop(columns="parameter").values,
        rowLabels=table_df["parameter"].values,
        colLabels=[c for c in table_df.columns if c != "parameter"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    if has_u_offset:
        note = (
            "Note: U itself is still not directly estimated -- it is derived as\n"
            "U = mean_vc * (1 - edge) - exp(U_offset), using the U_offset row above."
        )
    else:
        note = (
            "Note: U is not an estimated parameter in this model -- it is derived\n"
            "deterministically as U = mean_vc * (1 - edge), so it has no toggle row here."
        )
    fig.text(0.5, 0.5, note, ha="center", fontsize=9, style="italic")
    return fig


def plot_fit_grid(plot_params: pd.DataFrame, corrected_data: pd.DataFrame, pl: str) -> plt.Figure:
    n = plot_params.mab_virus.nunique()
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()
    t = np.exp(np.linspace(np.log(1e-4), np.log(5e1), 10_000))

    for ax, (_, row) in zip(axes, plot_params.iterrows()):
        mv = row.mab_virus
        dat = corrected_data.loc[
            (corrected_data.mab_virus == mv) & (corrected_data.specrole != "cc")
        ].copy()
        if pl == "5PL":
            y = logistic_5pl(t, row.U, row.L, row.m, row.e, row.s)
        else:
            y = logistic_4pl(t, row.U, row.L, row.m, row.e)
        ax.plot(t, 1 - y, linewidth=2, color="red")

        for val, level in ((row.ic50, 0.5), (row.ic80, 0.8)):
            ax.plot([0, val], [level, level], color="gray", ls="--", linewidth=1)
            ax.plot([val, val], [0, level], color="gray", ls="--", linewidth=1)

        dat["normalized"] = (dat.rlu_edge_corrected - row.L_raw) / (row.U_raw - row.L_raw)
        rlus = dat.pivot(index=["run_id", "plate_col"], columns="plate_row", values="normalized")
        conc = dat.pivot(index=["run_id", "plate_col"], columns="plate_row", values="concentration")
        for idx in rlus.index:
            ax.scatter(
                conc.loc[idx].values, 1 - rlus.loc[idx].values, s=40, alpha=0.3, color="cornflowerblue"
            )

        annotation_lines = [
            f"L = {row.L_raw:.0f}",
            f"U = {row.U_raw:.0f}",
            f"h = {row.m:.2f}",
            f"f = {row.e:.2g}",
        ]
        if pl == "5PL":
            annotation_lines.append(f"s = {row.s:.2f}")
        ax.text(
            0.98, 0.02, "\n".join(annotation_lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
        )
        ax.set_xscale("log")
        ax.set_title(mv, fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Model fits by mab_virus (edge-corrected, normalized to this curve's own U/L)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report for a best-fit pure_rlu model.")
    parser.add_argument("project", choices=PURE_RLU_PROJECTS, help="which pure_rlu project")
    parser.add_argument("model", help='model name, e.g. "m64"')
    parser.add_argument(
        "--data-csv", type=Path, default=None,
        help="overrides the data file this project's own saved .mlxtran points to",
    )
    parser.add_argument(
        "--report-csv", type=Path, default=None,
        help="likelihood report CSV to use for BICc/AIC and the design-matrix page; "
             "defaults to a fresh report combining every registered project in the same "
             "family (e.g. 4PL_pure_rlu + 5PL_pure_rlu, or 4PL_pure_rlu_v2 alone until "
             "5PL_pure_rlu_v2 is registered) -- projects in the same family fit the same "
             "raw-rlu scale and data, so their BICc is directly comparable",
    )
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    project = args.project
    pl = "5PL" if project.startswith("5PL") else "4PL"

    if args.report_csv is not None:
        report = pd.read_csv(args.report_csv)
        if "project" not in report.columns:
            report.insert(0, "project", project)
    else:
        family = project_family(project)
        reports = []
        for proj in PURE_RLU_PROJECTS:
            if project_family(proj) != family:
                continue
            try:
                r = build_report(proj)
            except SystemExit:
                continue  # registered but not run yet
            r.insert(0, "project", proj)
            reports.append(r)
        report = pd.concat(reports, ignore_index=True, sort=False)

    row_match = report.loc[(report.project == project) & (report.model == args.model)]
    if row_match.empty:
        raise SystemExit(f"no row for project={project} model={args.model}")
    report_row = row_match.iloc[0]

    model_dir = PROJECTS[project] / "model_files" / args.model
    params = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    has_u_offset = "U_offset_mode" in params.columns

    data_csv = args.data_csv or resolve_data_csv(model_dir / f"{args.model}_fitted.mlxtran")
    data = pd.read_csv(data_csv)

    raw_params = summarize_raw_params(params, pl)
    plot_params = calc_params_ics(data, params, pl, has_u_offset)
    corrected_data = build_corrected_data(data, params, pl, has_u_offset)

    output = args.output or Path(f"{project}_{args.model}_report.pdf")

    with PdfPages(output) as pdf:
        for fig in (
            plot_info_page(f"{project} {args.model}", report_row, pl, has_u_offset),
            plot_design_matrix(report, f"{project} {args.model}"),
            plot_param_boxplots(raw_params, pl),
            plot_fit_grid(plot_params, corrected_data, pl),
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
