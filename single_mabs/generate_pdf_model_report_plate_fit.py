from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from report_likelihood import PROJECTS, build_report

# Any registered project with "plate_fit" in its name -- so 4PL_plate_fit_v6
# etc. become available automatically once registered in
# report_likelihood.PROJECTS, with no change needed here.
PLATE_FIT_PROJECTS = [p for p in PROJECTS if "plate_fit" in p.lower()]

TOGGLE_SUFFIXES = ["random", "mab_virus", "goes_down", "run_id"]

# Preferred left-to-right order for known plate_fit parameters; anything not
# listed here (a future new parameter) is appended alphabetically at the end
# rather than silently dropped.
PARAM_ORDER_HINT = ["U", "L", "L_gap", "h", "f", "alpha", "k"]


def project_family(project: str) -> str:
    """Strip a leading 4PL_/5PL_ so e.g. '4PL_plate_fit_v2' and a
    hypothetical '5PL_plate_fit_v2' would group together, but not with plain
    'plate_fit' (unversioned)."""
    for prefix in ("4PL_", "5PL_"):
        if project.startswith(prefix):
            return project[len(prefix):]
    return project


def resolve_data_csv(model_mlxtran: Path) -> Path:
    """Read the data file path directly out of the model's own generated
    .mlxtran, rather than maintaining a separate hardcoded constant per
    project -- avoids drift between what was actually fit and what a report
    script assumes."""
    text = model_mlxtran.read_text()
    m = re.search(r"file=\{path='([^']*)'\}", text)
    if not m:
        raise SystemExit(f"could not find data file path in {model_mlxtran}")
    raw_path = m.group(1)
    if re.match(r"^([A-Za-z]:[\\/]|/)", raw_path):
        return Path(raw_path)
    return (model_mlxtran.parent / raw_path).resolve()


def discover_params(report_row: pd.Series) -> list[str]:
    """Which model parameters (L, U, h, f, alpha, k, L_gap, ...) this
    particular model has toggle columns for, derived from the report row
    itself rather than hardcoded -- so this keeps working when a project
    adds/drops a parameter (e.g. L_gap only exists from v3 onward) without
    editing this file. NaN-valued columns (present in the combined report
    only because some *other* project in the merge has that column) are
    excluded, so this reflects only params real for this row's project.
    """
    found: set[str] = set()
    for col in report_row.index:
        if col in ("project", "model", "AIC", "BICc"):
            continue
        if pd.isna(report_row[col]):
            continue
        for suf in TOGGLE_SUFFIXES:
            if col.endswith(f"_{suf}"):
                found.add(col[: -(len(suf) + 1)])
                break

    ordered = [p for p in PARAM_ORDER_HINT if p in found]
    ordered += sorted(found - set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# Info page (toggle table) -- no model-equation assumptions, safe to trust.
# ---------------------------------------------------------------------------

def build_toggle_table(report_row: pd.Series, params: list[str]) -> pd.DataFrame:
    rows = []
    for p in params:
        row = {
            "parameter": p,
            "random effect": report_row.get(f"{p}_random", ""),
            "mab_virus effect": report_row.get(f"{p}_mab_virus", ""),
            "run_id effect": report_row.get(f"{p}_run_id", ""),
            "goes_down effect": report_row.get(f"{p}_goes_down", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def plot_info_page(model_label: str, report_row: pd.Series, params: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, model_label, ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.915, f"BICc = {report_row['BICc']:.2f}    AIC = {report_row['AIC']:.2f}", ha="center", fontsize=12)

    table_df = build_toggle_table(report_row, params)
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

    if "L_gap" in params:
        note = (
            "Note: this model's structural equation (see model.txt) uses L_gap as\n"
            "an offset on the upper amplitude: numerator = U - (L + exp(L_gap)),\n"
            "not a floor offset on L directly -- do not reuse the pure_rlu family's\n"
            "L_gap formula for this project, it means something different here."
        )
        fig.text(0.5, 0.5, note, ha="center", fontsize=9, style="italic")

    return fig


# ---------------------------------------------------------------------------
# Design matrix across all completed models -- generic, no equation
# assumptions. Column labels are built directly from the toggle column
# names themselves instead of a hardcoded per-project rename dict (which is
# how generate_pdf_model_report.DESIGN_MATRIX_COL_RENAME does it, and why it
# can't be reused as-is here -- it has no entries for h/f/alpha/k/L_gap).
# ---------------------------------------------------------------------------

SUFFIX_LABEL = {
    "random": "random effect",
    "mab_virus": "mab_virus covariate",
    "run_id": "run_id covariate",
    "goes_down": "goes_down covariate",
}


def humanize_toggle_col(col: str) -> str | None:
    for suf, label in SUFFIX_LABEL.items():
        if col.endswith(f"_{suf}"):
            param = col[: -(len(suf) + 1)]
            return f"{param} ({label})"
    return None


def plot_design_matrix(
    report: pd.DataFrame,
    highlight_label: str,
    score_col: str = "BICc",
    ascending: bool = True,
    max_rows_to_plot: int | None = 40,
) -> plt.Figure:
    """Binary design matrix across all models in the report, ordered by score_col,
    with the model being depicted in this PDF circled in the grid."""
    df = report.copy()
    df["model"] = df["project"] + " " + df["model"]
    df = df.replace({"Yes": 1, "No": 0})

    rename = {c: humanize_toggle_col(c) for c in df.columns if humanize_toggle_col(c)}
    binary_cols_raw = list(rename)
    df = df[["model"] + binary_cols_raw + ["AIC", score_col]].rename(columns=rename)
    binary_cols = [rename[c] for c in binary_cols_raw]

    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    plot_df = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)

    if max_rows_to_plot is not None and len(plot_df) > max_rows_to_plot:
        top = plot_df.head(max_rows_to_plot)
        if highlight_label not in top["model"].values and highlight_label in plot_df["model"].values:
            top = pd.concat([top, plot_df[plot_df.model == highlight_label]])
        plot_df = top.sort_values(score_col, ascending=ascending).reset_index(drop=True)

    X = plot_df[binary_cols]
    row_labels = plot_df["model"].astype(str).tolist()
    score_matrix = plot_df[[score_col]]

    n_rows = len(plot_df)
    n_cols = len(binary_cols)

    fig_height = max(8, min(0.25 * n_rows + 2, 30))
    fig_width = max(12, 0.55 * n_cols + 5)
    max_id_len = max(len(s) for s in row_labels) if row_labels else 5
    left_margin = min(0.45, max(0.20, 0.012 * max_id_len))

    sns.set_theme(style="white", context="notebook")
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[n_cols, 1.4], wspace=0.05)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])

    binary_cmap = ListedColormap(["#f2f2f2", "#1f77b4"])
    sns.heatmap(
        X, ax=ax_heat, cmap=binary_cmap, cbar=False, vmin=0, vmax=1,
        linewidths=0.3, linecolor="white", xticklabels=binary_cols, yticklabels=False,
    )

    y_positions = [i + 0.5 for i in range(len(row_labels))]
    ax_heat.set_yticks(y_positions)
    ax_heat.set_yticklabels(row_labels, rotation=0, fontsize=6, va="center")

    ax_heat.set_title(f"Binary design matrix ordered by {score_col}", pad=12)
    ax_heat.set_xlabel("Toggles", fontsize=11)
    ax_heat.set_ylabel("Model ID")
    ax_heat.tick_params(axis="x", rotation=90, labelsize=9)
    ax_heat.tick_params(axis="y", length=0)

    score_cmap = "viridis_r" if ascending else "viridis"
    sns.heatmap(
        score_matrix, ax=ax_score, cmap=score_cmap, cbar=False,
        yticklabels=False, xticklabels=[score_col], linewidths=0.3, linecolor="white",
    )
    for i, val in enumerate(score_matrix[score_col]):
        ax_score.text(
            0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="white" if val > score_matrix[score_col].median() else "black",
        )

    ax_score.set_ylim(ax_heat.get_ylim())
    ax_score.set_xlabel("")
    ax_score.set_ylabel("")
    ax_score.tick_params(axis="x", rotation=90, labelsize=9)

    ax_heat.xaxis.tick_top()
    ax_heat.xaxis.set_label_position("top")
    ax_score.xaxis.tick_top()
    ax_score.xaxis.set_label_position("top")

    if highlight_label in row_labels:
        idx = row_labels.index(highlight_label)
        for ax, width in ((ax_heat, n_cols), (ax_score, 1)):
            ax.add_patch(
                Rectangle(
                    (0, idx), width, 1,
                    fill=False, edgecolor="crimson", linewidth=3, zorder=5, clip_on=False,
                )
            )
        ax_heat.get_yticklabels()[idx].set_color("crimson")
        ax_heat.get_yticklabels()[idx].set_fontweight("bold")

    fig.subplots_adjust(left=left_margin, right=0.95, top=0.94, bottom=0.12)
    return fig


# ---------------------------------------------------------------------------
# Parameter boxplots -- descriptive only (raw fitted values per mab_virus),
# no model-equation assumptions.
# ---------------------------------------------------------------------------

def summarize_raw_params(params: pd.DataFrame, has_l_gap: bool) -> pd.DataFrame:
    p = params.copy()
    p["f_mode_geommean"] = p.groupby("mab_virus").f_mode.transform(lambda x: np.exp(np.mean(np.log(x))))
    cols = ["mab_virus", "L_mode", "U_mode", "h_mode", "f_mode_geommean"]
    if has_l_gap:
        cols.append("L_gap_mode")
    return p[cols].drop_duplicates()


def plot_param_boxplots(raw_params: pd.DataFrame, has_l_gap: bool) -> plt.Figure:
    cols = ["L_mode", "U_mode", "h_mode", "f_mode_geommean"] + (["L_gap_mode"] if has_l_gap else [])
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(2.2 * len(cols), 3.5))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        y = raw_params[col].dropna()
        ax.violinplot(y)
        x = 1 + np.random.uniform(-0.08, 0.08, size=len(y))
        ax.scatter(x, y, s=10, alpha=0.5, zorder=3, color="orange")
        ax.boxplot(y)
        ax.set_xlabel(col)
        if col == "f_mode_geommean":
            ax.set_yscale("log")

    fig.suptitle("Parameter distributions across mab_virus fits")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
# ============================== FILL ME IN ================================
# Model-fit-curve page: overlays the fitted 4PL curve + IC50/IC80 markers on
# top of the (edge-corrected, normalized) data, per mab_virus.
#
# This is deliberately left unfinished. From 4pl_plate_fit_v{2,3,4,5}_model.txt:
#
#   v2:          RLU_sample = (L + (U-L)/(1+exp(h*(log(t)-log(f)))))*E_ij
#   v3/v4/v5:    RLU_sample = (L + (U-(L+exp(L_gap)))/(1+exp(h*(log(t)-log(f)))))*E_ij
#
#   v2/v3 edge:  E_ij = 1 - alpha*exp(-k*min(min(plate_row,9-plate_row), min(plate_col,13-plate_col)))
#   v4/v5 edge:  E_ij = 1 - alpha*(exp(-k*plate_row)+exp(-k*(9-plate_row))+exp(-k*plate_col)+exp(-k*(13-plate_col)))
#
# logistic_4pl_plate/logistic_4pl_plate_gap and icp_4pl_plate/icp_4pl_plate_gap
# below are algebraically inverted from those two RLU_sample forms (solved
# for t given a target fraction p), but are UNVERIFIED against how this
# project's U_pop=1-fixed convention and its data's rlu/rlu_norm columns are
# meant to be normalized for plotting -- confirm that before trusting the
# curve this produces. correct_rlus_plate below is a stub: it needs the
# actual per-row edge-correction + renormalization logic (see
# generate_pdf_model_report.correct_rlus for the edge_effects-family version
# and generate_pdf_report_pure_rlu.build_corrected_data for the pure_rlu-
# family version -- neither applies here unmodified).
# ---------------------------------------------------------------------------

def logistic_4pl_plate(t, U, L, h, f):
    return L + (U - L) / (1 + np.exp(h * (np.log(t) - np.log(f))))


def logistic_4pl_plate_gap(t, U, L, L_gap, h, f):
    return L + (U - (L + np.exp(L_gap))) / (1 + np.exp(h * (np.log(t) - np.log(f))))


def icp_4pl_plate(p, U, L, h, f):
    return f * (((U - L) / (1 - p - L) - 1) ** (1 / h))


def icp_4pl_plate_gap(p, U, L, L_gap, h, f):
    return f * ((((U - (L + np.exp(L_gap))) / (1 - p - L)) - 1) ** (1 / h))


def correct_rlus_plate(data: pd.DataFrame, params: pd.DataFrame, has_l_gap: bool, edge_formula: str) -> pd.DataFrame:
    """STUB -- fill in the per-row plate-edge correction + renormalization
    for this project before using plot_fit_grid_plate. edge_formula is
    "min_distance" for v2/v3 or "sum_exp" for v4/v5 (see comment block
    above); has_l_gap selects the amplitude form of RLU_sample.
    """
    raise NotImplementedError(
        "correct_rlus_plate is a stub -- fill in the plate_fit-specific edge "
        "correction/renormalization before generating the fit-curve page."
    )


def calc_params_ics_plate(data: pd.DataFrame, params: pd.DataFrame, has_l_gap: bool) -> pd.DataFrame:
    """STUB -- per-mab_virus canonical curve params + IC50/IC80, using
    icp_4pl_plate/icp_4pl_plate_gap above once correct_rlus_plate is filled in.
    """
    raise NotImplementedError(
        "calc_params_ics_plate is a stub -- depends on correct_rlus_plate above."
    )


def plot_fit_grid_plate(plot_params: pd.DataFrame, corrected_data: pd.DataFrame, has_l_gap: bool) -> plt.Figure:
    """STUB -- once calc_params_ics_plate/correct_rlus_plate are filled in,
    this can mirror generate_pdf_model_report.plot_fit_grid /
    generate_pdf_report_pure_rlu.plot_fit_grid almost exactly, just calling
    logistic_4pl_plate(_gap) instead."""
    raise NotImplementedError(
        "plot_fit_grid_plate is a stub -- depends on calc_params_ics_plate/correct_rlus_plate above."
    )


# ============================================================================


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report for a best-fit 4PL_plate_fit model.")
    parser.add_argument("project", choices=PLATE_FIT_PROJECTS, help="which plate_fit project, e.g. 4PL_plate_fit_v3")
    parser.add_argument("model", help='model name, e.g. "m0"')
    parser.add_argument(
        "--data-csv", type=Path, default=None,
        help="overrides the data file this model's own generated .mlxtran points to",
    )
    parser.add_argument(
        "--report-csv", type=Path, default=None,
        help="likelihood report CSV to use for BICc/AIC and the design-matrix page; "
             "defaults to a fresh report combining every registered project in the same "
             "family (plate_fit_v2..v5, etc.)",
    )
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    parser.add_argument(
        "--skip-fit-grid", action="store_true",
        help="omit the model-fit-curve page (use until the FILL ME IN section above is finished)",
    )
    args = parser.parse_args()

    project = args.project

    if args.report_csv is not None:
        report = pd.read_csv(args.report_csv)
        if "project" not in report.columns:
            report.insert(0, "project", project)
    else:
        family = project_family(project)
        reports = []
        for proj in PLATE_FIT_PROJECTS:
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
    params_toggles = discover_params(report_row)

    model_dir = PROJECTS[project] / "model_files" / args.model
    params = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    has_l_gap = "L_gap_mode" in params.columns

    output = args.output or Path(f"{project}_{args.model}_report.pdf")

    figures = [
        plot_info_page(f"{project} {args.model}", report_row, params_toggles),
        plot_design_matrix(report, f"{project} {args.model}"),
        plot_param_boxplots(summarize_raw_params(params, has_l_gap), has_l_gap),
    ]

    if not args.skip_fit_grid:
        data_csv = args.data_csv or resolve_data_csv(PROJECTS[project] / "model_files" / f"{args.model}.mlxtran")
        data = pd.read_csv(data_csv)
        edge_formula = "sum_exp" if project.endswith(("_v4", "_v5")) else "min_distance"
        corrected_data = correct_rlus_plate(data, params, has_l_gap, edge_formula)
        plot_params = calc_params_ics_plate(data, params, has_l_gap)
        figures.append(plot_fit_grid_plate(plot_params, corrected_data, has_l_gap))

    with PdfPages(output) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
