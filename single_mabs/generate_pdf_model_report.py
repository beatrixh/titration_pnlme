from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from report_likelihood import PROJECTS

DATA_CSV = Path(
    "/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/input_data/"
    "atlas_data_single_mabs_with_plate_locs_2026-07-30.csv"
)
REPORT_CSV = Path("/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/combined_likelihood_report.csv")


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def logistic_4pl(t, U, L, m, e):
    return L + (U - L) / (1 + np.exp(m * (np.log(t) + np.log(e))))


def logistic_5pl(t, U, L, m, e, s):
    return L + (U - L) / (1 + np.exp(m * (np.log(t) + np.log(e)))) ** s


def icp_4pl(p, U, L, m, e):
    return (((U - L) / (1 - p - L) - 1) ** (1 / m)) / e


def icp_5pl(p, U, L, m, e, s):
    return ((((U - L) / (1 - p - L)) ** (1 / s) - 1) ** (1 / m)) / e


# ---------------------------------------------------------------------------
# Data prep (identical for 4PL/5PL; the PL-specific bits live in the model
# functions above and in calc_params_ics below)
# ---------------------------------------------------------------------------

def correct_rlus(data: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    corrected = data.copy()
    corrected = corrected.merge(
        params[["run_id", "alpha_mode", "k_mode"]].drop_duplicates(), on="run_id", how="left"
    )
    corrected["dist_edge"] = np.minimum(
        np.minimum(corrected.plate_row, 9 - corrected.plate_row),
        np.minimum(corrected.plate_col, 13 - corrected.plate_col),
    )
    corrected["edge_adjustment"] = corrected.alpha_mode * np.exp(-corrected.k_mode * corrected.dist_edge)
    corrected["rlu_norm_edge_adjusted"] = corrected.rlu_norm / (1 - corrected.edge_adjustment)

    correction_factors = (
        corrected.groupby(["run_id", "specrole"]).rlu_norm_edge_adjusted.mean()
        .reset_index()
        .pivot(index="run_id", columns="specrole", values="rlu_norm_edge_adjusted")[["cc", "vc"]]
        .reset_index()
    )
    correction_factors.columns = ["run_id", "cc_min", "vc_max"]

    corrected = corrected.merge(correction_factors, on="run_id", how="left")
    corrected["rlu_renormed"] = (
        (corrected.rlu_norm_edge_adjusted - corrected.cc_min) / (corrected.vc_max - corrected.cc_min)
    )
    return_cols = [
        "run_id", "run_name", "run", "monolix_id", "mab_virus", "mab", "virus_col",
        "plate_row", "plate_col", "specrole", "cc_min", "vc_max", "concentration",
        "rlu_norm", "rlu_norm_edge_adjusted", "rlu_renormed",
    ]
    return corrected[return_cols]


def summarize_raw_params(params: pd.DataFrame, pl: str) -> pd.DataFrame:
    """Per-mab_virus fitted parameter values, uncorrected -- used for the boxplot."""
    p = params.copy()
    p["e_mode_geommean"] = p.groupby("mab_virus").e_mode.transform(lambda x: np.exp(np.mean(np.log(x))))
    cols = ["mab_virus", "U_mode", "L_mode", "m_mode", "e_mode_geommean"]
    if pl == "5PL":
        cols.append("s_mode")
    return p[cols].drop_duplicates()


def calc_params_ics(data: pd.DataFrame, params: pd.DataFrame, pl: str) -> pd.DataFrame:
    corrected = correct_rlus(data, params)
    correction_factors = corrected[["run_id", "cc_min", "vc_max"]].drop_duplicates()
    params = params.merge(correction_factors, on="run_id", how="left")
    params["U_renormed"] = (params.U_mode - params.cc_min) / (params.vc_max - params.cc_min)
    params["L_renormed"] = (params.L_mode - params.cc_min) / (params.vc_max - params.cc_min)
    params["mabvirus_simple"] = (
        params.mab_virus.str.replace(" ", "").str.replace("+", "").str.replace("|", "").str.replace(".", "")
    )

    params_samples = params.loc[
        ~(params.id.str.contains("cc") | params.id.str.contains("vc"))
    ].reset_index(drop=True)
    params_samples["plate_col"] = params_samples.id.str.split("|", expand=True)[3].astype(int)
    params_samples = params_samples.sort_values(by=["run_id", "plate_col"]).reset_index(drop=True)

    grouped = params_samples.groupby("mab_virus")
    values = {
        "mab_virus": params_samples.mab_virus,
        "mab_virus_simple": params_samples.mabvirus_simple,
        "e": grouped.e_mode.transform(lambda x: np.exp(np.mean(np.log(x)))),
        "m": grouped.m_mode.transform("mean"),
        "U": grouped.U_renormed.transform("mean"),
        "L": grouped.L_renormed.transform("mean"),
    }
    if pl == "5PL":
        values["s"] = grouped.s_mode.transform("mean")

    mab_virus_params = pd.DataFrame(values).drop_duplicates().reset_index(drop=True)

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def build_toggle_table(report_row: pd.Series, pl: str) -> pd.DataFrame:
    params = ["U", "L", "m", "e"] + (["s"] if pl == "5PL" else []) + ["alpha", "k"]
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


def plot_info_page(model_label: str, report_row: pd.Series, pl: str) -> plt.Figure:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, model_label, ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.915, f"BICc = {report_row['BICc']:.2f}    AIC = {report_row['AIC']:.2f}", ha="center", fontsize=12)

    table_df = build_toggle_table(report_row, pl)
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

    return fig


DESIGN_MATRIX_COL_RENAME = {
    "U_random": "U (random effect)",
    "U_mab_virus": "U (mab_virus covariate)",
    "U_goes_down": "U (binary neutralization present covariate)",
    "L_random": "L (random effect)",
    "L_mab_virus": "L (mab_virus covariate)",
    "L_goes_down": "L (binary neutralization present covariate)",
    "m_random": "h (random effect)",
    "m_mab_virus": "h (mab_virus covariate)",
    "m_goes_down": "h (binary neutralization present covariate)",
    "e_random": "inflection (random effect)",
    "e_mab_virus": "inflection (mab_virus covariate)",
    "e_goes_down": "inflection (binary neutralization present covariate)",
    "s_random": "s (random effect)",
    "s_mab_virus": "s (mab_virus covariate)",
    "s_goes_down": "s (binary neutralization present covariate)",
}


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

    binary_cols = [c for c in DESIGN_MATRIX_COL_RENAME if c in df.columns]
    df = df[["model"] + binary_cols + ["AIC", score_col]].rename(columns=DESIGN_MATRIX_COL_RENAME)
    binary_cols = [DESIGN_MATRIX_COL_RENAME[c] for c in binary_cols]

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

    # Circle the row for the model this report is about.
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


def plot_param_boxplots(raw_params: pd.DataFrame, pl: str) -> plt.Figure:
    cols = ["L_mode", "m_mode", "e_mode_geommean"] + (["s_mode"] if pl == "5PL" else [])
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(2.2 * len(cols), 3.5))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        y = raw_params[col].dropna()
        ax.violinplot(y)
        x = 1 + np.random.uniform(-0.08, 0.08, size=len(y))
        ax.scatter(x, y, s=10, alpha=0.5, zorder=3, color="orange")
        ax.boxplot(y)
        ax.set_xlabel(col)
        if col == "e_mode_geommean":
            ax.set_yscale("log")

    fig.suptitle("Parameter distributions across mab_virus fits")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
        dat = corrected_data.loc[corrected_data.mab_virus == mv]
        if pl == "5PL":
            y = logistic_5pl(t, row.U, row.L, row.m, row.e, row.s)
        else:
            y = logistic_4pl(t, row.U, row.L, row.m, row.e)
        ax.plot(t, 1 - y, linewidth=2, color="red")

        for val, level in ((row.ic50, 0.5), (row.ic80, 0.8)):
            ax.plot([0, val], [level, level], color="gray", ls="--", linewidth=1)
            ax.plot([val, val], [0, level], color="gray", ls="--", linewidth=1)

        rlus = dat.pivot(index=["run_id", "plate_col"], columns="plate_row", values="rlu_renormed")
        conc = dat.pivot(index=["run_id", "plate_col"], columns="plate_row", values="concentration")
        for idx in rlus.index:
            ax.scatter(
                conc.loc[idx].values, 1 - rlus.loc[idx].values, s=40, alpha=0.3, color="cornflowerblue"
            )

        annotation_lines = [
            f"L = {1 - row.U:.2f}",
            f"U = {1 - row.L:.2f}",
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

    fig.suptitle("Model fits by mab_virus")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report for a best-fit 4PL/5PL model.")
    parser.add_argument("model_name", help='project and model, e.g. "5PL m182"')
    parser.add_argument("--data-csv", type=Path, default=DATA_CSV)
    parser.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    try:
        pl, model = args.model_name.split()
    except ValueError:
        raise SystemExit(f'model_name must be "<project> <model>", e.g. "5PL m182", got {args.model_name!r}')
    if pl not in PROJECTS:
        raise SystemExit(f"unknown project {pl!r}; expected one of {list(PROJECTS)}")

    report = pd.read_csv(args.report_csv)
    row_match = report.loc[(report.project == pl) & (report.model == model)]
    if row_match.empty:
        raise SystemExit(f"no row for project={pl} model={model} in {args.report_csv}")
    report_row = row_match.iloc[0]

    model_dir = PROJECTS[pl] / "model_files" / model
    params = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    data = pd.read_csv(args.data_csv)

    raw_params = summarize_raw_params(params, pl)
    plot_params = calc_params_ics(data, params, pl)
    corrected_data = correct_rlus(data, params)

    output = args.output or Path(f"{pl}_{model}_report.pdf")

    with PdfPages(output) as pdf:
        for fig in (
            plot_info_page(f"{pl} {model}", report_row, pl),
            plot_design_matrix(report, f"{pl} {model}"),
            plot_param_boxplots(raw_params, pl),
            plot_fit_grid(plot_params, corrected_data, pl),
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
