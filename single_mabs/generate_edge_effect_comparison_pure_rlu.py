from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_pdf_model_report import logistic_4pl

PROJECT = "4PL_pure_rlu"

DATA_CSV = Path(
    "/mnt/c/Users/bhaddock/repos/titration_pnlme/single_mabs/input_data/"
    "atlas_data_single_mabs_with_plate_locs_2026-07-30_added_vc_col_08082026.csv"
)

PLATE_ROWS = 9
PLATE_COLS = 13


def dist_edge(plate_row: pd.Series, plate_col: pd.Series) -> pd.Series:
    return np.minimum(
        np.minimum(plate_row, PLATE_ROWS - plate_row),
        np.minimum(plate_col, PLATE_COLS - plate_col),
    )


# ---------------------------------------------------------------------------
# Data prep -- both plots stay on raw-RLU units (not normalized/neutralization
# scale) so they're directly visually comparable to each other.
# ---------------------------------------------------------------------------

def build_plot_data(data: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Per-observation edge-inclusive prediction and edge-corrected value.

    f_edge_pred: the model's own prediction at this row's actual plate
    position (edge = alpha*exp(-k*dist_edge) included) -- 'the fit including
    the edge effect'.

    rlu_edge_corrected: the observed rlu with the plate-edge nuisance effect
    detrended out (residual against f_edge_pred added onto the edge=0,
    plate-center prediction) -- 'the corrected data'. 'cc' rows (background,
    virus_cc==0) pass through unmodified since they aren't part of the fitted
    curve.
    """
    merged = data.merge(
        params[["id", "L_mode", "m_mode", "e_mode", "alpha_mode", "k_mode"]],
        left_on="monolix_id", right_on="id", how="inner",
    )
    merged["dist_edge"] = dist_edge(merged.plate_row, merged.plate_col)
    D = 1 / (1 + np.exp(merged.m_mode * (np.log(merged.concentration) + np.log(merged.e_mode))))
    edge = merged.alpha_mode * np.exp(-merged.k_mode * merged.dist_edge)

    merged["f_edge_pred"] = merged.virus_cc * (1 - edge) * (
        merged.L_mode + D * (merged.mean_vc * (1 - edge) - merged.L_mode)
    )
    f_center_pred = merged.L_mode + D * (merged.mean_vc - merged.L_mode)
    residual = merged.rlu - merged.f_edge_pred

    merged["rlu_edge_corrected"] = np.where(
        merged.specrole == "cc", merged.rlu, f_center_pred + residual
    )
    keep = [
        "run_id", "mab_virus", "plate_row", "plate_col", "specrole",
        "concentration", "rlu", "f_edge_pred", "rlu_edge_corrected",
    ]
    return merged[keep]


def calc_canonical_params(data: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Per-mab_virus canonical (plate-center, edge=0) curve, on raw-RLU scale
    (U = mean_vc, L = fitted L -- not normalized to [0,1])."""
    mean_vc_by_run = data[["run_id", "mean_vc"]].drop_duplicates()
    p = params.merge(mean_vc_by_run, on="run_id", how="left")
    grouped = p.groupby("mab_virus")
    return pd.DataFrame({
        "mab_virus": p.mab_virus,
        "e": grouped.e_mode.transform(lambda x: np.exp(np.mean(np.log(x)))),
        "m": grouped.m_mode.transform("mean"),
        "L": grouped.L_mode.transform("mean"),
        "U": grouped.mean_vc.transform("mean"),
    }).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting -- one row per mab_virus, raw|corrected side by side so the two
# views of the same curve are directly comparable at a glance.
# ---------------------------------------------------------------------------

def plot_side_by_side(plot_data: pd.DataFrame, canonical_params: pd.DataFrame) -> plt.Figure:
    mvs = sorted(canonical_params.mab_virus.unique())
    n = len(mvs)
    t = np.exp(np.linspace(np.log(1e-4), np.log(5e1), 10_000))

    # sharex across all rows (concentration grid is common), but NOT sharey --
    # each row gets its own y-axis, shared only between its raw/corrected pair.
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(9, 2.6 * n), sharex=True)
    axes = np.atleast_2d(axes)
    canonical_by_mv = canonical_params.set_index("mab_virus")

    for i, mv in enumerate(mvs):
        ax_raw, ax_corrected = axes[i, 0], axes[i, 1]
        ax_corrected.sharey(ax_raw)
        dat = plot_data.loc[(plot_data.mab_virus == mv) & (plot_data.specrole != "cc")]

        for _, grp in dat.groupby(["run_id", "plate_col"]):
            grp = grp.sort_values("concentration")
            ax_raw.plot(grp.concentration, grp.f_edge_pred, color="red", alpha=0.5, linewidth=1)
            ax_raw.scatter(grp.concentration, grp.rlu, s=25, alpha=0.4, color="cornflowerblue")

        row = canonical_by_mv.loc[mv]
        y = logistic_4pl(t, row.U, row.L, row.m, row.e)
        ax_corrected.plot(t, y, color="red", linewidth=2)
        ax_corrected.scatter(
            dat.concentration, dat.rlu_edge_corrected, s=25, alpha=0.4, color="cornflowerblue"
        )
        annotation_lines = [
            f"L = {row.L:.0f}", f"U = {row.U:.0f}", f"h = {row.m:.2f}", f"f = {row.e:.2g}",
        ]
        ax_corrected.text(
            0.98, 0.02, "\n".join(annotation_lines), transform=ax_corrected.transAxes,
            ha="right", va="bottom", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
        )

        ax_raw.set_xscale("log")
        ax_corrected.set_xscale("log")

        row_ylo = min(dat.rlu.min(), dat.rlu_edge_corrected.min())
        row_yhi = max(dat.rlu.max(), dat.rlu_edge_corrected.max())
        pad = 0.05 * (row_yhi - row_ylo)
        ax_raw.set_ylim(row_ylo - pad, row_yhi + pad)
        ax_raw.set_ylabel(mv, fontsize=8, rotation=0, ha="right", va="center")

    axes[0, 0].set_title("Raw data + fit including plate-edge effect (alpha/k)", fontsize=10)
    axes[0, 1].set_title("Edge-corrected data + fit with no edge effect", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw-vs-edge-inclusive-fit and corrected-vs-edge-free-fit for a 4PL_pure_rlu model."
    )
    parser.add_argument("model", help='model name, e.g. "m59"')
    parser.add_argument("--data-csv", type=Path, default=DATA_CSV)
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    model_dir = PROJECTS[PROJECT] / "model_files" / args.model
    params = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    data = pd.read_csv(args.data_csv)

    plot_data = build_plot_data(data, params)
    canonical_params = calc_canonical_params(data, params)

    output = args.output or Path(f"{PROJECT}_{args.model}_edge_effect_comparison.pdf")

    fig = plot_side_by_side(plot_data, canonical_params)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
