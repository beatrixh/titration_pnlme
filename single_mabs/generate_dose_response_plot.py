from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_predicted_vs_observed import PROJECT_ALIASES, load_merged


def calc_curve_params(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-mab_virus averaged L/U/m/e/(s), on the model's own raw-RLU
    scale (no normalization), for drawing the canonical (plate-center,
    edge=0) fit curve alongside the raw data."""
    grouped = merged.groupby("mab_virus")
    values = {
        "mab_virus": merged.mab_virus,
        "L": grouped.L_mode.transform("mean"),
        "U": grouped.U_center.transform("mean"),
        "m": grouped.m_mode.transform("mean"),
        "e": grouped.e_mode.transform(lambda x: np.exp(np.mean(np.log(x)))),
    }
    if "s_mode" in merged.columns:
        values["s"] = grouped.s_mode.transform("mean")
    return pd.DataFrame(values).drop_duplicates().reset_index(drop=True)


def plot_dose_response(merged: pd.DataFrame, obs_col: str) -> plt.Figure:
    """One panel per mab_virus: x = concentration, y = raw rlu, with the
    raw observed data scattered against the canonical (edge-free) fit
    curve, both on the model's native, unnormalized scale."""
    curve_params = calc_curve_params(merged).set_index("mab_virus")
    mvs = sorted(merged.mab_virus.unique())
    n = len(mvs)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    t = np.exp(np.linspace(np.log(1e-4), np.log(5e1), 2_000))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, mv in zip(axes, mvs):
        dat = merged.loc[merged.mab_virus == mv]
        row = curve_params.loc[mv]

        D = 1 / (1 + np.exp(row.m * (np.log(t) + np.log(row.e))))
        if "s" in curve_params.columns:
            D = D ** row.s
        curve = row.L + D * (row.U - row.L)
        ax.plot(t, curve, color="red", linewidth=2)

        ax.scatter(dat.time, dat[obs_col], s=20, alpha=0.4, color="cornflowerblue")
        ax.set_xscale("log")
        ax.set_title(mv, fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("concentration")

    fig.suptitle(f"Dose-response: raw {obs_col} vs. canonical (edge-free) fit")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot concentration (x) vs. raw rlu (y), fit and data, per mab_virus."
    )
    parser.add_argument(
        "project", choices=sorted(set(PROJECTS) | set(PROJECT_ALIASES)),
        help='project, e.g. "4PL_pure_rlu_larger_data" or "5PL_edge_effects" (alias for "5PL")',
    )
    parser.add_argument("model", help='model name, e.g. "m59"')
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    project = PROJECT_ALIASES.get(args.project, args.project)
    merged, obs_col, _pl = load_merged(PROJECTS[project], args.model)

    output = args.output or Path(f"{args.project}_{args.model}_dose_response.pdf")
    fig = plot_dose_response(merged, obs_col)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
