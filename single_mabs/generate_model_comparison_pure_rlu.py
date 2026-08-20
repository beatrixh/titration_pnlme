from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_pdf_model_report import logistic_4pl, logistic_5pl
from generate_pdf_report_pure_rlu import (
    logistic_4pl_v3, resolve_data_csv, build_corrected_data, calc_params_ics,
)

COLOR_A = "crimson"
COLOR_B = "steelblue"


# ---------------------------------------------------------------------------
# Data prep -- reuses generate_pdf_report_pure_rlu's own build_corrected_data/
# calc_params_ics (has_l_gap-aware) rather than re-deriving the per-project
# formulas here, so a v1/v2/v3+ model is loaded exactly the same way this
# script's single-model report would load it.
# ---------------------------------------------------------------------------

def load_model(project: str, model: str, data_csv: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    model_dir = PROJECTS[project] / "model_files" / model
    params = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    has_u_offset = "U_offset_mode" in params.columns
    has_l_gap = "L_gap_mode" in params.columns
    pl = "5PL" if project.startswith("5PL") else "4PL"

    csv_path = data_csv or resolve_data_csv(model_dir / f"{model}_fitted.mlxtran")
    data = pd.read_csv(csv_path)

    corrected = build_corrected_data(data, params, pl, has_u_offset, has_l_gap)
    plot_params = calc_params_ics(data, params, pl, has_u_offset, has_l_gap)
    return corrected, plot_params, has_l_gap, pl


# ---------------------------------------------------------------------------
# Plotting -- one panel per mab_virus common to both models, each project's
# edge-corrected curve+data overlaid in its own color. Both curves are
# already normalized to their own U=1/L=0 (own-curve neutralization scale)
# by calc_params_ics, which is what makes them directly comparable even
# though the two projects' raw-RLU/edge conventions differ.
# ---------------------------------------------------------------------------

def plot_comparison_grid(
    label_a: str, corrected_a: pd.DataFrame, params_a: pd.DataFrame, has_l_gap_a: bool, pl_a: str,
    label_b: str, corrected_b: pd.DataFrame, params_b: pd.DataFrame, has_l_gap_b: bool, pl_b: str,
) -> plt.Figure:
    mvs_a = set(params_a.mab_virus)
    mvs_b = set(params_b.mab_virus)
    common = sorted(mvs_a & mvs_b)
    only_a = sorted(mvs_a - mvs_b)
    only_b = sorted(mvs_b - mvs_a)
    if only_a:
        print(f"note: mab_virus only in {label_a}, skipped: {', '.join(only_a)}")
    if only_b:
        print(f"note: mab_virus only in {label_b}, skipped: {', '.join(only_b)}")
    if not common:
        raise SystemExit("no mab_virus values in common between the two models")

    n = len(common)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).flatten()
    t = np.exp(np.linspace(np.log(1e-4), np.log(5e1), 10_000))

    params_a_by_mv = params_a.set_index("mab_virus")
    params_b_by_mv = params_b.set_index("mab_virus")
    role_mask_a = (corrected_a.specrole == "sample") if has_l_gap_a else (corrected_a.specrole != "cc")
    role_mask_b = (corrected_b.specrole == "sample") if has_l_gap_b else (corrected_b.specrole != "cc")

    series = (
        (label_a, COLOR_A, corrected_a, params_a_by_mv, role_mask_a, has_l_gap_a, pl_a),
        (label_b, COLOR_B, corrected_b, params_b_by_mv, role_mask_b, has_l_gap_b, pl_b),
    )

    for ax, mv in zip(axes, common):
        for label, color, corrected, params_by_mv, role_mask, has_l_gap, pl in series:
            row = params_by_mv.loc[mv]
            if pl == "5PL":
                y = logistic_5pl(t, row.U, row.L, row.m, row.e, row.s)
            elif has_l_gap:
                y = logistic_4pl_v3(t, row.U, row.L, row.m, row.e)
            else:
                y = logistic_4pl(t, row.U, row.L, row.m, row.e)
            ax.plot(t, 1 - y, linewidth=2, color=color, label=label)

            dat = corrected.loc[(corrected.mab_virus == mv) & role_mask]
            normalized = (dat.rlu_edge_corrected - row.L_raw) / (row.U_raw - row.L_raw)
            ax.scatter(dat.concentration, 1 - normalized, s=25, alpha=0.3, color=color)

        ax.set_xscale("log")
        ax.set_xlim(t.min(), t.max())
        ax.set_title(mv, fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{label_a} (red) vs {label_b} (blue) -- edge-corrected, own-curve-normalized")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay edge-corrected, own-curve-normalized fits for the mab_virus curves "
                     "common to two pure_rlu models -- e.g. compare a 4PL_pure_rlu model to a "
                     "4PL_pure_rlu_v3 model to see how the L_gap reparametrization moved the fits."
    )
    parser.add_argument("model_a", help='model name in --project-a, e.g. "m59"')
    parser.add_argument("model_b", help='model name in --project-b, e.g. "m63"')
    parser.add_argument("--project-a", default="4PL_pure_rlu", choices=sorted(PROJECTS))
    parser.add_argument("--project-b", default="4PL_pure_rlu_v3", choices=sorted(PROJECTS))
    parser.add_argument("--data-csv-a", type=Path, default=None, help="overrides project-a's data file")
    parser.add_argument("--data-csv-b", type=Path, default=None, help="overrides project-b's data file")
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    corrected_a, params_a, has_l_gap_a, pl_a = load_model(args.project_a, args.model_a, args.data_csv_a)
    corrected_b, params_b, has_l_gap_b, pl_b = load_model(args.project_b, args.model_b, args.data_csv_b)

    label_a = f"{args.project_a} {args.model_a}"
    label_b = f"{args.project_b} {args.model_b}"

    fig = plot_comparison_grid(
        label_a, corrected_a, params_a, has_l_gap_a, pl_a,
        label_b, corrected_b, params_b, has_l_gap_b, pl_b,
    )

    output = args.output or Path(
        f"{args.project_a}_{args.model_a}_vs_{args.project_b}_{args.model_b}_comparison.pdf"
    )
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
