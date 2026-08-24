"""Three diagnostic plot sets for a plate_fit model (4PL_plate_fit_v2/v3), one
subplot per (run_id, mab_virus) pair, x-axis = concentration (log scale):

  1. raw_data_and_fits   -- raw data, and the fit with edge effect applied
                             pointwise (at each observed well's own plate
                             position), connected by straight lines between
                             observed concentrations (no smooth grid, because
                             the edge-corrected fit is not a smooth function
                             of concentration alone -- it also depends on
                             where on the plate each point sits).
  2. normalized           -- same as (1), but rescaled so U -> 1, L -> 0.
  3. edge_removed         -- data with its own pointwise edge effect divided
                             back out (so it estimates what the well would
                             have read with no edge effect), plotted against
                             the *bare* sigmoid curve (no E_ij at all). Once
                             E_ij is removed, the curve is a clean function of
                             concentration alone, so this one *is* drawn on a
                             smooth grid. Normalized the same way as (2).

Run for a single (project, model) pair; swap via --model. See the model.txt
files for the actual fitted equations this script re-implements:

  E_ij = 1 - alpha*exp(-k*d_ij),   d_ij = min(min(plate_row, 9-plate_row),
                                              min(plate_col, 13-plate_col))
  D(t) = 1 / (1 + exp(h*(log(t) - log(f))))
  RLU_sample = (L_s + (U - L_s)*D(t)) * E_ij

  where L_s = L for 4PL_plate_fit_v2 (no L_gap term in that model), and
        L_s = L + exp(L_gap) for 4PL_plate_fit_v3 (has_l_gap below).

Per-panel curve parameters (U, L, h, f, alpha, k, L_gap) are averaged across
every "sample"-role individual in that (run_id, mab_virus) group before
plugging into the equation above -- one fit line per panel, not one per
plate_col replicate. Per the user's explicit rule: parameters Monolix fits
with distribution=normal (U, L, h) are averaged arithmetically; parameters
fit with distribution=logNormal (f, k, alpha, L_gap) are averaged
geometrically (exp(mean(log(x)))) -- confirmed against each project's own
m0.mlxtran DEFINITION block, not assumed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_pdf_model_report_plate_fit import resolve_data_csv
from generate_pdf_report_pure_rlu import dist_edge  # min(row,9-row)/min(col,13-col) helper, identical formula here

# Parameters that Monolix fits with distribution=normal in both v2 and v3's
# m0.mlxtran -- averaged arithmetically across a panel's sample individuals.
NORMAL_PARAMS = ["U_mode", "L_mode", "h_mode"]
# Parameters fit with distribution=logNormal -- averaged geometrically
# (exp(mean(log(x)))), since arithmetic mean of a lognormal is biased high
# by its right skew. L_gap_mode is only present for v3 (has_l_gap).
LOGNORMAL_PARAMS = ["f_mode", "k_mode", "alpha_mode"]


def geomean(x: pd.Series) -> float:
    return float(np.exp(np.mean(np.log(x))))


def build_panel_params(ind: pd.DataFrame, has_l_gap: bool) -> pd.DataFrame:
    """One row per (run_id, mab_virus), with each curve parameter averaged
    across every 'sample'-role individual in that group (see NORMAL_PARAMS/
    LOGNORMAL_PARAMS above for which averaging rule applies to which
    parameter), plus the derived L_s (the curve's own lower asymptote --
    equal to L for v2, or L + exp(geometric-mean(L_gap)) for v3).
    """
    samples = ind.loc[ind.id.str.contains("sample")].copy()

    rows = []
    for (run_id, mab_virus), grp in samples.groupby(["run_id", "mab_virus"]):
        row = {"run_id": run_id, "mab_virus": mab_virus, "n_individuals": len(grp)}
        for col in NORMAL_PARAMS:
            row[col] = grp[col].mean()
        for col in LOGNORMAL_PARAMS:
            row[col] = geomean(grp[col])
        if has_l_gap:
            row["L_gap_mode"] = geomean(grp["L_gap_mode"])
            row["L_s"] = row["L_mode"] + np.exp(row["L_gap_mode"])
        else:
            row["L_s"] = row["L_mode"]
        rows.append(row)

    return pd.DataFrame(rows)


def sigmoid_fraction(t: np.ndarray | float, h: float, f: float) -> np.ndarray | float:
    """D(t) = 1 / (1 + exp(h*(log(t) - log(f)))) -- the part of RLU_sample
    that goes from 1 (t->0) to 0 (t->inf); D=1 means 'fully at U', D=0 means
    'fully at L_s'."""
    return 1.0 / (1.0 + np.exp(h * (np.log(t) - np.log(f))))


def bare_curve(t: np.ndarray | float, U: float, L_s: float, h: float, f: float) -> np.ndarray | float:
    """RLU_sample with E_ij removed: L_s + (U - L_s)*D(t)."""
    return L_s + (U - L_s) * sigmoid_fraction(t, h, f)


def edge_effect(alpha: float, k: float, plate_row, plate_col) -> np.ndarray | float:
    """E_ij = 1 - alpha*exp(-k*d_ij), d_ij = the plate_fit family's
    min-distance-to-nearest-edge formula (identical in v2 and v3's model.txt;
    dist_edge is imported from generate_pdf_report_pure_rlu, which already
    implements exactly this)."""
    return 1.0 - alpha * np.exp(-k * dist_edge(plate_row, plate_col))


def plot_panels(
    data: pd.DataFrame,
    panel_params: pd.DataFrame,
    mode: str,
    output: Path,
) -> None:
    """mode is one of:
      "raw"        -> plot 1: raw data + pointwise edge-applied fit, raw RLU scale
      "normalized" -> plot 2: same as "raw" but rescaled so U->1, L->0
      "edge_removed" -> plot 3: edge-corrected data + bare (no-E_ij) smooth
                        fit curve, on the same U->1/L->0 scale as "normalized"
    Note the normalization anchor is always the *plate's* (U, L) -- not the
    curve's own L_s -- per the user: U/L are "the plate['s] limits", L_s is
    only "the limit for the curve". Using plain L means v3's curve visibly
    does not reach 0 at high concentration; that gap *is* L_gap, and is the
    point of showing it this way.
    """
    n = len(panel_params)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    # Smooth grid only needed for the "edge_removed" curve (mode=="edge_removed"),
    # since that curve has no plate-position dependence left to force pointwise
    # evaluation the way modes "raw"/"normalized" do.
    t_smooth = np.exp(np.linspace(np.log(1e-5), np.log(1e2), 2_000))

    for ax, (_, panel) in zip(axes, panel_params.iterrows()):
        U, L, h, f, alpha, k, L_s = (
            panel.U_mode, panel.L_mode, panel.h_mode, panel.f_mode,
            panel.alpha_mode, panel.k_mode, panel.L_s,
        )

        def norm(y):
            # U -> 1, L -> 0. Applied to both data and fit whenever mode
            # isn't "raw" -- see docstring above for why L (not L_s) is used.
            return (y - L) / (U - L)

        panel_rows = data.loc[
            (data.run_id == panel.run_id)
            & (data.mab_virus == panel.mab_virus)
            & (data.specrole == "sample")
        ]

        # One line per plate_col replicate -- each replicate is a full
        # dilution series down one column, so plate_row (and therefore d_ij,
        # and therefore E_ij) varies point-to-point *within* a single
        # replicate line, which is exactly why the fit needs pointwise
        # correction rather than one edge-effect value per replicate.
        for plate_col, rep in panel_rows.groupby("plate_col"):
            rep = rep.sort_values("concentration")
            d_ij = dist_edge(rep.plate_row, plate_col)
            E_ij_pointwise = 1.0 - alpha * np.exp(-k * d_ij)

            if mode in ("raw", "normalized"):
                # Fit evaluated at each *observed* concentration, edge effect
                # applied using that exact well's own plate position, then
                # connected point-to-point (matplotlib draws straight
                # segments between consecutive points -- this *is* the
                # requested linear interpolation, no separate step needed).
                fit_vals = bare_curve(rep.concentration, U, L_s, h, f) * E_ij_pointwise
                data_vals = rep.rlu
                if mode == "normalized":
                    fit_vals = norm(fit_vals)
                    data_vals = norm(data_vals)
            else:  # "edge_removed"
                # Divide the edge effect back out of the data; the fit curve
                # is bare_curve alone (no E_ij), evaluated on t_smooth below,
                # not per-observation -- computed once per panel, outside
                # this per-replicate loop, since it no longer depends on
                # plate position.
                data_vals = norm(rep.rlu / E_ij_pointwise)

            ax.scatter(rep.concentration, data_vals, s=14, alpha=0.5)
            if mode in ("raw", "normalized"):
                ax.plot(rep.concentration, fit_vals, linewidth=1.5)

        if mode == "edge_removed":
            fit_smooth = bare_curve(t_smooth, U, L_s, h, f)
            ax.plot(t_smooth, norm(fit_smooth), color="black", linewidth=1.5)

        if mode == "raw":
            # Per the user: show mean cc/vc as horizontal reference lines,
            # raw-RLU plot only -- computed directly from this panel's own
            # cc/vc rows rather than trusting a precomputed column, so it's
            # correct regardless of how mean_cc/mean_vc were derived upstream.
            cc_rows = data.loc[
                (data.run_id == panel.run_id) & (data.mab_virus == panel.mab_virus) & (data.specrole == "cc")
            ]
            vc_rows = data.loc[
                (data.run_id == panel.run_id) & (data.mab_virus == panel.mab_virus) & (data.specrole == "vc")
            ]
            if len(cc_rows):
                ax.axhline(cc_rows.rlu.mean(), color="gray", linestyle="--", linewidth=1, label="mean cc")
            if len(vc_rows):
                ax.axhline(vc_rows.rlu.mean(), color="gray", linestyle=":", linewidth=1, label="mean vc")

        ax.set_xscale("log")
        ax.set_title(f"{panel.mab_virus}, run {panel.run_id}", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(output.stem)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-correction diagnostic plots for a plate_fit model.")
    parser.add_argument("project", choices=[p for p in PROJECTS if "plate_fit" in p.lower()])
    parser.add_argument("--model", default="m0", help="model name, swap freely, e.g. m0, m16 (default: m0)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    model_dir = PROJECTS[args.project] / "model_files" / args.model
    ind = pd.read_csv(model_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")
    has_l_gap = "L_gap_mode" in ind.columns

    data_csv = resolve_data_csv(PROJECTS[args.project] / "model_files" / f"{args.model}.mlxtran")
    data = pd.read_csv(data_csv)

    panel_params = build_panel_params(ind, has_l_gap)

    prefix = args.output_dir / f"{args.project}_{args.model}"
    plot_panels(data, panel_params, "raw", Path(f"{prefix}_1_raw_data_and_fits.pdf"))
    plot_panels(data, panel_params, "normalized", Path(f"{prefix}_2_normalized.pdf"))
    plot_panels(data, panel_params, "edge_removed", Path(f"{prefix}_3_edge_removed.pdf"))


if __name__ == "__main__":
    main()
