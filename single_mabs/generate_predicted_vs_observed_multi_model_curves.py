from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from report_likelihood import PROJECTS
# load_merged does essentially all the heavy numerical work (per-family
# edge-correction math, plate-control [cc/vc] normalization bounds) and
# hands back one row per observation with all the columns this script reads
# below (see generate_predicted_vs_observed.py for exactly how each column
# is derived -- this file only consumes them and draws pictures).
from generate_predicted_vs_observed import PROJECT_ALIASES, load_merged, _square_limits

# A fixed, repeatable color per model column (cycles if there are more than
# 10 models -- tab10 only has 10 distinct colors).
COLORS = plt.get_cmap("tab10").colors


def calc_curve_params(merged: pd.DataFrame) -> pd.DataFrame:
    """Collapse one model's per-observation rows down to one row per
    mab_virus, holding just the numbers needed to draw that mab_virus's
    smooth reference curve in the rightmost 'concentration-response' column.

    Averaging convention (matches the rest of this project's scripts):
      - m, L, U, lower, upper: plain arithmetic mean across that
        mab_virus's individuals/rows.
      - e: geometric mean (mean of log(e), then exponentiated) -- e is an
        EC50-like concentration parameter, so it's naturally log-scaled;
        averaging on the log scale is the standard way to summarize it.
      - s (5PL shape parameter only): plain arithmetic mean, same as m.

    Why L/U/lower/upper are needed here (unlike a simpler, now-outdated
    version of this function): the curve is normalized to
    [lower, upper] = [mean(cc rlus), mean(vc rlus)] -- the *plate control*
    range -- not to the model's own fitted [L, U] asymptotes anymore. Those
    two ranges are NOT the same in general, so we can no longer take the
    shortcut of "normalizing a logistic curve by its own asymptotes always
    gives the bare sigmoid" -- that shortcut only holds when the
    normalization range IS the curve's own [L, U]. Now we have to actually
    build the raw-scale curve first (L + D(t)*(U-L)) and only then rescale
    it into the [lower, upper] control-well range.
    """
    grouped = merged.groupby("mab_virus")
    values = {
        "mab_virus": merged.mab_virus,
        "m": grouped.m_mode.transform("mean"),
        "e": grouped.e_mode.transform(lambda x: np.exp(np.mean(np.log(x)))),
        "L": grouped.L_mode.transform("mean"),
        "U": grouped.U_center.transform("mean"),
        "lower": grouped.lower.transform("mean"),
        "upper": grouped.upper.transform("mean"),
    }
    if "s_mode" in merged.columns:
        values["s"] = grouped.s_mode.transform("mean")
    # merged has one row per *observation*; every column above was built
    # with .transform(...), which broadcasts the group's aggregate back onto
    # every row in that group -- so at this point every row belonging to the
    # same mab_virus has identical values. drop_duplicates() then collapses
    # that down to exactly one row per mab_virus.
    return pd.DataFrame(values).drop_duplicates().reset_index(drop=True)


def plot_grid(runs: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    """Build the whole figure.

    Layout:
      - Rows: one per mab_virus (the union of mab_virus values seen across
        ALL the requested model runs -- if a mab_virus is missing from one
        model's data, that model's cell in that row is just left blank).
      - Columns 0 .. len(runs)-1: one per requested model. Each panel is an
        "observed vs. predicted" scatter (edge-corrected, normalized,
        flipped to 1-x/1-y so it reads as "fraction neutralized" and
        matches the convention used in the rightmost column), with a
        dashed gray y=x reference line. Every model's dots use ITS OWN
        fixed color from COLORS, so the same color always means the same
        model everywhere in the figure.
      - Column len(runs) (the rightmost, extra column): overlays every
        model's own smooth concentration-response curve plus its own
        edge-corrected data points (both in that model's color), PLUS the
        model's raw, uncorrected data as open black circles (so you can
        see how far the edge correction moved each point). x-axis is real
        concentration (log scale), not a 0/1 model index.

    runs: list of (column label, merged_df) tuples, one entry per model
    the caller asked for, in the order given on the command line. Each
    merged_df is exactly what load_merged() returned for that model.
    """
    # mvs: sorted union of every mab_virus name appearing in ANY of the
    # requested models' data, so every row lines up across all columns even
    # if the underlying datasets differ (e.g. 4PL_pure_rlu and
    # 4PL_pure_rlu_v2 are fit to genuinely different source CSVs).
    mvs = sorted(set().union(*(set(df.mab_virus.unique()) for _, df in runs)))
    n_rows, n_cols = len(mvs), len(runs) + 1  # +1 for the concentration-response column
    colors = [COLORS[j % len(COLORS)] for j in range(len(runs))]
    # Dense log-spaced concentration grid purely for drawing smooth curves
    # (not tied to any model's actual observed concentrations).
    t = np.exp(np.linspace(np.log(1e-4), np.log(5e1), 2_000))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    # plt.subplots collapses to a bare Axes (not an array) whenever
    # nrows==ncols==1; np.atleast_2d + reshape guarantees axes is always
    # indexable as axes[row, col] regardless of how many rows/cols there are.
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    # One curve-parameter row per mab_virus, per model -- computed once up
    # front (outside the row/column loops below) since it doesn't depend on
    # which panel we're currently drawing.
    curve_params_by_run = [calc_curve_params(df).set_index("mab_virus") for _, df in runs]

    # ---- Left-hand columns: one "observed vs. predicted" scatter per model ----
    for j, (label, df) in enumerate(runs):
        color = colors[j]
        for i, mv in enumerate(mvs):
            ax = axes[i, j]
            dat = df.loc[df.mab_virus == mv]
            if dat.empty:
                # This model's data has no rows for this mab_virus (e.g.
                # different source datasets) -- leave a blank panel rather
                # than plotting nothing onto empty/default axes.
                ax.axis("off")
                continue
            # corrected_obs_norm / center_pred_norm are already:
            #   (edge-corrected value - lower) / (upper - lower)
            # i.e. normalized to [mean(cc rlus), mean(vc rlus)] for that
            # row's own run_id/plate (see load_merged). The "1 - " flips it
            # into a "fraction neutralized" convention: 0 at the vc
            # (no-neutralization) control level, 1 at the cc (fully
            # neutralized / background) control level.
            x, y = 1 - dat.corrected_obs_norm, 1 - dat.center_pred_norm
            lo, hi = _square_limits(x, y)  # symmetric range covering both x and y, with padding
            ax.plot([lo, hi], [lo, hi], color="gray", ls="--", linewidth=1)  # y = x reference
            ax.scatter(x, y, s=20, alpha=0.4, color=color)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")  # keeps the y=x line visually at 45 degrees
        axes[0, j].set_title(label, fontsize=10)

    # ---- Rightmost column: every model's curve + data overlaid together ----
    curve_col = len(runs)
    for i, mv in enumerate(mvs):
        ax = axes[i, curve_col]
        for j, (label, df) in enumerate(runs):
            color = colors[j]
            curve_params = curve_params_by_run[j]
            if mv not in curve_params.index:
                continue  # this model has no data at all for this mab_virus
            row = curve_params.loc[mv]

            # D(t): the base logistic/sigmoid term shared by 4PL and 5PL,
            # 0 at t->0 and 1 at t->inf (or the mirror image, depending on
            # sign conventions -- what matters here is it's the same D(t)
            # used inside center_pred back in load_merged). 5PL raises it to
            # the extra shape power s; 4PL leaves it as-is.
            D = 1 / (1 + np.exp(row.m * (np.log(t) + np.log(row.e))))
            if "s" in curve_params.columns:
                D = D ** row.s

            # Reconstruct the model's own raw-scale canonical curve
            # (L + D(t)*(U-L), same form as center_pred in load_merged, but
            # evaluated on a smooth t grid instead of at each observation's
            # actual concentration), then rescale it into the SAME
            # [lower, upper] = [mean(cc), mean(vc)] control range used to
            # normalize the data, so the curve and the scattered points
            # below are on a directly comparable scale.
            raw_curve = row.L + D * (row.U - row.L)
            norm_curve = (raw_curve - row.lower) / (row.upper - row.lower)
            ax.plot(t, 1 - norm_curve, color=color, linewidth=2)

            dat = df.loc[df.mab_virus == mv]
            # Filled dots: this model's own edge-corrected data (same
            # quantity as the left-hand columns' x-axis), plotted here
            # against its real concentration instead of against the
            # predicted value.
            ax.scatter(dat.time, 1 - dat.corrected_obs_norm, s=15, alpha=0.4, color=color)
            # Open black circles: the SAME points but WITHOUT the edge
            # correction (obs_norm = raw observed value, normalized but not
            # detrended for plate position) -- comparing a filled dot to its
            # matching open circle shows exactly how much the edge
            # correction moved that specific observation.
            ax.scatter(
                dat.time, 1 - dat.obs_norm, s=20, alpha=0.6,
                facecolors="none", edgecolors="black", linewidths=0.6,
            )
        ax.set_xscale("log")
    axes[0, curve_col].set_title("Concentration-response, all models overlaid", fontsize=10)

    # ---- Row/column labels ----
    for i, mv in enumerate(mvs):
        # Row label lives on the leftmost column only, as a horizontal
        # (rotation=0) label so mab_virus names stay readable.
        axes[i, 0].set_ylabel(mv, fontsize=8, rotation=0, ha="right", va="center")
    for ax in axes[-1, :len(runs)]:
        ax.set_xlabel("1 - observed (edge-corrected, normalized)")
    axes[-1, curve_col].set_xlabel("concentration")

    # ---- Legend: one colored line+marker per model, plus one entry
    # explaining the open-black-circle raw-data marker used in the
    # rightmost column. ----
    legend_handles = [
        Line2D([0], [0], color=colors[j], marker="o", linestyle="-", label=label)
        for j, (label, _df) in enumerate(runs)
    ]
    legend_handles.append(
        Line2D(
            [0], [0], color="none", marker="o", markeredgecolor="black", markerfacecolor="none",
            linestyle="", label="raw (uncorrected)",
        )
    )
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(runs) + 1, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        "Observed vs. predicted (edge-corrected, normalized) plus overlaid model curves", y=1.02
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare edge-corrected, normalized observed-vs-predicted across several models "
                    "(one column per model, color-coded), plus a shared concentration-response curve column."
    )
    parser.add_argument(
        "pairs", nargs="+",
        help='one or more "<project> <model>" pairs, e.g. 4PL_pure_rlu m59 4PL_pure_rlu m64 4PL_pure_rlu_v2 m11',
    )
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    # args.pairs is a flat list like ["4PL_pure_rlu", "m59", "4PL_pure_rlu_v2", "m11"];
    # zip(pairs[::2], pairs[1::2]) regroups it into [("4PL_pure_rlu", "m59"), ("4PL_pure_rlu_v2", "m11")].
    if len(args.pairs) % 2 != 0:
        raise SystemExit("expected an even number of arguments (alternating project, model)")
    pairs = list(zip(args.pairs[::2], args.pairs[1::2]))

    runs = []
    for raw_project, model in pairs:
        # raw_project is exactly what the user typed (e.g. "5PL_edge_effects");
        # project is the normalized key report_likelihood.PROJECTS actually
        # uses (e.g. "5PL"). The column label below keeps the user's own
        # spelling so the legend/titles read naturally.
        project = PROJECT_ALIASES.get(raw_project, raw_project)
        if project not in PROJECTS:
            raise SystemExit(f"unknown project {raw_project!r}; expected one of {sorted(PROJECTS)}")
        merged, _obs_col, _pl = load_merged(PROJECTS[project], model)
        runs.append((f"{raw_project} {model}", merged))

    output = args.output or Path("predicted_vs_observed_multi_model_curves.pdf")
    fig = plot_grid(runs)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
