"""Compare the fitted dose-response curve across every *completed* model in a
4PL_plate_fit_small_data project, one subplot per (run_id, mab_virus) pair,
one overlaid fit line per model.

This is a deliberately simpler cousin of generate_edge_correction_plots.py:
that script does true pointwise (per-observation) edge correction, one fit
line per plate_col replicate. Overlaying that across a dozen-plus models on
the same panel would be unreadable (models x replicates lines per panel).
Instead, each model gets exactly ONE fit line per panel here, edge-corrected
by a single per-panel MEAN E_ij (averaged over that panel's actual observed
well positions) rather than corrected point-by-point. This is an approximation
-- ask for the pointwise version (extend edge-correction-plots style logic
per model instead) if you need per-well precision rather than a quick
cross-model comparison.

IMPORTANT: the edge-effect formula below is transcribed from this project's
own model.txt (4pl_plate_fit_v5_model.txt) and is DIFFERENT from the
min-distance formula used by the original 4PL_plate_fit family:
  - no free decay-rate parameter k -- decay rate is fixed at 1, baked
    directly into the equation (not a Monolix parameter at all; there's no
    k_mode column in this project's IndividualParameters).
  - E_ij is built from all 4 one-sided exponentials (not nearest-edge-only),
    but re-centered by subtracting the value that sum takes at the plate's
    exact center (row=4.5, col=6.5 for an 8x12 plate), so E_ij=1 by
    construction at the center regardless of alpha.
This has only been confirmed against 4pl_plate_fit_v5_model.txt -- other
small_data versions may use a different equation; check their own model.txt
before reusing this script on them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS, discover_models
from generate_pdf_model_report_plate_fit import resolve_data_csv

# Parameters this project's m0.mlxtran declares distribution=normal for --
# averaged arithmetically. (alpha is *normal* here, unlike the original
# plate_fit family where it was logNormal -- confirmed against this
# project's own DEFINITION block, not assumed.)
NORMAL_PARAMS = ["U_mode", "L_mode", "h_mode", "alpha_mode"]
# distribution=logNormal -- averaged geometrically (exp(mean(log(x)))).
LOGNORMAL_PARAMS = ["f_mode"]


def geomean(x: pd.Series) -> float:
    return float(np.exp(np.mean(np.log(x))))


def build_panel_params(ind: pd.DataFrame, has_l_gap: bool) -> pd.DataFrame:
    """One row per (run_id, mab_virus), each curve parameter averaged across
    every 'sample'-role individual in that group -- see NORMAL_PARAMS/
    LOGNORMAL_PARAMS above for which rule applies to which parameter."""
    samples = ind.loc[ind.id.str.contains("sample")].copy()

    rows = []
    for (run_id, mab_virus), grp in samples.groupby(["run_id", "mab_virus"]):
        row = {"run_id": run_id, "mab_virus": mab_virus}
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


def bare_curve(t: np.ndarray, U: float, L_s: float, h: float, f: float) -> np.ndarray:
    """RLU_sample with E_ij removed: L_s + (U - L_s)/(1 + exp(h*(log(t)-log(f))))."""
    return L_s + (U - L_s) / (1.0 + np.exp(h * (np.log(t) - np.log(f))))


# Plate is 8 rows x 12 cols; center is row=4.5, col=6.5. The two "-2*exp(...)"
# terms subtract the value the raw 4-exponential sum takes exactly at that
# center, so E_ij==1 there by construction for any alpha -- transcribed
# directly from 4pl_plate_fit_v5_model.txt, not re-derived.
_CENTER_OFFSET = 2 * np.exp(-4.5) + 2 * np.exp(-6.5)


def edge_effect(alpha: float, plate_row, plate_col) -> np.ndarray:
    raw_sum = (
        np.exp(-plate_row) + np.exp(-(9 - plate_row))
        + np.exp(-plate_col) + np.exp(-(13 - plate_col))
    )
    return 1.0 - alpha * (raw_sum - _CENTER_OFFSET)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay every completed model's fit curve per (run_id, mab_virus) panel."
    )
    parser.add_argument("project", choices=[p for p in PROJECTS if "plate_fit_small_data" in p.lower()])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    models_dir = PROJECTS[args.project] / "model_files"
    model_names = discover_models(models_dir)  # same "has LogLikelihood/logLikelihood.txt" definition as report_likelihood.py
    if not model_names:
        raise SystemExit(f"no completed models found in {models_dir}")
    print(f"completed models: {model_names}")

    # Data is identical across every model in this project (same tracker,
    # same datafile) -- resolve it once from whichever model happens to be
    # first, rather than re-reading it once per model.
    data_csv = resolve_data_csv(models_dir / f"{model_names[0]}.mlxtran")
    data = pd.read_csv(data_csv)
    sample_rows = data.loc[data.specrole == "sample"]

    # Canonical panel list comes from the raw data itself (not from any one
    # model's IndividualParameters) so every panel shows up even if some
    # model's own grouping were to differ.
    panels = sample_rows[["run_id", "mab_virus"]].drop_duplicates().sort_values(["mab_virus", "run_id"])

    panel_params_by_model: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        ind = pd.read_csv(models_dir / model_name / "IndividualParameters" / "estimatedIndividualParameters.txt")
        has_l_gap = "L_gap_mode" in ind.columns
        panel_params_by_model[model_name] = build_panel_params(ind, has_l_gap).set_index(["run_id", "mab_virus"])

    n = len(panels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.8), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    colors = cm.get_cmap("tab20", len(model_names))
    t_smooth = np.exp(np.linspace(np.log(1e-5), np.log(1e2), 2_000))

    for ax, (_, panel) in zip(axes, panels.iterrows()):
        panel_rows = sample_rows.loc[
            (sample_rows.run_id == panel.run_id) & (sample_rows.mab_virus == panel.mab_virus)
        ]
        ax.scatter(panel_rows.concentration, panel_rows.rlu, s=10, alpha=0.35, color="gray", zorder=1)

        cc_rows = data.loc[
            (data.run_id == panel.run_id) & (data.mab_virus == panel.mab_virus) & (data.specrole == "cc")
        ]
        vc_rows = data.loc[
            (data.run_id == panel.run_id) & (data.mab_virus == panel.mab_virus) & (data.specrole == "vc")
        ]
        if len(cc_rows):
            ax.axhline(cc_rows.rlu.mean(), color="crimson", linestyle="--", linewidth=1.2, label="mean cc", zorder=3)
        if len(vc_rows):
            ax.axhline(vc_rows.rlu.mean(), color="forestgreen", linestyle="--", linewidth=1.2, label="mean vc", zorder=3)

        for i, model_name in enumerate(model_names):
            params = panel_params_by_model[model_name]
            key = (panel.run_id, panel.mab_virus)
            if key not in params.index:
                continue  # shouldn't happen (same dataset every model), but don't crash if it does
            row = params.loc[key]

            # Single per-panel mean E_ij (averaged alpha for this model,
            # averaged over this panel's own actual observed well
            # positions) -- see module docstring for why this isn't
            # pointwise like generate_edge_correction_plots.py.
            mean_eij = edge_effect(row.alpha_mode, panel_rows.plate_row, panel_rows.plate_col).mean()

            fit = bare_curve(t_smooth, row.U_mode, row.L_s, row.h_mode, row.f_mode) * mean_eij
            ax.plot(t_smooth, fit, color=colors(i), linewidth=1.2, label=model_name, zorder=2)

        ax.set_xscale("log")
        ax.set_title(f"{panel.mab_virus}, run {panel.run_id}", fontsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    # One shared legend (per-subplot legends with 14+ entries each would be
    # unreadable). Collected across *all* subplots and deduped by label,
    # rather than taken from just one panel -- not every panel necessarily
    # has both cc and vc rows, so a single panel's handles could miss one.
    seen: dict[str, object] = {}
    for ax in axes[:n]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()), list(seen.keys()),
        loc="upper center", ncol=min(len(seen), 8), fontsize=7, bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(f"{args.project}: fit comparison across completed models", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output = args.output or Path(f"{args.project}_all_models_fit_comparison.pdf")
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
