from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_pdf_report_pure_rlu import resolve_data_csv

# Friendly aliases for the edge_effects projects, which are registered under
# short keys ("4PL"/"5PL") elsewhere in this repo's scripts.
PROJECT_ALIASES = {"4PL_edge_effects": "4PL", "5PL_edge_effects": "5PL"}


def _find_results_dir(model_files_dir: Path, model: str) -> Path:
    """Find the directory holding predictions.txt/estimatedIndividualParameters.txt.

    Usually model_files/{model} (e.g. model_files/m59), but at least one
    project's Monolix exportpath was set to the full prefixed filename stem
    instead (model_files/4PL_m59) -- fall back to scanning for a template
    named "*_{model}.mlxtran" and trying its stem as the folder name too.
    """
    candidates = [model_files_dir / model]
    candidates += [model_files_dir / p.stem for p in model_files_dir.glob(f"*_{model}.mlxtran")]
    for c in candidates:
        if (c / "predictions.txt").exists():
            return c
    raise SystemExit(
        f"no predictions.txt found for model {model!r} in {model_files_dir} (tried {candidates})"
    )


def _find_fitted_mlxtran(model_files_dir: Path, model: str) -> Path:
    """Find this model's own saved *_fitted.mlxtran (used only to resolve the
    input data CSV path). Usually model_files/{model}/{model}_fitted.mlxtran
    (written by the R runner using the bare model name, independent of
    whatever exportpath Monolix itself used) -- but check the prefixed
    folder too, for consistency with _find_results_dir above.
    """
    candidates = [model_files_dir / model / f"{model}_fitted.mlxtran"]
    candidates += [
        model_files_dir / p.stem / f"{p.stem}_fitted.mlxtran"
        for p in model_files_dir.glob(f"*_{model}.mlxtran")
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        f"no *_fitted.mlxtran found for model {model!r} in {model_files_dir} (tried {candidates})"
    )


# ---------------------------------------------------------------------------
# This works across every project family (edge_effects and pure_rlu, v1/v2)
# by leaning on Monolix's own predictions.txt for the edge-inclusive
# prediction (indivPred_mode) instead of re-deriving each family's
# structural equation by hand -- it already reflects whatever alpha/k/edge
# math that project's own model file specifies, for that exact observation's
# plate position, correctly regardless of family. The only family-specific
# piece left is the *edge-free* ('canonical', plate-center) counterfactual
# prediction, which Monolix has no built-in notion of -- that requires
# knowing how U behaves as edge->0:
#   - edge_effects projects: U is a directly estimated parameter (U_mode in
#     estimatedIndividualParameters.txt), constant regardless of edge.
#   - pure_rlu (v1): U = mean_vc, derived from the mean_vc regressor.
#   - pure_rlu (v2): U = mean_vc - exp(U_offset), where U_offset is now an
#     estimated individual parameter (U_offset_mode).
# Detected automatically from which columns are present.
# ---------------------------------------------------------------------------

def load_merged(project_dir: Path, model: str) -> tuple[pd.DataFrame, str, str]:
    model_files_dir = project_dir / "model_files"
    results_dir = _find_results_dir(model_files_dir, model)

    predictions = pd.read_csv(results_dir / "predictions.txt")
    params = pd.read_csv(results_dir / "IndividualParameters" / "estimatedIndividualParameters.txt")

    obs_col = predictions.columns[2]  # id, time, <observation>, ... -- name varies (rlu vs rlu_norm)
    pl = "5PL" if "s_mode" in params.columns else "4PL"

    param_cols = ["id", "mab_virus", "run_id", "L_mode", "m_mode", "e_mode"]
    if pl == "5PL":
        param_cols.append("s_mode")
    if "U_mode" in params.columns:
        u_kind = "direct"
        param_cols.append("U_mode")
    elif "U_offset_mode" in params.columns:
        u_kind = "offset"
        param_cols.append("U_offset_mode")
    else:
        u_kind = "derived"

    # ------------------------------------------------------------------
    # Plate-control normalization bounds: upper = mean(vc rlus), lower =
    # mean(cc rlus), computed per run_id (i.e. per plate), since control
    # wells are plate-specific. 'specrole' (which rows are 'vc'/'cc'/
    # 'sample') only lives in the raw input CSV -- it isn't a Monolix
    # covariate, so it never gets echoed into predictions.txt or
    # estimatedIndividualParameters.txt. We read it here just for that
    # label; the actual observed rlu values used for the means come from
    # predictions.txt itself (predictions.txt covers every well, including
    # the cc/vc ones, before we filter those out below for the main frame).
    # ------------------------------------------------------------------
    data_csv = resolve_data_csv(_find_fitted_mlxtran(model_files_dir, model))
    raw_data = pd.read_csv(data_csv)
    specrole_by_id = raw_data[["monolix_id", "specrole"]].drop_duplicates()

    predictions_with_labels = predictions.merge(
        specrole_by_id, left_on="id", right_on="monolix_id", how="left"
    ).merge(params[["id", "run_id"]], on="id", how="left")
    vc_mean_by_run = (
        predictions_with_labels.loc[predictions_with_labels.specrole == "vc"]
        .groupby("run_id")[obs_col].mean()
    )
    cc_mean_by_run = (
        predictions_with_labels.loc[predictions_with_labels.specrole == "cc"]
        .groupby("run_id")[obs_col].mean()
    )

    merged = predictions.merge(params[param_cols], on="id", how="inner")
    # drop 'cc' background wells (virus_cc==0) and zero-concentration control
    # wells (e.g. 'vc'), which would otherwise hit log(0) below.
    merged = merged.loc[(merged.virus_cc != 0) & (merged.time > 0)].copy()

    merged["upper"] = merged.run_id.map(vc_mean_by_run)
    merged["lower"] = merged.run_id.map(cc_mean_by_run)

    # U_center ('canonical', plate-center/edge=0 upper asymptote) is still
    # the model's own *fitted* value -- this is what actually goes into the
    # edge-correction math below (center_pred/corrected_obs), which has
    # nothing to do with the vc/cc normalization bounds computed above.
    if u_kind == "direct":
        U_center = merged.U_mode
    elif u_kind == "offset":
        U_center = merged.mean_vc - np.exp(merged.U_offset_mode)
    else:
        U_center = merged.mean_vc

    D_center = 1 / (1 + np.exp(merged.m_mode * (np.log(merged.time) + np.log(merged.e_mode))))
    if pl == "5PL":
        D_center = D_center ** merged.s_mode

    merged["center_pred"] = merged.L_mode + D_center * (U_center - merged.L_mode)
    merged["corrected_obs"] = merged["center_pred"] + (merged[obs_col] - merged.indivPred_mode)

    # Normalize everything to [lower, upper] = [mean(cc rlus), mean(vc rlus)]
    # for that row's own run_id/plate, so panels for different mab_virus
    # (whose raw magnitudes can differ by orders of magnitude) sit on a
    # common, visually comparable ~[0, 1] scale. Values can still land
    # outside [0, 1] -- e.g. the edge-inclusive ones, or any point beyond
    # the vc/cc control range.
    merged["U_center"] = U_center
    span = merged.upper - merged.lower
    merged["obs_norm"] = (merged[obs_col] - merged.lower) / span
    merged["indivPred_norm"] = (merged.indivPred_mode - merged.lower) / span
    merged["corrected_obs_norm"] = (merged.corrected_obs - merged.lower) / span
    merged["center_pred_norm"] = (merged.center_pred - merged.lower) / span

    return merged, obs_col, pl


# ---------------------------------------------------------------------------
# Plotting -- one row per mab_virus: predicted-vs-observed with the edge
# effect included (left), and the same with the edge effect corrected out
# (right). Each panel gets its own square x=y-referenced axes.
# ---------------------------------------------------------------------------

def _square_limits(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_predicted_vs_observed(merged: pd.DataFrame) -> plt.Figure:
    mvs = sorted(merged.mab_virus.unique())
    n = len(mvs)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 4 * n))
    axes = np.atleast_2d(axes)

    for i, mv in enumerate(mvs):
        ax_raw, ax_corrected = axes[i, 0], axes[i, 1]
        dat = merged.loc[merged.mab_virus == mv]

        for ax, x, y in (
            (ax_raw, dat.obs_norm, dat.indivPred_norm),
            (ax_corrected, dat.corrected_obs_norm, dat.center_pred_norm),
        ):
            lo, hi = _square_limits(x, y)
            ax.plot([lo, hi], [lo, hi], color="gray", ls="--", linewidth=1)
            ax.scatter(x, y, s=20, alpha=0.4, color="cornflowerblue")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

        ax_raw.set_ylabel(mv, fontsize=8, rotation=0, ha="right", va="center")

    axes[0, 0].set_title("Observed vs. predicted, edge effect included", fontsize=10)
    axes[0, 1].set_title("Observed vs. predicted, edge effect corrected", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("observed (normalized to [mean(cc), mean(vc)])")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="For each mab_virus, plot observed-vs-predicted with and without the plate-edge effect."
    )
    parser.add_argument(
        "project", choices=sorted(set(PROJECTS) | set(PROJECT_ALIASES)),
        help='project, e.g. "4PL_pure_rlu_v2" or "5PL_edge_effects" (alias for "5PL")',
    )
    parser.add_argument("model", help='model name, e.g. "m182"')
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    project = PROJECT_ALIASES.get(args.project, args.project)

    merged, obs_col, pl = load_merged(PROJECTS[project], args.model)

    output = args.output or Path(f"{args.project}_{args.model}_predicted_vs_observed.pdf")
    fig = plot_predicted_vs_observed(merged)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
