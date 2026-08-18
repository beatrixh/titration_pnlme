from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from report_likelihood import PROJECTS
from generate_predicted_vs_observed import PROJECT_ALIASES, load_merged, _square_limits


def plot_grid(runs: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    """runs: list of (column label, merged_df) -- one column per model run,
    one row per mab_virus (union across all runs). Edge-corrected only,
    normalized to [mean(cc rlus), mean(vc rlus)] per run_id/plate."""
    mvs = sorted(set().union(*(set(df.mab_virus.unique()) for _, df in runs)))
    n_rows, n_cols = len(mvs), len(runs)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for j, (label, df) in enumerate(runs):
        for i, mv in enumerate(mvs):
            ax = axes[i, j]
            dat = df.loc[df.mab_virus == mv]
            if dat.empty:
                ax.axis("off")
                continue
            x, y = dat.corrected_obs_norm, dat.center_pred_norm
            lo, hi = _square_limits(x, y)
            ax.plot([lo, hi], [lo, hi], color="gray", ls="--", linewidth=1)
            ax.scatter(x, y, s=20, alpha=0.4, color="cornflowerblue")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

        axes[0, j].set_title(label, fontsize=10)

    for i, mv in enumerate(mvs):
        axes[i, 0].set_ylabel(mv, fontsize=8, rotation=0, ha="right", va="center")
    for ax in axes[-1, :]:
        ax.set_xlabel("observed (edge-corrected, normalized)")

    fig.suptitle("Observed vs. predicted, edge effect corrected, normalized to [mean(cc), mean(vc)]")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare edge-corrected, normalized observed-vs-predicted across several models, "
                    "one column per model."
    )
    parser.add_argument(
        "pairs", nargs="+",
        help='one or more "<project> <model>" pairs, e.g. 4PL_pure_rlu m59 4PL_pure_rlu m64 4PL_pure_rlu_v2 m11',
    )
    parser.add_argument("--output", type=Path, default=None, help="output PDF path")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise SystemExit("expected an even number of arguments (alternating project, model)")
    pairs = list(zip(args.pairs[::2], args.pairs[1::2]))

    runs = []
    for raw_project, model in pairs:
        project = PROJECT_ALIASES.get(raw_project, raw_project)
        if project not in PROJECTS:
            raise SystemExit(f"unknown project {raw_project!r}; expected one of {sorted(PROJECTS)}")
        merged, _obs_col, _pl = load_merged(PROJECTS[project], model)
        runs.append((f"{raw_project} {model}", merged))

    output = args.output or Path("predicted_vs_observed_multi_model.pdf")
    fig = plot_grid(runs)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
