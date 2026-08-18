from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd

from retag_repo import (
    BETA_TOKEN_RE,
    CATEGORIES_RE,
    FILE_PATH_RE,
    PARAM_GROUP_VALUE_RE,
    RETAGGED_DATA_DIR,
    TABULAR_ID_COLS,
    build_token_maps,
    classify,
    iter_target_files,
    load_mapping,
)

# ---------------------------------------------------------------------------
# Final gate, run AFTER retag_repo.py has actually rewritten the files.
#
# Two complementary checks, chosen for what's actually feasible/meaningful
# for each column, rather than one blanket "scan every file for every old
# value" pass (with ~50k+ old values across all 10 columns, a naive
# alternation-regex scan of every file would be both slow and mostly
# pointless, since most of those columns provably never appear inside
# .mlxtran/Monolix-output text at all -- see retag_repo.py's module note).
#
#   1. Structural checks (cheap, precise, no false positives): re-parse
#      every retagged .mlxtran (categories={} lists, beta_ token names,
#      file=/header=) and every retagged tabular output (id/mab_virus/
#      run_id columns), and confirm nothing that survives is a member of
#      the OLD value/token sets. This is the primary check.
#   2. A blanket text scan for the two covariate columns only (mab_virus,
#      run_id -- a few thousand old values total, small enough to compile
#      into one regex) across every retagged .mlxtran/tabular file, as a
#      safety net for any leak outside the specific structural slots
#      retag_repo.py knows about.
#   3. For the other 8 columns (run_name, run, virus_name, virus_col, mab,
#      mablot, id_unique -- monolix_id is covered by the 'id' column check
#      in #1), verification is limited to the retagged input CSVs
#      themselves (a courtesy check of your own retagging, since those
#      values never propagate into .mlxtran/Monolix outputs at all).
# ---------------------------------------------------------------------------


def check_mlxtran(path: Path, mapping: dict, token_maps: dict) -> list[str]:
    text = path.read_text(errors="replace")
    problems = []

    for m in BETA_TOKEN_RE.finditer(text):
        cov, old_tok = m.group(2), m.group(3)
        if old_tok in token_maps[cov]:
            problems.append(f"leftover OLD beta token {m.group(0)!r}")

    for m in CATEGORIES_RE.finditer(text):
        cov = m.group("cov")
        for v in re.findall(r"'([^']*)'", m.group("body")):
            if v in mapping[cov]:
                problems.append(f"leftover OLD {cov} value in categories={{}}: {v!r}")

    fm = FILE_PATH_RE.search(text)
    if fm:
        ref = fm.group(1)
        if RETAGGED_DATA_DIR.name not in ref:
            problems.append(f"file={{path=...}} does not point into retagged_data/: {ref!r}")
    else:
        problems.append("no file={path=...} found")

    return problems


def check_tabular(path: Path, mapping: dict) -> list[str]:
    text = path.read_text(errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return []

    problems = []
    for col, cov in TABULAR_ID_COLS.items():
        if col not in df.columns:
            continue
        old_values = set(mapping[cov].keys())
        # A genuinely-missing cell isn't an identifying value -- exclude it
        # before comparing, so it can't false-positive just because
        # str(NaN) == 'nan' happens to also be a literal key in the mapping
        # (from rows where the ORIGINAL value was itself missing).
        present = df[col].notna()
        leftover = present & df[col].astype(str).isin(old_values)
        if leftover.any():
            bad = df.loc[leftover, col].astype(str).unique()[:5]
            problems.append(f"column {col!r} still has {leftover.sum()} OLD {cov} value(s), e.g. {list(bad)}")

    if "parameter" in df.columns:
        for cell in df["parameter"].astype(str).unique():
            m = PARAM_GROUP_VALUE_RE.match(cell)
            if not m:
                continue
            cov, raw_val = m.group(2), m.group(3)
            if raw_val in mapping[cov]:
                problems.append(f"'parameter' column still has OLD {cov} value embedded: {cell!r}")

    return problems


def check_input_csv(path: Path, mapping: dict) -> list[str]:
    """Courtesy check of the user's own retagged input CSVs, covering all
    10 remapped columns (not just the two Monolix covariates)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"could not read as CSV: {e}"]

    problems = []
    for col in mapping:
        if col not in df.columns:
            continue
        old_values = set(mapping[col].keys())
        present = df[col].notna()
        leftover = present & df[col].astype(str).isin(old_values)
        if leftover.any():
            bad = df.loc[leftover, col].astype(str).unique()[:5]
            problems.append(f"column {col!r} still has {leftover.sum()} OLD value(s), e.g. {list(bad)}")
    return problems


def build_covariate_scan_regex(mapping: dict) -> re.Pattern:
    values = set(mapping["mab_virus"].keys()) | set(mapping["run_id"].keys())
    alternation = "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))
    return re.compile(alternation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify no old identifying values survive after retag_repo.py has run."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    mapping = load_mapping()
    token_maps = build_token_maps(mapping)
    covariate_scan_re = build_covariate_scan_regex(mapping)
    print("Mapping loaded, token maps OK. Scanning...")

    targets = list(iter_target_files(args.root))
    if args.limit:
        targets = targets[: args.limit]

    n_problem_files = 0
    for path, kind in targets:
        problems = check_mlxtran(path, mapping, token_maps) if kind == "mlxtran" else check_tabular(path, mapping)

        text = path.read_text(errors="replace")
        blanket_hit = covariate_scan_re.search(text)
        if blanket_hit:
            problems.append(f"blanket covariate-value scan matched: {blanket_hit.group(0)!r}")

        if problems:
            n_problem_files += 1
            print(f"\n[{path.relative_to(args.root)}]")
            for p in problems:
                print(f"    {p}")

    print(f"\nChecked {len(targets)} files under model_files/-style trees; {n_problem_files} have leftover old values.")

    print("\n--- retagged input CSVs (courtesy check of your own retagging) ---")
    csv_dir = args.root / "input_data" / "retagged_data"
    n_csv_problems = 0
    for csv_path in sorted(csv_dir.glob("*_RETAGGED_*.csv")):
        problems = check_input_csv(csv_path, mapping)
        if problems:
            n_csv_problems += 1
            print(f"\n[{csv_path.relative_to(args.root)}]")
            for p in problems:
                print(f"    {p}")
    print(f"\nChecked retagged CSVs; {n_csv_problems} have leftover old values.")


if __name__ == "__main__":
    main()
