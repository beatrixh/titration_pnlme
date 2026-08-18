from __future__ import annotations

import argparse
import io
import os
import pickle
import re
from pathlib import Path

import pandas as pd

from generate_mlxtran import sanitize

REPO_ROOT = Path(__file__).resolve().parent
PICKLE_PATH = REPO_ROOT / "input_data" / "retagged_data" / "var_map.pkl"
RETAGGED_DATA_DIR = REPO_ROOT / "input_data" / "retagged_data"
RETAG_SUFFIX = "_RETAGGED_2026_08_17"

# Only these two of the 10 remapped columns are ever used as Monolix
# covariates (embedded in categories={} lists and beta_ parameter names).
# The other 8 (run_name, run, virus_name, virus_col, mab, mablot, id_unique)
# never appear inside .mlxtran text at all -- only as raw column values in
# the linked CSV, which is already retagged, and monolix_id is handled
# separately below (it's the 'id' column in every Monolix output file, but
# never embedded in .mlxtran structure itself).
COVARIATE_COLS = ["mab_virus", "run_id"]

# column name in a tabular output file -> which pickle key maps its values
TABULAR_ID_COLS = {"id": "monolix_id", "mab_virus": "mab_virus", "run_id": "run_id"}

BETA_TOKEN_RE = re.compile(r"\bbeta_(\w+?)_(mab_virus|run_id)_([A-Za-z0-9_]+)")
CATEGORIES_RE = re.compile(
    r"(?P<cov>mab_virus|run_id)\s*=\s*\{type=categorical,\s*categories=\{(?P<body>[^}]*)\}\}"
)
FILE_PATH_RE = re.compile(r"file=\{path='([^']*)'\}")
HEADER_RE = re.compile(r"header=\{[^}]*\}")


# ---------------------------------------------------------------------------
# Mapping setup
# ---------------------------------------------------------------------------

def load_mapping(pkl_path: Path = PICKLE_PATH) -> dict[str, dict]:
    with open(pkl_path, "rb") as f:
        mapping = pickle.load(f)
    # run_id keys/values come out of the pickle as numpy.int64/int (from a
    # pandas column), but every place we need to look values up -- quoted
    # strings inside categories={...}, string cells read back out of a CSV,
    # etc. -- deals in plain strings. Normalize everything to str once here
    # so every lookup elsewhere can just be a plain string dict lookup.
    return {col: {str(k): str(v) for k, v in d.items()} for col, d in mapping.items()}


def build_token_maps(mapping: dict[str, dict]) -> dict[str, dict[str, str]]:
    """old_sanitized_token -> new_sanitized_token, per covariate, with
    collision checks in both directions before anything is trusted:
      - two different old values sanitizing to the same old token but
        pointing at different new values (can't safely resolve)
      - two different old tokens collapsing onto the same new token
        (would merge two distinct beta parameters into one name)
    """
    token_maps: dict[str, dict[str, str]] = {}
    for cov in COVARIATE_COLS:
        old_to_new_candidates: dict[str, set[str]] = {}
        for old_val, new_val in mapping[cov].items():
            old_tok = sanitize(str(old_val))
            new_tok = sanitize(str(new_val))
            old_to_new_candidates.setdefault(old_tok, set()).add(new_tok)

        ambiguous = {k: v for k, v in old_to_new_candidates.items() if len(v) > 1}
        if ambiguous:
            raise SystemExit(
                f"[{cov}] old values collide after sanitize() but disagree on the new "
                f"token -- cannot resolve unambiguously: {ambiguous}"
            )
        old_to_new = {k: next(iter(v)) for k, v in old_to_new_candidates.items()}

        new_tok_sources: dict[str, set[str]] = {}
        for old_tok, new_tok in old_to_new.items():
            new_tok_sources.setdefault(new_tok, set()).add(old_tok)
        merged = {k: v for k, v in new_tok_sources.items() if len(v) > 1}
        if merged:
            raise SystemExit(
                f"[{cov}] distinct old tokens collapse onto the same new token after "
                f"retagging -- would merge separate beta parameters: {merged}"
            )

        token_maps[cov] = old_to_new
    return token_maps


def build_new_value_sets(mapping: dict[str, dict]) -> dict[str, set[str]]:
    """Set of NEW (already-retagged) raw values per covariate -- lets every
    retag_* function recognize 'this value has already been retagged' and
    leave it alone silently, instead of treating anything not found in the
    OLD-value map as an error. Without this, re-running the retagger on
    already-retagged output (e.g. after adding new columns to the retagged
    CSV, or on any second pass) floods warnings for every already-correct
    value, since a new value is never a key in the old->new map."""
    return {cov: set(d.values()) for cov, d in mapping.items()}


def build_new_token_sets(token_maps: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    return {cov: set(d.values()) for cov, d in token_maps.items()}


# ---------------------------------------------------------------------------
# .mlxtran structured retagging
# ---------------------------------------------------------------------------

def retag_beta_tokens(
    text: str, token_maps: dict[str, dict[str, str]], new_token_sets: dict[str, set[str]], warnings: list[str]
) -> str:
    def repl(m: re.Match) -> str:
        param, cov, old_tok = m.group(1), m.group(2), m.group(3)
        new_tok = token_maps[cov].get(old_tok)
        if new_tok is not None:
            return f"beta_{param}_{cov}_{new_tok}"
        if old_tok in new_token_sets[cov]:
            return m.group(0)  # already retagged (e.g. re-running on already-fixed output) -- leave as-is
        warnings.append(f"no token mapping for {old_tok!r} (cov={cov}) in {m.group(0)!r}")
        return m.group(0)

    return BETA_TOKEN_RE.sub(repl, text)


def retag_categories(
    text: str, mapping: dict[str, dict], new_value_sets: dict[str, set[str]], warnings: list[str]
) -> str:
    def repl(m: re.Match) -> str:
        cov = m.group("cov")
        values = re.findall(r"'([^']*)'", m.group("body"))
        cov_map = mapping[cov]
        new_values = []
        for v in values:
            if v in cov_map:
                new_values.append(cov_map[v])
            elif v in new_value_sets[cov]:
                new_values.append(v)  # already retagged -- leave as-is
            else:
                warnings.append(f"no value mapping for {v!r} (cov={cov})")
                new_values.append(v)
        new_body = ", ".join(f"'{v}'" for v in new_values)
        return f"{cov} = {{type=categorical, categories={{{new_body}}}}}"

    return CATEGORIES_RE.sub(repl, text)


def _resolve_new_data_csv(old_path_str: str, mlxtran_path: Path) -> Path:
    if re.match(r"^([A-Za-z]:[\\/]|/)", old_path_str):
        old_abs = Path(old_path_str)
    else:
        old_abs = (mlxtran_path.parent / old_path_str).resolve()
    return RETAGGED_DATA_DIR / f"{old_abs.stem}{RETAG_SUFFIX}.csv"


def retag_datafile_block(text: str, mlxtran_path: Path, warnings: list[str]) -> str:
    m = FILE_PATH_RE.search(text)
    if not m:
        warnings.append("no file={path=...} found in <DATAFILE>")
        return text

    current_path_str = m.group(1)
    if re.match(r"^([A-Za-z]:[\\/]|/)", current_path_str):
        current_abs = Path(current_path_str)
    else:
        current_abs = (mlxtran_path.parent / current_path_str).resolve()

    if current_abs.parent == RETAGGED_DATA_DIR and current_abs.exists():
        # Already points at a retagged CSV (re-running on already-retagged
        # output) -- don't re-derive the path (that would double the
        # _RETAGGED suffix), just refresh header= in case columns were
        # added to the CSV since the last run.
        new_csv = current_abs
    else:
        new_csv = _resolve_new_data_csv(current_path_str, mlxtran_path)
        if not new_csv.exists():
            warnings.append(f"expected retagged CSV not found: {new_csv}")
            return text
        rel = Path(os.path.relpath(new_csv, mlxtran_path.parent)).as_posix()
        text = text[: m.start(1)] + rel + text[m.end(1) :]

    with open(new_csv, "r") as f:
        header_line = f.readline().rstrip("\n\r")
    new_header = ", ".join(col.strip() for col in header_line.split(","))
    text = HEADER_RE.sub(f"header={{{new_header}}}", text, count=1)
    return text


def retag_mlxtran_text(
    text: str, mlxtran_path: Path, mapping: dict, token_maps: dict,
    new_value_sets: dict[str, set[str]], new_token_sets: dict[str, set[str]],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    text = retag_beta_tokens(text, token_maps, new_token_sets, warnings)
    text = retag_categories(text, mapping, new_value_sets, warnings)
    text = retag_datafile_block(text, mlxtran_path, warnings)
    return text, warnings


# ---------------------------------------------------------------------------
# Tabular output retagging (predictions.txt, estimatedIndividualParameters.txt,
# pop.csv, ind_*.csv, populationParametersByGroups.txt, etc.)
# ---------------------------------------------------------------------------


# populationParametersByGroups.txt embeds the RAW (unsanitized) mab_virus/
# run_id value directly at the end of the 'parameter' column's string, e.g.
# 'L_goes_down_True_mab_virus_Clinical VRC01|PVO.4' -- a different, looser
# format than the .mlxtran <PARAMETER> block's sanitized beta_ names, so it
# needs its own handler rather than BETA_TOKEN_RE (which only matches
# already-sanitized tokens). Greedy '.*' naturally finds the *last*
# '_mab_virus_'/'_run_id_' occurrence, which is what we want since that's
# unambiguously where the parameter-name prefix ends and the raw covariate
# value begins. The (?!beta_) exclusion is essential: populationParameters.txt
# (no 'ByGroups') ALSO has a 'parameter' column, but lists the .mlxtran
# <PARAMETER> block's already-sanitized 'beta_...' names -- those are already
# handled correctly by retag_beta_tokens above, and matching them here too
# would try to look up a sanitized TOKEN as if it were a raw value and never
# find it, warning on every single row for no reason.
PARAM_GROUP_VALUE_RE = re.compile(r"^(?!beta_)(.*_(mab_virus|run_id)_)(.*)$")


def retag_param_group_values(
    df: pd.DataFrame, mapping: dict[str, dict], new_value_sets: dict[str, set[str]], warnings: list[str]
) -> bool:
    if "parameter" not in df.columns:
        return False

    changed = False

    def repl(cell):
        nonlocal changed
        m = PARAM_GROUP_VALUE_RE.match(str(cell))
        if not m:
            return cell
        prefix, cov, raw_val = m.group(1), m.group(2), m.group(3)
        new_val = mapping[cov].get(raw_val)
        if new_val is not None:
            changed = True
            return f"{prefix}{new_val}"
        if raw_val in new_value_sets[cov]:
            return cell  # already retagged -- leave as-is
        warnings.append(f"no value mapping for {raw_val!r} (cov={cov}) in parameter name {cell!r}")
        return cell

    df["parameter"] = df["parameter"].map(repl)
    return changed


def retag_tabular_text(
    text: str, mapping: dict, token_maps: dict,
    new_value_sets: dict[str, set[str]], new_token_sets: dict[str, set[str]],
) -> tuple[str, list[str], bool]:
    """Returns (new_text, warnings, was_parsed_as_table).

    Beta-parameter-name text substitution is always safe to apply blanket
    (it's anchored to the unambiguous 'beta_..._mab_virus_/run_id_' pattern,
    which can't coincidentally appear elsewhere) -- this alone handles files
    like populationParameters.txt that list beta names as row *values*
    rather than as table headers. If the file also parses cleanly as a CSV
    with any of {id, mab_virus, run_id} as an actual column, those columns
    get retagged too via exact whole-cell dictionary lookup (never a text
    scan) -- this is what handles predictions.txt / estimatedIndividual
    Parameters.txt / our own R-written CSVs. A 'parameter' column (Monolix's
    populationParametersByGroups.txt) gets its embedded raw covariate value
    retagged too, via retag_param_group_values above.
    """
    warnings: list[str] = []
    beta_retagged_text = retag_beta_tokens(text, token_maps, new_token_sets, warnings)
    changed = beta_retagged_text != text
    text = beta_retagged_text

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return text, warnings, changed

    touched_cols = [c for c in TABULAR_ID_COLS if c in df.columns]

    for col in touched_cols:
        cov = TABULAR_ID_COLS[col]
        cov_map = mapping[cov]
        new_values = new_value_sets[cov]

        def lookup(v):
            nonlocal changed
            if v in cov_map:
                changed = True
                return cov_map[v]
            if str(v) in cov_map:
                changed = True
                return cov_map[str(v)]
            if v in new_values or str(v) in new_values:
                return v  # already retagged -- leave as-is
            warnings.append(f"unmapped {col} value: {v!r}")
            return v

        df[col] = df[col].map(lookup)

    changed = retag_param_group_values(df, mapping, new_value_sets, warnings) or changed

    if not changed:
        return text, warnings, False
    return df.to_csv(index=False), warnings, True


# ---------------------------------------------------------------------------
# File classification / walking
# ---------------------------------------------------------------------------

# Directories that hold Monolix's binary project cache -- can't be safely
# text-retagged, and aren't meant for human/public consumption anyway.
BINARY_SKIP_DIR_NAMES = {".Internals"}
BINARY_SKIP_SUFFIXES = {".dat"}

# Directories intentionally out of scope for this pass:
#   - deprecated_model_files/: confirmed-superseded leftovers from the
#     original wrong-input-data-file mixup (predates the current, correct
#     mlxtran templates for these projects).
#   - best_models/: not yet mapped -- skipped for now at the user's request
#     (see best_models/README.md).
OUT_OF_SCOPE_DIR_NAMES = {"deprecated_model_files", "best_models"}

TABULAR_FILENAMES = {
    "predictions.txt", "summary.txt", "populationParameters.txt",
    "populationParametersByGroups.txt", "estimatedIndividualParameters.txt",
    "estimatedRandomEffects.txt", "simulatedIndividualParameters.txt",
    "simulatedRandomEffects.txt", "individualLL.txt", "logLikelihood.txt",
    "normalityIndividualParameters.txt", "shrinkage.txt",
    "pop.csv", "loglik.csv",
}


def classify(path: Path) -> str:
    if any(part in BINARY_SKIP_DIR_NAMES or part in OUT_OF_SCOPE_DIR_NAMES for part in path.parts):
        return "skip"
    if path.suffix in BINARY_SKIP_SUFFIXES:
        return "skip"
    if path.suffix == ".mlxtran":
        return "mlxtran"
    if path.name in TABULAR_FILENAMES or path.name.startswith("ind_"):
        return "tabular"
    return "other"


def iter_target_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            kind = classify(path)
            if kind != "skip" and kind != "other":
                yield path, kind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retag identifying values across the single_mabs repo.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files (for testing)")
    parser.add_argument("--file", type=Path, default=None, help="process a single specific file only")
    parser.add_argument(
        "--kind", choices=["mlxtran", "tabular", "all"], default="all",
        help="restrict to just .mlxtran files or just tabular outputs (default: all)",
    )
    args = parser.parse_args()

    mapping = load_mapping()
    token_maps = build_token_maps(mapping)
    new_value_sets = build_new_value_sets(mapping)
    new_token_sets = build_new_token_sets(token_maps)
    print("Token maps built OK (no sanitize() collisions).")

    if args.file:
        file_path = args.file if args.file.is_absolute() else (args.root / args.file)
        targets = [(file_path, classify(file_path))]
    else:
        targets = list(iter_target_files(args.root))
        if args.kind != "all":
            targets = [(p, k) for p, k in targets if k == args.kind]
        if args.limit:
            targets = targets[: args.limit]

    n_changed = 0
    n_warnings = 0
    for path, kind in targets:
        text = path.read_text(errors="replace")
        if kind == "mlxtran":
            new_text, warnings = retag_mlxtran_text(text, path, mapping, token_maps, new_value_sets, new_token_sets)
        else:
            new_text, warnings, _parsed = retag_tabular_text(
                text, mapping, token_maps, new_value_sets, new_token_sets
            )

        changed = new_text != text
        if changed:
            n_changed += 1
        if warnings:
            n_warnings += len(warnings)
            print(f"[{path.relative_to(args.root)}] {len(warnings)} warning(s):")
            for w in warnings[:5]:
                print(f"    {w}")
            if len(warnings) > 5:
                print(f"    ... and {len(warnings) - 5} more")

        if not args.dry_run and changed:
            path.write_text(new_text)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Processed {len(targets)} files, "
          f"{n_changed} would change, {n_warnings} total warnings.")


if __name__ == "__main__":
    main()
