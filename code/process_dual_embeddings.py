"""Process dual-embedding CSVs into reusable embedding dictionaries.

This repository often stores dual-embeddings where vector columns are serialized as strings
like "[0.1, 0.2, ...]". This script:

- Loads a CSV containing at least: gene_name, evo2_proj, genept_proj, mean_evo2
- Converts vector columns into NumPy arrays
- Exports per-embedding dictionaries {gene_name: np.ndarray} as pickle files

Designed to be modular so notebooks can import the functions, and also runnable as a CLI.

Example:
    python code/process_dual_embeddings.py \
      --input ../../evo2/data/output/e2e_projected-attn-cls_uniqueTrue_91787_2026-01-18.csv \
      --out-dir data/output/dual_embeddings/e2e_projected-attn-cls_uniqueTrue_91787_2026-01-18

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_EMBEDDING_COLUMNS: List[str] = [
    "mean_evo2",
    "genept",  # optional if present in the input file
    "evo2_proj",
    "genept_proj",
]


def parse_vector_cell(x) -> np.ndarray:
    """Parse a single CSV cell into a 1D numpy array.

    Accepts:
      - numpy arrays (returned as-is)
      - python lists/tuples
      - strings like "[1, 2, 3]" or "1,2,3"
      - NaN/None -> empty array
    """

    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.array([], dtype=float)

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if s == "":
            return np.array([], dtype=float)
        return np.fromstring(s, sep=",", dtype=float)

    # Fallback: try to coerce scalar
    return np.asarray([x], dtype=float)


def load_dual_embeddings_csv(
    input_csv: str | os.PathLike,
    required_columns: Iterable[str] = ("gene_name",),
) -> pd.DataFrame:
    """Load the dual embedding CSV."""

    df = pd.read_csv(input_csv)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")
    return df


def normalize_and_extract_embeddings(
    df: pd.DataFrame,
    embedding_columns: Iterable[str],
    gene_name_col: str = "gene_name",
    concat_pairs: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return dict-of-dicts: {embedding_name: {gene_name: vector}}.

    Only columns present in df are processed.
    """

    if gene_name_col not in df.columns:
        raise ValueError(f"gene_name column '{gene_name_col}' not in dataframe")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    gene_names = df[gene_name_col].astype(str).tolist()

    for col in embedding_columns:
        if col not in df.columns:
            continue
        vectors = df[col].apply(parse_vector_cell).tolist()
        out[col] = dict(zip(gene_names, vectors))

    # Optional: derived concatenations
    if concat_pairs:
        for spec in concat_pairs:
            # Format: "colA+colB[:new_name]" where new_name default is "concat_colA_colB"
            name_part = spec
            new_name: Optional[str] = None
            if ":" in spec:
                name_part, new_name = spec.split(":", 1)
                new_name = new_name.strip() or None
            if "+" not in name_part:
                raise ValueError(
                    f"Invalid concat spec '{spec}'. Expected 'colA+colB' or 'colA+colB:new_name'."
                )
            col_a, col_b = [x.strip() for x in name_part.split("+", 1)]
            if col_a not in df.columns or col_b not in df.columns:
                continue
            out_name = new_name or f"concat_{col_a}_{col_b}"

            vec_a = df[col_a].apply(parse_vector_cell).tolist()
            vec_b = df[col_b].apply(parse_vector_cell).tolist()
            out[out_name] = {
                g: np.concatenate((a, b)) for g, a, b in zip(gene_names, vec_a, vec_b)
            }

    return out


def save_embedding_dicts(
    embedding_dicts: Mapping[str, Mapping[str, np.ndarray]],
    out_dir: str | os.PathLike,
    prefix: str = "dict_",
) -> Dict[str, str]:
    """Save each embedding dict to out_dir as a pickle file.

    Returns {embedding_name: saved_path}.
    """

    import pickle

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}
    for name, d in embedding_dicts.items():
        file_path = out_path / f"{prefix}{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(dict(d), f, protocol=pickle.HIGHEST_PROTOCOL)
        saved[name] = str(file_path)

    return saved


def save_embedding_metadata(
    embedding_dicts: Mapping[str, Mapping[str, np.ndarray]],
    out_dir: str | os.PathLike,
    input_file: str | os.PathLike,
) -> str:
    """Write a small JSON file describing what was exported."""

    def dim_of_any(v: Mapping[str, np.ndarray]) -> Optional[int]:
        for arr in v.values():
            if isinstance(arr, np.ndarray) and arr.size > 0:
                return int(arr.shape[0])
        return None

    meta = {
        "input_file": str(input_file),
        "exported": {
            name: {
                "n_genes": int(len(d)),
                "dim": dim_of_any(d),
            }
            for name, d in embedding_dicts.items()
        },
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return str(meta_path)


def process_and_save_dual_embeddings(
    input_csv: str | os.PathLike,
    out_dir: str | os.PathLike,
    embedding_columns: Iterable[str] = DEFAULT_EMBEDDING_COLUMNS,
    gene_name_col: str = "gene_name",
    concat_pairs: Optional[List[str]] = None,
) -> Dict[str, str]:
    """High-level convenience wrapper: load -> parse -> save."""

    df = load_dual_embeddings_csv(input_csv)
    embedding_dicts = normalize_and_extract_embeddings(
        df,
        embedding_columns=embedding_columns,
        gene_name_col=gene_name_col,
        concat_pairs=concat_pairs,
    )

    saved = save_embedding_dicts(embedding_dicts, out_dir=out_dir)
    save_embedding_metadata(embedding_dicts, out_dir=out_dir, input_file=input_csv)
    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process a dual-embedding CSV into per-embedding {gene_name: vector} pickles",
    )
    p.add_argument("--input", required=True, help="Path to the dual-embedding CSV")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output folder to write pickles into (created if missing)",
    )
    p.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_EMBEDDING_COLUMNS,
        help="Embedding columns to export (only those present are written)",
    )
    p.add_argument(
        "--concat",
        nargs="*",
        default=[],
        help=(
            "Optional derived concatenations to export, e.g. 'evo2_proj+genept_proj' "
            "or 'evo2_proj+genept_proj:concat_evo2_genept'"
        ),
    )
    p.add_argument(
        "--gene-name-col",
        default="gene_name",
        help="Column containing the gene symbol/name",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    saved = process_and_save_dual_embeddings(
        input_csv=args.input,
        out_dir=args.out_dir,
        embedding_columns=args.columns,
        gene_name_col=args.gene_name_col,
        concat_pairs=args.concat or None,
    )

    print("Saved embedding dictionaries:")
    for k, v in saved.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
