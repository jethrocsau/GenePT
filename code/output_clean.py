import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
import argparse

def string_to_numpy(s):
    """Converts string "[1.2, 3.4]" to a real NumPy array."""
    if isinstance(s, str):
        return np.fromstring(s.strip("[]"), sep=",")
    return s

def export_embedding_dicts(
    csv_path: str | Path,
    output_root: str | Path,
    *,
    gene_col: str = "gene_name",
    cols_to_convert: tuple[str, ...] = ("evo2_proj", "genept_proj", "mean_evo2"),
    concat_col: str = "concat_evo2_genept",
    concat_sources: tuple[str, str] = ("evo2_proj", "genept_proj"),
    dict_cols: tuple[str, ...] = ("evo2_proj", "genept_proj", "mean_evo2", "concat_evo2_genept"),
    index_col: int | None = 0,
) -> list[Path]:
    """
    Load the embeddings CSV, convert stringified arrays to numpy arrays, create a concatenated column,
    and export selected columns as pickled dicts {gene_name: np.ndarray}.

    Output folder will be: output_root/<csv_stem>/  (i.e., named after the file).
    Returns a list of written file paths.
    """
    csv_path = Path(csv_path)
    output_root = Path(output_root)

    out_dir = output_root / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    dual_embeddings = pd.read_csv(csv_path, index_col=index_col)

    required = {gene_col, *cols_to_convert}
    missing = required - set(dual_embeddings.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Convert existing columns
    for col in cols_to_convert:
        dual_embeddings[col] = dual_embeddings[col].apply(string_to_numpy)

    # Create concatenated column (if requested and not already present)
    if concat_col in dict_cols or concat_col not in dual_embeddings.columns:
        a, b = concat_sources
        if a not in dual_embeddings.columns or b not in dual_embeddings.columns:
            raise ValueError(f"Cannot build '{concat_col}': missing '{a}' or '{b}'")
        dual_embeddings[concat_col] = dual_embeddings.apply(
            lambda row: np.concatenate((row[a], row[b])),
            axis=1,
        )

    # Export dictionaries
    written: list[Path] = []
    for col in dict_cols:
        if col not in dual_embeddings.columns:
            raise ValueError(f"Requested dict column '{col}' not found in CSV/dataframe")

        temp_dict = dict(zip(dual_embeddings[gene_col], dual_embeddings[col]))
        file_path = out_dir / f"dict_{col}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(temp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        written.append(file_path)

        # Free memory
        del temp_dict
        gc.collect()

    return written

def main():
    p = argparse.ArgumentParser(description="Export embedding columns as {gene: embedding} pickle dicts.")
    p.add_argument(
        "--csv",
        default="../../evo2/data/output/e2e_projected-attn-cls_uniqueTrue_91787_2026-01-18.csv",
        help="Path to input CSV.",
    )
    p.add_argument(
        "--out",
        default="../data/output",
        help="Output root directory. A subfolder named after the CSV file will be created here.",
    )
    args = p.parse_args()

    written = export_embedding_dicts(args.csv, args.out)
    for fp in written:
        print(f"Saved {fp}")

if __name__ == "__main__":
    main()
