"""
Create a small stratified subset of the Fakeddit TSV files for local development.

Usage:
    # Create a 5,000-sample subset (stratified on 2_way_label)
    python scripts/make_subset.py --n 5000

    # Use a fraction instead
    python scripts/make_subset.py --frac 0.05

Output is written to data/subset_train.tsv, data/subset_validate.tsv, data/subset_test.tsv.
These are tiny enough to commit or share via chat if needed.
"""

import argparse
from pathlib import Path

import pandas as pd


SPLITS = {
    "train": "data/multimodal_train.tsv",
    "validate": "data/multimodal_validate.tsv",
    "test": "data/multimodal_test_public.tsv",
}

LABEL_COL = "2_way_label"


def make_subset(src: Path, n: int | None, frac: float | None) -> pd.DataFrame:
    df = pd.read_csv(src, sep="\t")
    df = df.fillna("")

    if n is not None:
        # Stratified sample: equal proportion per class up to n total
        n_per_class = n // df[LABEL_COL].nunique()
        df = (
            df.groupby(LABEL_COL, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=42))
            .reset_index(drop=True)
        )
    else:
        df = df.groupby(LABEL_COL, group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42)
        ).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Make a stratified Fakeddit subset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n", type=int, help="Total number of samples (split evenly across labels)")
    group.add_argument("--frac", type=float, help="Fraction of each split to keep (e.g. 0.05)")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing TSV files"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for split, rel_path in SPLITS.items():
        src = Path(rel_path)
        if not src.exists():
            print(f"  Skipping {split} — {src} not found")
            continue

        subset = make_subset(src, args.n, args.frac)
        out_path = data_dir / f"subset_{split}.tsv"
        subset.to_csv(out_path, sep="\t", index=False)

        label_counts = subset[LABEL_COL].value_counts().to_dict()
        print(f"  {split}: {len(subset):,} rows -> {out_path}  (labels: {label_counts})")


if __name__ == "__main__":
    main()
