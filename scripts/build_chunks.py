"""Turn the downloaded corpus into chunks (both strategies) and save as Parquet.

Usage:  uv run python scripts/build_chunks.py
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from biomed_rag.data.chunking import chunk_by_abstract, chunk_by_section
from biomed_rag.data.pubmed import Article

STRATEGIES = {"abstract": chunk_by_abstract, "section": chunk_by_section}


def load_articles(path: Path) -> list[Article]:
    """Read the corpus Parquet file back into Article objects."""
    articles = []
    for row in pd.read_parquet(path).to_dict(orient="records"):
        for col in ("authors", "mesh_terms", "publication_types"):
            row[col] = list(row[col])  # Parquet returns NumPy arrays, not lists
        row["year"] = None if pd.isna(row["year"]) else int(row["year"])
        articles.append(Article(**row))
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/pubmed_alzheimers.parquet")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    articles = load_articles(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, chunk_fn in STRATEGIES.items():
        chunks = [chunk for article in articles for chunk in chunk_fn(article)]
        df = pd.DataFrame([asdict(c) for c in chunks])
        df.to_parquet(out_dir / f"chunks_{name}.parquet", index=False)
        mean_len = df["text"].str.len().mean()
        print(f"{name:>8}: {len(chunks)} chunks, mean length {mean_len:.0f} chars")


if __name__ == "__main__":
    main()
