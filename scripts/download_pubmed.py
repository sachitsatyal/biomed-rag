"""Download an Alzheimer's disease PubMed corpus and save it as Parquet.

Usage:  uv run python scripts/download_pubmed.py --max-results 2000
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from biomed_rag.data.pubmed import PubMedClient

# Research-type papers only: exclude types that don't report findings.
# Retracted papers are deliberately KEPT (needed for the conflict research).
DEFAULT_QUERY = (
    '"alzheimer disease"[MeSH Terms] AND hasabstract AND english[lang] '
    "AND 2015:2026[dp] "
    "NOT (editorial[pt] OR comment[pt] OR letter[pt] OR news[pt] OR published erratum[pt])"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--out", default="data/raw/pubmed_alzheimers.parquet")
    args = parser.parse_args()

    load_dotenv()
    client = PubMedClient(os.environ["NCBI_API_KEY"], os.environ["NCBI_EMAIL"])

    print(f"Searching PubMed: {args.query}")
    pmids = client.search(args.query, args.max_results)
    print(f"Found {len(pmids)} PMIDs. Downloading...")
    articles = client.fetch(pmids)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(a) for a in articles]).to_parquet(out, index=False)

    manifest = {
        "query": args.query,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "n_pmids_found": len(pmids),
        "n_articles_saved": len(articles),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(articles)} articles to {out}")


if __name__ == "__main__":
    main()
