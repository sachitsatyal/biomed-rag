"""Split articles into retrievable chunks that always keep their source PMID."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from biomed_rag.data.pubmed import Article


@dataclass
class Chunk:
    """One retrievable passage. Every chunk can be traced back to its paper."""

    chunk_id: str
    pmid: str
    text: str
    section: str | None = None
    year: int | None = None
    publication_types: list[str] = field(default_factory=list)


def chunk_by_abstract(article: Article) -> list[Chunk]:
    """Strategy A (baseline): one chunk per paper = title + full abstract."""
    return [
        Chunk(
            chunk_id=f"{article.pmid}-0",
            pmid=article.pmid,
            text=f"{article.title}\n\n{article.abstract}",
            year=article.year,
            publication_types=list(article.publication_types),
        )
    ]


# A section line looks like "RESULTS: text" or "MATERIALS AND METHODS: text"
_SECTION = re.compile(r"^([A-Z][A-Z ,&/-]+): (.+)$")


def chunk_by_section(article: Article) -> list[Chunk]:
    """Strategy B: one chunk per labeled abstract section, title prepended.

    Unstructured abstracts (no labels) fall back to one whole-abstract chunk.
    """
    chunks = []
    for i, line in enumerate(article.abstract.split("\n")):
        match = _SECTION.match(line)
        if match is None:
            return chunk_by_abstract(article)
        section, text = match.groups()
        chunks.append(
            Chunk(
                chunk_id=f"{article.pmid}-{i}",
                pmid=article.pmid,
                text=f"{article.title}\n\n{section}: {text}",
                section=section,
                year=article.year,
                publication_types=list(article.publication_types),
            )
        )
    return chunks
