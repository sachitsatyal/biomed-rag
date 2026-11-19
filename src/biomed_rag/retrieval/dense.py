"""Dense (embedding) retrieval with FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from biomed_rag.retrieval.base import SearchResult

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseRetriever:
    """Semantic search over a FAISS index of chunk embeddings."""

    def __init__(
        self,
        chunk_ids: list[str],
        pmids: list[str],
        texts: list[str],
        model: SentenceTransformer,
        index: faiss.Index,
        model_name: str,
    ) -> None:
        self.chunk_ids = chunk_ids
        self.pmids = pmids
        self.texts = texts
        self.model = model
        self.index = index
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        chunk_ids: list[str],
        pmids: list[str],
        texts: list[str],
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
    ) -> DenseRetriever:
        """Embed all texts (slow, done once) and build an exact FAISS index."""
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(chunk_ids, pmids, texts, model, index, model_name)

    @classmethod
    def from_parquet(cls, path: str | Path, model_name: str = DEFAULT_MODEL) -> DenseRetriever:
        """Build from a chunks Parquet file."""
        df = pd.read_parquet(path)
        ids, pmids, texts = df["chunk_id"].tolist(), df["pmid"].tolist(), df["text"].tolist()
        return cls.build(ids, pmids, texts, model_name)

    def save(self, directory: str | Path) -> None:
        """Write the index, the chunk metadata and the config to a folder."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        pd.DataFrame(
            {"chunk_id": self.chunk_ids, "pmid": self.pmids, "text": self.texts}
        ).to_parquet(directory / "chunks.parquet", index=False)
        config = {
            "model_name": self.model_name,
            "dimension": self.index.d,
            "n_chunks": self.index.ntotal,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> DenseRetriever:
        """Load a saved index (fast). Uses the SAME model it was built with."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        index = faiss.read_index(str(directory / "index.faiss"))
        df = pd.read_parquet(directory / "chunks.parquet")
        model = SentenceTransformer(config["model_name"])
        ids, pmids, texts = df["chunk_id"].tolist(), df["pmid"].tolist(), df["text"].tolist()
        return cls(ids, pmids, texts, model, index, config["model_name"])

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return the k chunks whose meaning is closest to the query."""
        query_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, positions = self.index.search(query_vec, k)
        return [
            SearchResult(self.chunk_ids[i], self.pmids[i], float(score), self.texts[i])
            for score, i in zip(scores[0], positions[0], strict=True)
            if i != -1
        ]
