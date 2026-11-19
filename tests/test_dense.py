"""Tests for the dense retriever, using a tiny fake embedding model.

The fake model means no download and no internet, so these run fast in CI.
"""

import json

import numpy as np
import pytest

from biomed_rag.retrieval import dense
from biomed_rag.retrieval.dense import DenseRetriever

VOCAB = ["trem2", "amyloid", "sleep"]


class FakeEncoder:
    """Stands in for SentenceTransformer: one vector dimension per vocabulary word."""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        vecs = np.array([[float(w in t.lower()) for w in VOCAB] for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1, norms)
        return vecs


@pytest.fixture
def retriever(monkeypatch):
    # Replace the real model loader with the fake one for this test only.
    monkeypatch.setattr(dense, "SentenceTransformer", lambda name: FakeEncoder())
    return DenseRetriever.build(
        chunk_ids=["1-0", "2-0", "3-0"],
        pmids=["1", "2", "3"],
        texts=[
            "TREM2 variants alter microglia.",
            "Lecanemab lowers amyloid.",
            "Sleep and amyloid in older adults.",
        ],
        model_name="fake-model",
    )


def test_top_result_is_closest_in_meaning(retriever):
    assert retriever.search("TREM2 receptor", k=1)[0].pmid == "1"


def test_scores_are_cosine_and_sorted(retriever):
    results = retriever.search("amyloid", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)


def test_k_larger_than_corpus_returns_only_real_chunks(retriever):
    assert len(retriever.search("amyloid", k=10)) == 3


def test_save_and_load_round_trip(retriever, tmp_path):
    retriever.save(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {"model_name": "fake-model", "dimension": 3, "n_chunks": 3}

    loaded = DenseRetriever.load(tmp_path)
    assert loaded.model_name == "fake-model"
    assert loaded.search("sleep", k=1)[0].chunk_id == "3-0"
