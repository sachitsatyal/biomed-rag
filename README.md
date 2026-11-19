# biomed-rag

Evidence-grounded research assistant for Alzheimer's disease literature: hybrid retrieval
(BM25 + dense), neural reranking, cited answers with abstention, an agentic layer, and rigorous
evaluation.

> **Status:** in active development. Research prototype, not for clinical use.

## Architecture

PubMed (NCBI E-utilities) → parsing → chunking → BM25 + dense (FAISS) retrieval → reranker →
LLM answer with citations → LangGraph agent (planned) → FastAPI service (planned)

## Retrieval results

24 Alzheimer's questions (exact, paraphrase and contested), 1,982 PubMed abstracts, pooled
relevance labels v3 (521 judged papers, every system fully judged at depth 10). Laptop CPU.

| System | Recall@5 | Recall@10 | MRR | ms / query |
| --- | --- | --- | --- | --- |
| BM25 | 0.312 | 0.501 | 0.740 | 4 |
| Dense (bge-small + FAISS) | 0.424 | 0.700 | 0.896 | 24 |
| Hybrid (RRF) | 0.389 | 0.643 | 0.851 | 29 |
| Dense + MS MARCO reranker | 0.388 | 0.651 | 0.869 | 1,775 |
| **Dense + MedCPT reranker** | **0.463** | **0.719** | **0.917** | 10,170 |

- A general-domain reranker (MS MARCO MiniLM) **hurt** every metric; the biomedical MedCPT
  reranker improved all three, most on paraphrased questions (Recall@10 0.589 → 0.680).
- MedCPT costs about 10 s per query on CPU, so the default pipeline uses dense retrieval and
  MedCPT is optional until it runs on a GPU or a smaller candidate pool.
- Caveats: 24 questions is small (significance testing pending); labels were made by an LLM
  from titles and abstracts, with a human spot-check pending. Details in `golden/README.md`.

Reproduce: `uv run python scripts/evaluate_retrieval.py --reranker ncbi/MedCPT-Cross-Encoder`

## Roadmap

- [x] Project setup, CI, code quality
- [x] PubMed ingestion + chunking
- [x] BM25, dense and hybrid retrieval
- [x] Golden set + retrieval evaluation (Recall@K, MRR)
- [x] Reranking (general vs. biomedical cross-encoder)
- [ ] Cited answer generation with abstention
- [ ] LangGraph agent with tools + agent evaluation
- [ ] Conflict-aware answering (research track)
- [ ] FastAPI service + Docker
- [ ] Cloud deployment + monitoring

## Quick start

```bash
git clone https://github.com/sachitsatyal/biomed-rag.git
cd biomed-rag
uv sync
uv run pytest
```

Data and indexes are rebuilt locally (they are not stored in Git):

```bash
cp .env.example .env   # add your NCBI API key and email
uv run python scripts/download_pubmed.py
uv run python scripts/build_chunks.py
uv run python scripts/build_dense_index.py
```

## License

Apache-2.0
