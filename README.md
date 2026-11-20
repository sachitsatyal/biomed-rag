# biomed-rag

Evidence-grounded research assistant for Alzheimer's disease literature: hybrid retrieval
(BM25 + dense), neural reranking, cited answers with abstention, an agentic layer, and rigorous
evaluation.

**Live demo:** https://huggingface.co/spaces/sawcheet/biomed-rag
(API docs at https://sawcheet-biomed-rag.hf.space/docs)

> **Status:** in active development. Research prototype, not for clinical use.

## Architecture

PubMed (NCBI E-utilities) → parsing → chunking → BM25 + dense (FAISS) retrieval → reranker →
LLM answer with citations → LangGraph agent (corpus + live PubMed tools) → FastAPI service +
web page → Docker image → Hugging Face Spaces

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

## Answer and agent results

30 questions (24 golden + 6 unanswerable traps: fictional drugs and genes, false premises,
future events). Same answer model for both systems (`gemini-3.1-flash-lite`), so the only
difference is the agent. Claims judged by `gemini-3.5-flash-lite`, validated with 70 negative
controls (balanced accuracy 0.993).

| | Plain RAG | Agent |
| --- | --- | --- |
| Gave up on answerable questions | 12.5% | **0%** |
| Declined unanswerable traps | 100% | 100% |
| Claims judged supported | 100% | 99% |
| Cited papers labeled relevant | 0.822 | **0.854** |
| … on contested questions | 0.852 | **0.929** |
| Invalid or uncited citations | 0 | 0 |
| Search rounds / used live PubMed | 1 / 0% | 1.6 / 37% |
| Tokens / seconds per question | 1.8k / 1.8 s | 7.3k / 8.9 s |

- The agent plans 2-3 queries (one deliberately seeking contradicting evidence), grades whether
  the evidence covers both sides, and chooses between the local corpus and live PubMed.
- It answered every answerable question (RAG gave up on 3) while still declining all 6 traps,
  at about 4x the cost and latency.
- Every citation is checked in code: a cited PMID must be in the retrieved evidence.

**Caveats:** small question sets; relevance labels were made by an LLM (human spot-check
pending, see `golden/README.md`); 30 agent citations (mostly live PubMed) are not yet labeled;
judge and answerer come from the same model family; significance testing pending.

## Roadmap

- [x] Project setup, CI, code quality
- [x] PubMed ingestion + chunking
- [x] BM25, dense and hybrid retrieval
- [x] Golden set + retrieval evaluation (Recall@K, MRR)
- [x] Reranking (general vs. biomedical cross-encoder)
- [x] Cited answer generation with abstention
- [x] LangGraph agent with tools + agent evaluation
- [x] FastAPI service + web UI + Docker
- [ ] Conflict-aware answering (research track)
- [x] Cloud deployment (Hugging Face Spaces)
- [ ] Monitoring

## Quick start

```bash
git clone https://github.com/sachitsatyal/biomed-rag.git
cd biomed-rag
uv sync
uv run pytest
```

Data and indexes are rebuilt locally (they are not stored in Git):

```bash
cp .env.example .env   # add your NCBI and Gemini API keys
uv run python scripts/download_pubmed.py
uv run python scripts/build_chunks.py
uv run python scripts/build_dense_index.py
uv run python scripts/ask_agent.py "Do anti-amyloid antibodies slow cognitive decline?"
```

## Run the web app

```bash
uv run uvicorn biomed_rag.serving.app:app --port 8000   # open http://localhost:8000
```

Or with Docker (keys are passed at run time, never baked into the image):

```bash
docker build -t biomed-rag .
docker run --rm -p 8000:8000 --env-file .env biomed-rag
```

Deploy to a Hugging Face Space (needs a write token; set `GEMINI_API_KEY`, `NCBI_API_KEY` and
`NCBI_EMAIL` as Space secrets):

```bash
uv run python scripts/deploy_hf_space.py --space <user>/biomed-rag
```

## License

Apache-2.0
