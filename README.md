# biomed-rag

Evidence-grounded biomedical research assistant: hybrid retrieval (BM25 + dense + knowledge graph),
neural reranking, cited answers with uncertainty/abstention, and rigorous RAG evaluation.

>  **Status:** in active development. Research prototype - not for clinical use yet.

## Planned architecture
PubMed/PMC → cleaning → chunking → BM25 + dense + graph retrieval → reranker → LLM → answer + citations

## Roadmap
- [x] Project setup, CI, code quality
- [ ] PubMed ingestion + chunking
- [ ] BM25, dense and hybrid retrieval
- [ ] Reranking + cited answer generation
- [ ] Evaluation suite (Recall@K, MRR, NDCG, groundedness)
- [ ] FastAPI service + Docker
- [ ] Cloud deployment + monitoring

## Quick start
```bash
git clone https://github.com/sachitsatyal/biomed-rag.git
cd biomed-rag
uv sync
uv run pytest
```

## License
Apache-2.0
