# Golden set (retrieval evaluation)

`questions_v1.jsonl` holds 24 Alzheimer's questions in three types:

- **exact**: uses the key term (gene, drug, biomarker), where keyword search should do well
- **paraphrase**: describes the idea in plain words, where meaning-based search should help
- **contested**: the literature disagrees, the core of the research track

## How relevance labels are made (pooling)

1. `scripts/pool_for_labeling.py` takes the top 10 papers from BM25, dense and hybrid for each
   question and writes the unique papers to `labels_v1.csv`.
2. A person opens the CSV and fills the `relevant` column:
   - `1` = the paper directly addresses the question (it would be useful evidence in an answer)
   - `0` = off-topic, or only mentions the topic in passing
3. Judge from the title and abstract start; open the PMID on PubMed when unsure.

Caveat: labels only cover papers some system retrieved (standard TREC-style pooling), so recall
is "pooled recall". This is a development set, not the research benchmark.

## Labels v1

`labels_v1.csv`: 414 pooled rows, 190 marked relevant (3 to 13 per question). Labeled by Claude
(the `labeled_by` column) from title and abstract start; a human spot-check is pending. Treat
these as development labels, not expert ground truth.

## Labels v2

`labels_v2.csv` = v1 plus 66 papers that the rerank systems placed in a top 10 but no v1 system had
retrieved (17 relevant). Same rules, labeled by Claude. Use v2 for any comparison that includes
rerankers; `reports/retrieval_eval_v1.json` was computed on v1.

## Labels v3

`labels_v3.csv` = v2 plus 41 papers placed in a top 10 by the MedCPT reranker (15 relevant).
Results in `reports/retrieval_eval_v3*.json` use v3.
