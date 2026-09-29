"""Search and download PubMed abstracts via NCBI E-utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests


@dataclass
class Article:
    """One PubMed paper: the unit our whole RAG pipeline is built on."""

    pmid: str
    title: str
    abstract: str
    journal: str
    year: int | None
    doi: str | None
    pmcid: str | None
    authors: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Small, polite client for NCBI E-utilities (max 10 requests/s with an API key)."""

    def __init__(self, api_key: str, email: str, delay: float = 0.11) -> None:
        self.base_params = {"api_key": api_key, "email": email, "tool": "biomed-rag"}
        self.delay = delay

    def _get(self, endpoint: str, params: dict, retries: int = 3) -> requests.Response:
        for attempt in range(retries):
            resp = requests.get(
                f"{EUTILS}/{endpoint}", params={**self.base_params, **params}, timeout=30
            )
            if resp.ok:
                time.sleep(self.delay)  # stay under NCBI's rate limit
                return resp
            time.sleep(2**attempt)  # wait 1s, 2s, 4s before retrying
        resp.raise_for_status()
        raise RuntimeError("unreachable")

    def search(self, query: str, max_results: int) -> list[str]:
        """Return PMIDs matching the query, most relevant first."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        }
        data = self._get("esearch.fcgi", params).json()
        return data["esearchresult"]["idlist"]
