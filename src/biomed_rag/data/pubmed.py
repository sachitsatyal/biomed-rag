"""Search and download PubMed abstracts via NCBI E-utilities."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
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
    publication_types: list[str] = field(default_factory=list)


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

    def fetch(self, pmids: list[str], batch_size: int = 200) -> list[Article]:
        """Download full records in batches and parse them into Articles."""
        articles: list[Article] = []
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            articles.extend(parse_pubmed_xml(self._get("efetch.fcgi", params).text))
            print(f"  fetched {min(start + batch_size, len(pmids))}/{len(pmids)}")
        return articles


def _text(node: ET.Element | None) -> str:
    """All text inside an element, including nested tags like <sup> or <i>."""
    return "".join(node.itertext()).strip() if node is not None else ""


def _year(art: ET.Element) -> int | None:
    """Publication year; some records only have a MedlineDate like '2019 Nov-Dec'."""
    pub_date = art.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None
    year = pub_date.findtext("Year") or pub_date.findtext("MedlineDate", "")[:4]
    return int(year) if year.isdigit() else None


def parse_pubmed_xml(xml_text: str) -> list[Article]:
    """Parse efetch XML into Articles. Papers without an abstract are skipped."""
    articles = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        # Abstract: only <AbstractText> pieces, keeping section labels
        parts = []
        for node in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            label, text = node.get("Label"), _text(node)
            if text:
                parts.append(f"{label}: {text}" if label else text)
        if not parts:
            continue

        # IDs: ONLY this paper's own list, never the ReferenceList
        ids = {i.get("IdType"): i.text for i in art.findall("./PubmedData/ArticleIdList/ArticleId")}

        authors = [
            f"{a.findtext('ForeName', '')} {a.findtext('LastName')}".strip()
            for a in art.findall("./MedlineCitation/Article/AuthorList/Author")
            if a.findtext("LastName")
        ]

        articles.append(
            Article(
                pmid=art.findtext("./MedlineCitation/PMID", ""),
                title=_text(art.find("./MedlineCitation/Article/ArticleTitle")),
                abstract="\n".join(parts),
                journal=art.findtext("./MedlineCitation/Article/Journal/Title", ""),
                year=_year(art),
                doi=ids.get("doi"),
                pmcid=ids.get("pmc"),
                authors=authors,
                mesh_terms=[
                    m.text
                    for m in art.findall(
                        "./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"
                    )
                    if m.text
                ],
                publication_types=[
                    p.text
                    for p in art.findall(
                        "./MedlineCitation/Article/PublicationTypeList/PublicationType"
                    )
                    if p.text
                ],
            )
        )
    return articles
