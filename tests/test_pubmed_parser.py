from pathlib import Path

import pytest

from biomed_rag.data.pubmed import parse_pubmed_xml

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_sample.xml"


@pytest.fixture
def articles():
    return parse_pubmed_xml(FIXTURE.read_text(encoding="utf-8"))


def test_skips_articles_without_abstract(articles):
    assert [a.pmid for a in articles] == ["111", "333"]


def test_structured_abstract_keeps_labels_and_nested_text(articles):
    abstract = articles[0].abstract
    assert abstract.startswith("BACKGROUND: Tau matters.")
    assert "RESULTS: p < 10-8 was found." in abstract
    assert "Copyright" not in abstract


def test_title_includes_nested_tags(articles):
    assert articles[0].title == "Tau and APOE in AD."


def test_ids_come_from_own_list_not_references(articles):
    assert articles[0].pmcid == "PMC111"
    assert articles[0].doi == "10.1000/test"


def test_authors_skip_collective_names_and_investigators(articles):
    assert articles[0].authors == ["Jane Smith"]


def test_mesh_terms_and_publication_types(articles):
    assert articles[0].mesh_terms == ["Alzheimer Disease", "Mice"]
    assert "Retracted Publication" in articles[0].publication_types


def test_missing_fields_and_medline_date(articles):
    plain = articles[1]
    assert plain.year == 2019
    assert plain.doi is None
    assert plain.pmcid is None
    assert plain.abstract == "Unlabeled text."
