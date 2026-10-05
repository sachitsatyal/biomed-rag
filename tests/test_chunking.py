from biomed_rag.data.chunking import chunk_by_abstract, chunk_by_section
from biomed_rag.data.pubmed import Article


def make_article(abstract: str) -> Article:
    return Article(
        pmid="123",
        title="A title.",
        abstract=abstract,
        journal="J",
        year=2024,
        doi=None,
        pmcid=None,
        publication_types=["Journal Article"],
    )


def test_abstract_strategy_makes_one_chunk_with_title():
    chunks = chunk_by_abstract(make_article("Some text."))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "123-0"
    assert chunks[0].text == "A title.\n\nSome text."
    assert chunks[0].publication_types == ["Journal Article"]


def test_section_strategy_splits_labeled_abstract():
    article = make_article("BACKGROUND: Why.\nMETHODS: How.\nRESULTS: What.")
    chunks = chunk_by_section(article)
    assert [c.section for c in chunks] == ["BACKGROUND", "METHODS", "RESULTS"]
    assert [c.chunk_id for c in chunks] == ["123-0", "123-1", "123-2"]
    assert all(c.text.startswith("A title.") for c in chunks)


def test_section_strategy_handles_multiword_labels():
    chunks = chunk_by_section(make_article("MATERIALS AND METHODS: How."))
    assert chunks[0].section == "MATERIALS AND METHODS"


def test_section_strategy_falls_back_for_unstructured_abstract():
    chunks = chunk_by_section(make_article("Just one plain paragraph."))
    assert len(chunks) == 1
    assert chunks[0].section is None


def test_chunk_copies_publication_types():
    article = make_article("Text.")
    chunk = chunk_by_abstract(article)[0]
    chunk.publication_types.append("Review")
    assert article.publication_types == ["Journal Article"]
