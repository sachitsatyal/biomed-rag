from biomed_rag.retrieval.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("APOE-related Tau.") == ["apoe", "related", "tau"]


def test_greek_letters_become_names():
    assert tokenize("Aβ42 TNF-α") == ["beta", "42", "tnf", "alpha"]


def test_keeps_identifiers_whole():
    assert tokenize("rs9323573 APOE4") == ["rs9323573", "apoe4"]


def test_removes_stopwords_but_keeps_negation():
    assert tokenize("The drug did not slow decline") == ["drug", "did", "not", "slow", "decline"]
