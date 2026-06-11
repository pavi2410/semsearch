from semsearch.index.nlp import preprocess, remove_stopwords, stem, tokenize

# ------------------------------------------------------------------
# tokenize
# ------------------------------------------------------------------


def test_tokenize_basic():
    assert tokenize("hello world") == ["hello", "world"]


def test_tokenize_handles_punctuation():
    assert tokenize("hello, world!") == ["hello", "world"]


def test_tokenize_handles_numbers():
    assert "42" in tokenize("item 42")


def test_tokenize_empty_string():
    assert tokenize("") == []


# ------------------------------------------------------------------
# remove_stopwords
# ------------------------------------------------------------------


def test_removes_common_stopwords():
    tokens = ["the", "quick", "brown", "fox"]
    result = remove_stopwords(tokens)
    assert "the" not in result
    assert "quick" in result
    assert "brown" in result
    assert "fox" in result


def test_remove_stopwords_empty():
    assert remove_stopwords([]) == []


def test_remove_stopwords_all_stops():
    assert remove_stopwords(["the", "a", "is"]) == []


# ------------------------------------------------------------------
# stem
# ------------------------------------------------------------------


def test_stem_reduces_words():
    result = stem(["running", "jumps", "easily"])
    assert result != ["running", "jumps", "easily"]  # something changed
    assert len(result) == 3


def test_stem_empty():
    assert stem([]) == []


# ------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------


def test_preprocess_returns_list():
    result = preprocess("The quick brown fox")
    assert isinstance(result, list)


def test_preprocess_lowercases():
    result = preprocess("HELLO WORLD")
    for token in result:
        assert token == token.lower()


def test_preprocess_removes_stopwords():
    result = preprocess("the cat sat on the mat")
    assert "the" not in result
    assert "on" not in result


def test_preprocess_empty_string():
    assert preprocess("") == []


def test_preprocess_consistent():
    assert preprocess("crawling pages") == preprocess("crawling pages")
