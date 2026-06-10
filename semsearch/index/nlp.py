import re
from snowballstemmer import stemmer as _stemmer

_stemmer = _stemmer("porter")

_STOP_WORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "then",
    "just",
    "so",
    "than",
    "such",
    "both",
    "through",
    "about",
    "for",
    "is",
    "of",
    "while",
    "during",
    "to",
    "from",
    "in",
    "on",
    "at",
    "by",
    "with",
    "without",
    "after",
    "before",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "its",
    "it",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "their",
    "our",
    "do",
    "did",
    "does",
    "doing",
    "done",
    "has",
    "have",
    "had",
    "having",
    "be",
    "been",
    "being",
    "was",
    "were",
    "am",
    "are",
    "is",
    "been",
    "being",
    "get",
    "got",
    "getting",
    "gets",
    "would",
    "could",
    "shall",
    "may",
    "might",
    "must",
    "need",
    "dare",
    "used",
    "ought",
    "please",
    "help",
    "every",
    "each",
    "ever",
    "never",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP_WORDS]


def stem(tokens: list[str]) -> list[str]:
    return _stemmer.stemWords(tokens)


def propagate_negations(tokens: list[str]) -> list[str]:
    negated = False
    result: list[str] = []
    for token in tokens:
        if token in {"not", "no", "never", "nor", "neither", "n't", "nt"}:
            negated = True
        elif negated:
            result.append(f"not_{token}")
            negated = False
        else:
            result.append(token)
    return result


def preprocess(text: str) -> list[str]:
    tokens = tokenize(text.lower())
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tokens = propagate_negations(tokens)
    return tokens
