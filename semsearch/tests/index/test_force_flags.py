import pytest

from semsearch.index.force_flags import extract_force_flags, resolve_force_targets, split_force_targets


def test_split_force_targets_accepts_commas():
    assert split_force_targets("bm25,embeddings") == ["bm25", "embeddings"]
    assert split_force_targets("bm25, embeddings") == ["bm25", "embeddings"]


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (["all"], (True, True)),
        (["bm25"], (True, False)),
        (["fts"], (True, False)),
        (["embeddings"], (False, True)),
        (["bm25", "embeddings"], (True, True)),
    ],
)
def test_resolve_force_targets(targets, expected):
    assert resolve_force_targets(targets) == expected


def test_resolve_force_targets_rejects_unknown_target():
    with pytest.raises(Exception, match="unknown --force target"):
        resolve_force_targets(["wat"])


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (False, False, [])),
        (["--force", "all"], (True, True, [])),
        (["--force=all"], (True, True, [])),
        (["--force=bm25"], (True, False, [])),
        (["--force=bm25,embeddings"], (True, True, [])),
        (["--force", "bm25", "embeddings"], (True, True, [])),
        (["--force", "bm25", "--help"], (True, False, ["--help"])),
    ],
)
def test_extract_force_flags(argv, expected):
    assert extract_force_flags(argv) == expected
