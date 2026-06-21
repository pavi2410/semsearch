import sys

import pytest

from semsearch.index.indexer import main


def test_main_reads_force_flags_from_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["index", "--force=embeddings"])
    monkeypatch.setattr("semsearch.index.indexer.init_db", lambda: None)
    monkeypatch.setattr("semsearch.index.indexer.iter_page_metas", lambda: [])
    monkeypatch.setattr("semsearch.index.indexer.filter_pages_with_content", lambda pages: ([], 0))
    monkeypatch.setattr("semsearch.index.indexer.load_previous_doc_ids", lambda: [])

    main()

    output = capsys.readouterr().out
    assert "Force embeddings — ignoring embedding cache" in output
