"""Parse `index --force` targets."""

from __future__ import annotations

import argparse

_FORCE_ALIASES = {
    "all": frozenset({"bm25", "embeddings"}),
    "bm25": frozenset({"bm25"}),
    "fts": frozenset({"bm25"}),
    "tokens": frozenset({"bm25"}),
    "token": frozenset({"bm25"}),
    "embeddings": frozenset({"embeddings"}),
}


def split_force_targets(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def resolve_force_targets(targets: list[str]) -> tuple[bool, bool]:
    selected: set[str] = set()
    for target in targets:
        alias = _FORCE_ALIASES.get(target)
        if alias is None:
            valid = ", ".join(sorted(_FORCE_ALIASES))
            raise argparse.ArgumentTypeError(f"unknown --force target {target!r} (expected one of: {valid})")
        selected.update(alias)
    return "bm25" in selected, "embeddings" in selected


def extract_force_flags(argv: list[str] | None) -> tuple[bool, bool, list[str]]:
    """Return (force_bm25, force_embeddings, argv without --force flags)."""
    if not argv:
        return False, False, []

    remaining: list[str] = []
    force_bm25 = False
    force_embeddings = False
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--force":
            index += 1
            targets: list[str] = []
            while index < len(argv) and not argv[index].startswith("-") and "=" not in argv[index]:
                targets.extend(split_force_targets(argv[index]))
                index += 1
            if not targets:
                raise SystemExit("index: --force requires a target (all, bm25, embeddings)")
            bm25, embeddings = resolve_force_targets(targets)
            force_bm25 |= bm25
            force_embeddings |= embeddings
            continue

        if arg.startswith("--force="):
            targets = split_force_targets(arg.split("=", 1)[1])
            if not targets:
                raise SystemExit("index: --force requires a target (all, bm25, embeddings)")
            bm25, embeddings = resolve_force_targets(targets)
            force_bm25 |= bm25
            force_embeddings |= embeddings
            index += 1
            continue

        remaining.append(arg)
        index += 1

    return force_bm25, force_embeddings, remaining
