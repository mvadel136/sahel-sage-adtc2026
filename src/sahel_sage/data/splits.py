"""The train/eval leakage guard.

data/splits/holdout.json was frozen on 2026-08-11 and is never edited; the
documents it names must not appear in any training mix, distillation batch,
or calibration corpus. Every layer that consumes documents imports
:func:`assert_no_holdout` and calls it on its input ids, leakage becomes a
hard crash instead of a silent eval-inflating bug.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from sahel_sage.core.config import repo_root

HOLDOUT_PATH = Path("data") / "splits" / "holdout.json"


class HoldoutViolation(RuntimeError):
    """A holdout document reached a training-side consumer."""


def load_holdout(path: Path | None = None) -> set[str]:
    """-> the frozen holdout doc ids (default: data/splits/holdout.json)."""
    path = path or repo_root() / HOLDOUT_PATH
    data = json.loads(Path(path).read_text())
    ids = set(data["doc_ids"])
    if not ids:
        # an empty holdout means the guard silently guards nothing
        raise ValueError(f"holdout file {path} contains no doc_ids")
    return ids


def assert_no_holdout(doc_ids: Iterable[str], holdout: set[str]) -> None:
    """Raise :class:`HoldoutViolation` if any id is in the frozen holdout."""
    offending = sorted(set(doc_ids) & holdout)
    if offending:
        raise HoldoutViolation(
            "holdout documents leaked into a training-side consumer: " + ", ".join(offending)
        )
