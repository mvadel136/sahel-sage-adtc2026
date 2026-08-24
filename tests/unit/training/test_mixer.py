import json

import pytest

from sahel_sage.data.splits import HoldoutViolation
from sahel_sage.training.mixer import build_dataset


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _pair(i, doc="fao-x", critique="pass", q=None, a=None):
    return {
        "id": f"c{i}#0",
        "kind": "grounded_chunk",
        "q": q or f"My millet has spots, question {i}?",
        "a": a or f"LIKELY ISSUE: spots {i} [1]\nSTATUS: ANSWERED",
        "meta": {"source_docs": [doc], "passage_ids": [f"c{i}"], "lang": "en",
                 "teacher": "t", "critique": critique, "cluster": "crops"},
    }


@pytest.fixture
def replay_dir(tmp_path):
    d = tmp_path / "replay"
    d.mkdir()
    _write_jsonl(d / "arc_train.jsonl",
                 [{"text": f"Question: q{i}\nAnswer: a{i}", "meta": {}} for i in range(3)])
    _write_jsonl(d / "smoltalk.jsonl", [
        {"messages": [{"role": "user", "content": "hello there"},
                      {"role": "assistant", "content": "hi, how can I help?"}], "meta": {}},
        {"messages": [{"role": "user", "content": "solve $x^2=4$"},
                      {"role": "assistant", "content": "x=\\boxed{2}"}], "meta": {}},
    ])
    return d


def test_mix_filters_critique_dedups_and_writes(tmp_path, replay_dir):
    src = tmp_path / "pairs.jsonl"
    _write_jsonl(src, [
        _pair(1),
        _pair(2, critique="reject"),           # dropped: failed critique
        _pair(3, q="My millet has spots, question 1?"),  # dropped: dup question
    ])
    stats = build_dataset("test", [src], replay_dir, out_root=tmp_path / "mix")
    assert stats["strata"]["grounded_chunk"] == 2  # counted pre-dedup
    assert stats["strata"]["skip_critique"] == 1
    assert stats["strata"]["dedup_q"] == 1
    # math-flavored smoltalk record filtered out
    assert stats["strata"]["replay_chat"] == 1
    out = tmp_path / "mix" / "dataset-test"
    lines = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert stats["total"] == len(lines) == 1 + 3 + 1  # 1 pair + 3 arc + 1 chat
    assert (out / "manifest.json").exists() and (out / "stats.json").exists()


def test_mix_aborts_on_holdout_leakage(tmp_path, replay_dir):
    src = tmp_path / "pairs.jsonl"
    _write_jsonl(src, [_pair(1, doc="fao-ppr-field-manual")])  # a real holdout doc
    with pytest.raises(HoldoutViolation):
        build_dataset("test", [src], replay_dir, out_root=tmp_path / "mix")
