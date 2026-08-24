"""sage — the single command entrypoint.

Subcommands map 1:1 to library functions; anything not yet migrated says so
and points at the legacy script instead of failing cryptically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sahel_sage.core.config import repo_root


def cmd_index_build(args: argparse.Namespace) -> int:
    from sahel_sage.retrieval.indexer import build_index

    stats = build_index(args.db, args.txt_dir, args.sources)
    print(json.dumps(stats))
    return 0


def cmd_index_stats(args: argparse.Namespace) -> int:
    from sahel_sage.retrieval.store import open_retriever

    print(json.dumps(open_retriever(args.db).stats()))
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    from sahel_sage.data.sources import SourceRegistry
    from sahel_sage.data.status import corpus_status, format_table

    reg = SourceRegistry.load(args.sources)
    rows = corpus_status(reg, args.raw_dir, args.txt_dir)
    print(format_table(rows))
    return 0


def cmd_dataset_mix(args: argparse.Namespace) -> int:
    from sahel_sage.training.mixer import build_dataset

    stats = build_dataset(
        args.version,
        [Path(p) for p in args.distilled],
        args.replay_dir,
        wolof_path=args.wolof,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_imatrix_calib(args: argparse.Namespace) -> int:
    from sahel_sage.training.imatrix import build_calibration

    stats = build_calibration(args.out, args.corpus, args.replay_dir)
    print(json.dumps(stats))
    return 0


def cmd_app_run(args: argparse.Namespace) -> int:
    """Boot the offline console.

    Imports live inside the function so that `sage --help` and every other
    subcommand keep working without the optional ``app`` extra installed.
    """
    import functools

    import uvicorn

    from sahel_sage.app.api import create_app
    from sahel_sage.app.context import build_context, internet_reachable

    if args.strict_offline and internet_reachable():
        print(
            "REFUSING TO START: --strict-offline was requested but this machine "
            "can still reach the internet. Disable networking first."
        )
        return 2

    app = create_app(
        functools.partial(
            build_context,
            model=args.model,
            db=args.db,
            threads=args.threads,
            n_ctx=args.ctx,
            strict_offline=args.strict_offline,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


_NOT_MIGRATED = {
    "eval-arc": "eval/arc_runner.py (legacy)",
    "eval-judge": "eval/judge_eval.py (legacy)",
    "fetch": "the corpus fetcher (not shipped)",
}


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(prog="sage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="offline library (FTS5)")
    ps = p.add_subparsers(dest="sub", required=True)
    b = ps.add_parser("build")
    b.add_argument("--db", type=Path, default=root / "app/library.db")
    b.add_argument("--txt-dir", type=Path, default=root / "training/corpus_txt")
    b.add_argument("--sources", type=Path, default=root / "training/corpus_sources.json")
    b.set_defaults(fn=cmd_index_build)
    s = ps.add_parser("stats")
    s.add_argument("--db", type=Path, default=root / "app/library.db")
    s.set_defaults(fn=cmd_index_stats)

    d = sub.add_parser("data", help="corpus pipeline")
    ds = d.add_subparsers(dest="sub", required=True)
    st = ds.add_parser("status")
    st.add_argument("--sources", type=Path, default=root / "training/corpus_sources.json")
    st.add_argument("--raw-dir", type=Path, default=root / "training/corpus_raw")
    st.add_argument("--txt-dir", type=Path, default=root / "training/corpus_txt")
    st.set_defaults(fn=cmd_data_status)

    m = sub.add_parser("dataset", help="training dataset")
    ms = m.add_subparsers(dest="sub", required=True)
    mx = ms.add_parser("mix")
    mx.add_argument("--version", required=True)
    mx.add_argument("--distilled", nargs="+", required=True)
    mx.add_argument("--replay-dir", type=Path, default=root / "training/replay")
    mx.add_argument("--wolof", type=Path, default=None)
    mx.set_defaults(fn=cmd_dataset_mix)

    im = sub.add_parser("imatrix-calib", help="build imatrix calibration corpus")
    im.add_argument("--out", type=Path, required=True)
    im.add_argument("--corpus", type=Path, default=root / "training/corpus_txt")
    im.add_argument("--replay-dir", type=Path, default=root / "training/replay")
    im.set_defaults(fn=cmd_imatrix_calib)

    ap_app = sub.add_parser("app", help="offline advisory console")
    aps = ap_app.add_subparsers(dest="sub", required=True)
    run = aps.add_parser("run")
    run.add_argument("--model", type=Path, default=None, help="GGUF (default: configs/settings.toml)")
    run.add_argument("--db", type=Path, default=None, help="library.db (default: settings)")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8090)
    run.add_argument("--threads", type=int, default=None)
    # 8192, not 4096: ADR-006 renders all EIGHT retrieved passages into the
    # prompt (~3.7k tokens with the question), and a 4096 window left the
    # answer ~300 tokens of room — silent truncation waiting to happen.
    run.add_argument("--ctx", type=int, default=8192)
    run.add_argument(
        "--strict-offline",
        action="store_true",
        help="refuse to start if the internet is reachable, and re-check every minute",
    )
    run.set_defaults(fn=cmd_app_run)

    legacy = sub.add_parser("legacy", help="where the not-yet-migrated commands live")
    legacy.set_defaults(fn=lambda a: (print(json.dumps(_NOT_MIGRATED, indent=2)), 0)[1])

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
