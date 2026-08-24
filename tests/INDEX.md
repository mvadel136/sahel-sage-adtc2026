# tests/ — Index

Run: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, `-q`).
Coverage source: `src/sahel_sage`.

## Strategy

- **Unit tests mirror `src/`**: `tests/unit/<subpackage>/test_<module>.py`.
- **No network in tests.** Fetch code is tested against local files/stubs;
  nothing downloads. (A `SAHEL_OFFLINE` mode to hard-enforce this is planned,
  not implemented.)
- **Golden files planned, not present.** `core/prompts.py` refers to golden
  renderings in `tests/golden/prompts/`; that directory does not exist yet.
- Shared fixture: `tests/unit/retrieval/conftest.py` builds a tiny 3-doc
  corpus (EN millet, FR goat, EN maize + one too-short file) into a tmp
  SQLite library for indexer/query/search tests.
- Shared fixture: `tests/smoke/conftest.py` builds a one-doc library and wires
  a fake backend into `app.api.create_app` — the console is exercised end to
  end over HTTP with **no subprocess, no socket, no llama-server**.
- Legacy `eval/` and `training/*.py` scripts have **no tests**; coverage
  arrives as each piece is ported into the package.

## Test-to-component map (114 tests)

| Test file | Covers | Count |
|---|---|---|
| `unit/test_config.py` | `core/config` (toml load, env overrides) | 3 |
| `unit/test_textproc.py` | `core/textproc` (clean, sections, chunker) | 7 |
| `unit/data/test_sources.py` | `data/sources` registry + validation | 6 |
| `unit/data/test_fetch.py` | `data/fetch` | 5 |
| `unit/data/test_extract.py` | `data/extract` (pdf/html sniff + extract) | 7 |
| `unit/data/test_status.py` | `data/status` | 3 |
| `unit/data/test_splits.py` | `data/splits` holdout guard | 4 |
| `unit/data/test_kallaama.py` | `data/kallaama` segment extraction | 2 |
| `unit/retrieval/test_indexer.py` | `retrieval/indexer` | 2 |
| `unit/retrieval/test_query.py` | `retrieval/query` sanitize/expand | 4 |
| `unit/retrieval/test_search.py` | `retrieval/store` + `rank` + `evidence` | 6 |
| `unit/inference/test_contract.py` | `inference/contract` parse/repair | 9 |
| `unit/training/test_mixer.py` | `training/mixer` (incl. holdout abort) | 2 |
| `unit/app/test_service.py` | `app/service` pipeline, status invariant, repair budget | 14 |
| `unit/app/test_backend.py` | `app/backend` raw-completions HTTP shape | 3 |
| `smoke/test_app_api.py` | `app/api` routes + SSE event order | 7 |
