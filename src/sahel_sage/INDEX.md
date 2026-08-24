# src/sahel_sage — Module Index

Installable package (`pip install -e .`, hatchling). Entrypoints: `sage`
(`cli.main`), `training.imatrix.main`. The console lives here now
(`sage app run`); `eval/` is still legacy and does not import this package.
Tests mirror this tree under `tests/unit/`, plus `tests/smoke/` for the HTTP
surface.

## core/ — shared invariants (no I/O side effects)

| Module | Public interface |
|---|---|
| `textproc` | `chunk(body, *, target_words, overlap_words, min_words) -> list[str]` · `iter_doc_chunks(corpus_dir, *, target_words, overlap_words, min_words, exclude_docs=None) -> Iterator[tuple[str, str]]` — THE chunker. Retrieval uses 220/40/25, distillation 700/80/120. Also `clean_extracted_text`, `split_sections`, `chunk_id`. |
| `prompts` | `system_prompt(has_sources, lang="en") -> str` · `build_messages(question, items, lang="en", max_context_chars=4000) -> list[dict]` · `render_raw(question, items, lang="en", max_context_chars=4000) -> str` — THE prompt registry (train = serve = eval). `Status` enum: ANSWERED / EVIDENCE_LIMITED / OUT_OF_SCOPE. `EvidenceItem` dataclass. Note: docstring references `tests/golden/prompts/`, which does not exist yet. |
| `config` | `load_settings(path=None) -> Settings` — reads `configs/settings.toml`, then applies `SAHEL_<SECTION>_<KEY>` env overrides. `Settings` = `{lab: LabPaths, bench: BenchCfg, repo: RepoCfg}`. `repo_root() -> Path`. |

## retrieval/ — offline FTS5 RAG

| Module | Public interface |
|---|---|
| `schema` | `init_db(conn)` — chunks + FTS5 tables. |
| `indexer` | `build_index(db_path, txt_dir, sources_json) -> dict` — corpus_txt → library.db (constants 220w/40 overlap/min 25). |
| `query` | `tokenize`, `expand`, `sanitize_fts_query`, `match_variants(question) -> list[str]`. |
| `rank` | `rrf_fuse(rankings)`, `cap_per_doc`, `pool_size(k)`. |
| `store` | `Retriever(db_path).search(question, lang="en", k=4) -> list[Citation]` · `.stats()` · `.list_documents()` · `NullRetriever` fallback · `open_retriever(db_path)`. |
| `evidence` | `build_pack(citations, threshold=0.35, k=4) -> EvidencePack` — confidence-gated pack; `Citation` dataclass (`label`, `to_dict`). |

## inference/ — answer contract

`contract.parse(text, valid_source_ids=None) -> ParseResult` — parses the
markdown contract (`**Likely issue**` / `**What to do**` / `**Timing**` /
`**Caution**` / `**Sources**`, ADR-005) into `AnswerContract` (`.cited`);
`ParseResult.needs_repair` drives repair/abstain. `Status` is **inferred** from
the answer's language by `contract.infer_status(text)`, not parsed: the visible
`STATUS:` enum was dropped in round 4.

## app/ — offline advisory console (optional `app` extra)

| Module | Public interface |
|---|---|
| `backend` | `ChatBackend` Protocol (`complete`/`stream`, both on a **rendered raw prompt**) · `LlamaServerConfig` · `LlamaServerBackend` — owns a `llama-server` subprocess, talks `/v1/completions` only (the model is Base-trained on `render_raw`; a chat template would be a train/serve mismatch). Binary defaults to `lab.audit_server`. |
| `context` | `AppContext{cfg, backend, retriever, model_path, threads, strict_offline, internet_seen}` · `.offline_enforced` · `build_context(...)` · `internet_reachable()`. |
| `service` | `answer(question, lang, k, ctx) -> AnswerResult` · `answer_stream(...) -> Iterator[("pack"\|"token"\|"result", …)]` · `resolve_status(pack, model_status)` — **the more conservative of retrieval confidence and the model's status wins**; exactly one repair generation is budgeted. |
| `api` | `create_app(build_context) -> FastAPI`: `GET /`, `/api/status`, `/api/library`, `POST /api/search`, `POST /api/ask` (SSE `citations` → `token`* → `contract` → `done`). |
| `ui/index.html` | Single file, no CDN, light/dark; renders the parsed contract as sections with tappable `[n]` chips and three states (ANSWERED / EVIDENCE_LIMITED / OUT_OF_SCOPE). |

## training/ — dataset assembly (holdout-guarded)

- `markdownify.markdownify(answer) -> str | None` · `markdownify_pairs(in_path, out_path) -> dict` · `render_markdown(issue, actions, timing, caution, sources)` — round-3 ALL-CAPS answers → the round-4 markdown contract; drops `STATUS:`, keeps `[n]`.
- `derive.derive_all(pairs_path, out_path, n_closed=1800, n_bare=1800, n_limited=1000, n_scope=450, n_greetings=120, n_multi_turn=1000, seed=42) -> dict` — the judge-path strata: confident closed-book, bare questions, mismatched-evidence abstention, out-of-scope redirects, greetings, and 2-3 turn conversations.
- `normalize.normalize_pairs(in_path, out_path, valid_ids=None, seed=42) -> dict` · `sample_caution(rng, text="", cluster="")` · `CAUTION_VARIANTS` (15, topic-tagged) — anti-repetition boilerplate.
- `render.render_dataset(train_jsonl, chunks_path, out_path, style, dropout=0.2) -> dict` · `render_record(rec, style, chunks, dropout=0.2)` · `drop_system_prompt(rec, rate)` — `bare` rows get no system prompt and no cues, `multi_turn` rows render history and train only the last turn, ~20% of non-grounded rows drop the system prompt.
- `mixer.build_dataset(version, distilled_paths, replay_dir, wolof_path=None, out_root=None, seed=42, replay_chat_cap=1500) -> dict` — dedup + mix + dual chat/raw format; aborts on holdout leakage via `data.splits.assert_no_holdout`.
- `imatrix.build_calibration(out_path, corpus_dir, replay_dir, target_words=120_000, seed=42) -> dict` — 60/25/15 domain/chat/raw calibration text, holdout excluded; `main()` CLI (`python -m sahel_sage.training.imatrix`).

## data/ — corpus lifecycle + leakage guard

- `sources.SourceRegistry.load(path)` / `.by_cluster(name)`; `Source` record.
- `fetch.fetch(url, dest)`, `fetch_source(source, raw_dir)` (build-time network).
- `extract.extract_source(source, raw_dir, txt_dir) -> dict` (pypdf / HTML strip).
- `status.corpus_status(registry, raw_dir, txt_dir)`, `format_table(rows)`.
- `splits.load_holdout(path=None) -> set[str]` · `assert_no_holdout(doc_ids, holdout) -> None` (raises `HoldoutViolation`).
- `kallaama.extract_wolof_segments(kallaama_dir, out_path, checked_only=True, min_words=5) -> dict` — Wolof segments from Kallaama .trs transcriptions.
