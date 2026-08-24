# The advisory console

The offline product around the model: SQLite FTS5 retrieval over 56 field
manuals (`library.db`, 11,050 passages — built from the manual corpus, which is not
shipped; the database ships ready to use), citation-first answers, and refusal instead of
guessing when the library does not cover a question.

## Run it

From the repository root:

```bash
uv sync --all-extras
uv run sage app run          # http://127.0.0.1:8090
```

It needs a `llama-server` binary from
[llama.cpp](https://github.com/ggml-org/llama.cpp) (any recent release).
Set `SAHEL_LLAMA_SERVER=/path/to/llama-server` if it is not on PATH.

The model file must be present first: `bash download_model.sh`.

## What lives where

| path | what |
|---|---|
| `src/sahel_sage/app/api.py` | FastAPI routes: `/`, `/api/status`, `/api/ask` (SSE) |
| `src/sahel_sage/app/service.py` | retrieve → gate → generate → verify pipeline |
| `src/sahel_sage/app/backend.py` | llama-server subprocess owner |
| `src/sahel_sage/inference/safety.py` | the seven hazard refusals (code, not prompt) |
| `app/ui/index.html` | the single-file console UI |
| `app/library.db` | FTS5 index of the 56 manuals |

Answers are English — the submission's declared language scope.
