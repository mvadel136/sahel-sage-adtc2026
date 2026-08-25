# Sahel Sage 🌾

**An offline agricultural expert for the hardware the Sahel actually has.**

Sahel Sage is an on-device language model + advisory console for agricultural
extension officers, cooperatives and agro-pastoral communities in Mauritania
and across the Sahel, where connectivity is scarce, cloud AI is unaffordable,
and the region's one shared laptop must do everything.

Built for the [Africa Deep Tech Challenge 2026](https://adtc-2026.devpost.com)
(Laptop LLM track): 100% offline, CPU-only, within an 8 GB RAM laptop profile,
running on `llama.cpp` with GGUF weights.

## What it does

- Answers practical agro-pastoral questions (livestock health, pasture and
  drought tactics, cropping, storage, planting-time and selling-time decisions)
  **entirely offline**, in English.
- The advisory console (`app/`) grounds answers in a packaged 56-manual
  agronomy and livestock-health library and **cites the source manual for
  every recommendation**, declining to answer when the library does not cover
  a question.
- Seven hazard classes (pesticide doses, veterinary drugs, container reuse,
  minors spraying, and more) are **refused in code before the model runs**,
  see REPORT.md and `src/sahel_sage/inference/safety.py`.

## Quick start

```bash
bash download_model.sh        # fetch the GGUF weights (~0.3 GB)
# profile it exactly like the ADTC evaluation:
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
adtc-profiler run --submission . --mode participant --output submission.json
```

Run the advisory console, see [`app/README.md`](app/README.md). It needs a
`llama-server` binary from [llama.cpp](https://github.com/ggml-org/llama.cpp)
(any recent release; set `SAHEL_LLAMA_SERVER` if it is not on PATH):

```bash
uv sync --all-extras && uv run sage app run     # http://127.0.0.1:8090
```

Run the test suite:

```bash
uv sync --all-extras && uv run pytest
```

## What it looks like

Screenshots of the live console (offline, on this laptop) are in
[`docs/screenshots/`](docs/screenshots/) and in [REPORT.md §7](REPORT.md).
The two-minute demonstration video accompanies the Devpost submission.

## What is in this repository

```
metadata.json        ADTC submission metadata (team, domain, test prompts, model)
download_model.sh    Downloads the GGUF weights (public Hugging Face repo)
REPORT.md            Technical report: problem, design decisions, benchmarks
model/               Weights land here (never committed)
app/                 Offline advisory console (UI + packaged manual library)
src/, tests/         The package and its test suite
configs/             Model path, retrieval and confidence settings
data/reference/      The verified fact base the model ships with (auditable)
```

## Team

Sahel Sage, Mauritania 🇲🇷 · ADTC 2026 team `sahel-sage`
