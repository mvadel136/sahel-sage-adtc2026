# configs/: Index

| file | what |
|---|---|
| `settings.toml` | model path, llama.cpp binary locations, thread defaults |
| `retrieval.toml` | k, RRF constant, and the calibrated confidence threshold, each value carries its measurement in comments |

Overrides: `SAHEL_<SECTION>_<KEY>` environment variables; the console's server
binary can be overridden with `SAHEL_LLAMA_SERVER`.
