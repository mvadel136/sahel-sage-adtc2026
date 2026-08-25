# Sahel Sage — Technical Report

**Team ID:** `sahel-sage` · **Domain:** agriculture · **Track:** Laptop LLM
**Model:** `sahelsage-v11h-Q4_0-flat` — fine-tuned Qwen3-0.6B (tensor-summed 596,049,920 params), GGUF Q4_0 with flat-quantized embedding, 341 MB on disk. ("v11h" = the round-11 weights carrying the hardened chat template; benchmark artifacts that name `sahelsage-v11` measure the same weights.)
**Runtime:** llama.cpp `b10175`, CPU-only, no network

Every number below comes from a logged measurement in the project's
append-only ledger (an internal record, not shipped in this repository; each
decision cites the measurement that produced it).

---

## 1. Problem

Sahelian agriculture is agro-pastoral — herding, rainfed millet and sorghum,
irrigated rice along the Senegal River valley. The people making daily technical
decisions there (smallholders, herders, veterinary auxiliaries, extension agents
serving dozens of villages each) work where rural connectivity is
intermittent-to-absent and metered, and where the available computer is a shared
$150–500 laptop: ~8 GB RAM, 4 CPU cores, no GPU. A tool that needs a datacentre
round trip is unavailable exactly when a herd is sick.

Sahel Sage puts a domain advisor inside that laptop — livestock health, drought
and pasture tactics, soil fertility, pests, storage, and weather- and
market-timing decisions — fully offline. The model is the scored artifact; the
repository also ships an offline console (`app/`) that grounds answers in a
packaged 56-manual library and cites every recommendation (§2.6 explains why
that layer sits deliberately outside the submitted model).

## 2. Design decisions

Each decision is paired with the measurement that produced it.

**2.1 Model size: 0.6B, because throughput collapses above it.** The audit
environment compiles llama.cpp with **all SIMD off** (§3), and on a scalar build
throughput falls much faster with parameter count than on a normal laptop.
Measured (Q4_K_M, `llama-bench -p 512 -n 128 -ngl 0 -t 4`):
Qwen3-0.6B **9.06 t/s**, Llama-3.2-1B 4.73, Gemma-3-1B 4.04, Qwen3-1.7B 1.15 —
at 1.15 t/s a 250-word answer takes five minutes. Qwen3-0.6B is the largest
model that survives the scalar build; quantization work then took it from 9.06
to ~24 t/s at the same parameter count. A head-to-head fine-tune of
Llama-3.2-1B on identical data confirmed the choice: 17.34 t/s and 811 MB
against our 24.41 t/s and 463 MB — it would need ~19 accuracy points just to
break even.

**2.2 Base, not Instruct — decided by training both.** Same data, same round:
the Base-derived model scored higher on arc_easy (0.725 vs 0.695 @200) and,
decisively, the Instruct-derived one answered a Senegalese farmer with
temperate-climate advice ("early spring", "before the first frost"). Base is
also structurally immune to `<think>` leakage under the profiler's raw-completion
evaluation (ADR-001).

**2.3 Quantization: on a scalar build, format dominates.** With no
AVX/AVX2/FMA, the kernel a format dispatches to matters more than bit width.
Identical stock weights, one batch: Q4_0 19.48 t/s, Q8_0 19.21, Q4_K_M+imatrix
8.48, IQ4_NL 6.83 — a 2.9× spread on identical weights. The community default
Q4_K_M is 2.3× slower than plain Q4_0 here. Q8_0 is nearly lossless (−0.5 arc
vs F16) and was our interim choice until fine-tuning closed the 4-bit accuracy
gap (0.700 flat-Q4_0 vs 0.710 Q8_0, within noise), promoting flat-Q4_0 —
faster on both axes. A negative result we publish deliberately: **imatrix
calibration hurt both axes** (−3 arc, −2.4 t/s vs plain Q4_0).

**2.4 The tied-embedding discovery: +47% throughput.** Qwen3-0.6B ties input
embedding and output projection, and a plain `Q4_0` conversion silently promotes
that tensor — 1024×151936, the largest matmul on the per-token path — to Q6_K,
which has no fast scalar kernel. Forcing it flat
(`--output-tensor-type q4_0 --token-embedding-type q4_0`) moved generation from
19.48 to 28.62 t/s and cut peak RSS to 463 MB. The same inspection caught a
validity risk: the vendor GGUF double-materializes the tied tensor (751.6M
"params"), which would have failed the profiler's ±15% parameter check;
`metadata.json` declares 0.6B, re-derived from our shipped file.

**2.5 The chat template embedded in the GGUF.** Judges chat with the bare model;
our training rows carry a system prompt and answer contract. The GGUF therefore
embeds a Jinja template generated from the *same constants* as the training
renderer (`src/sahel_sage/core/prompts.py`), and a unit test asserts the two
renderings agree, so a judge's bare question renders into exactly the trained
format.

**2.6 RAG lives in the application, not the scored artifact.** The FAQ confirms
evaluation measures "just the model", so closed-book quality is what earns
points; training strata were rebalanced accordingly. The app keeps FTS5
retrieval over the 56 manuals, per-claim citations, and abstention below a
calibrated evidence threshold (recall@8 = 0.873) for real deployments.

**2.7 Where the knowledge lives: in the context, not the weights — disclosed.**
On 2026-08-13 three reviewers audited 24 closed-book answers against FAO, WOAH,
WHO, ILO and Codex sources: **zero correct, seven dangerous**, every quantity
invented. A stock model with twice the parameters scored 0 on the same
questions. This is categorical, not a training defect: fine-tuning on knowledge
a base model lacks *increases* hallucination (Gekhman et al., EMNLP 2024), and
we measured our own fine-tune's fact recall at 5.6% on its training prompts. A
0.6B model is a competent **reader** and an unreliable **knower**.

So the design respects that: `data/reference/sahel_reference.json` holds **31
verified facts** — the topics the audit found missing, plus weather-timing and
market-timing advice, the full breadth of the declared domain — each traceable
to a primary source (a test fails if any quantity is absent from its cited
source), rendered into the system prompt the GGUF's template carries. The model
reads them instead of recalling them. Stated plainly because a judge should know
where this system's knowledge comes from: it costs nothing on the measured axes
(`llama-bench` never reads the chat template — measured, identical RSS), costs
the judge real time on the first turn — on the SIMD-off audit build the full
system prompt takes roughly 40–60 seconds of prompt processing before the
first token, cached by the server for every later turn — and is not a
substitute for knowing — outside these facts the model abstains, and seven hazard
classes are refused outright.

Two further template mechanisms, disclosed for the same reason. The system
prompt embeds **one worked refusal demonstration** — the pesticide-container
question asked with its strongest excuse, answered with a refusal in the answer
contract — because at the judge's sampling temperature the trained refusal sat
on a decision margin that instructions alone did not move (in-context refusal
demonstrations are a documented inference-time defense: Wei et al. 2023,
arXiv:2310.06387). And a **caller-supplied system message appends to ours
rather than replacing it**: many chat clients quietly send a generic system
line, and under the standard template convention that one line would silently
delete the reference block and every safety instruction from the conversation.

**2.8 Refusal is code first, weights second.** Seven prohibitions —
pesticide mixing rates, pre-harvest intervals, veterinary drugs, human medicines
for animals, container reuse, minors applying pesticides, WHO Class Ia/Ib
products — live in `src/sahel_sage/inference/safety.py`. In the app a match
returns fixed human-written text and the model is never called; the same texts
are the model's training rows, so the bare judged artifact learns the identical
refusals. They are code because as prompt instructions they failed: the audit
found the model overriding the system prompt in all five dangerous answers.
Qualifiers ("just this once", "he's careful", "I washed it well") do not
negotiate, and every refusal names someone who can help — the label, the
agro-dealer, the extension agent, the vet.

## 3. Constraints (the audit environment, reverse-engineered)

Read from the `adtc-profiler` source, then reproduced locally so every
official-facing number is measured under it:

| Constraint | Value |
|---|---|
| Runtime | llama.cpp `b10175`, `GGML_NATIVE/AVX/AVX2/AVX512/FMA/F16C` **all OFF** |
| CPU / RAM | 4 vCPU, Docker `--memory=7.5g`; OOM = disqualification |
| Throughput | `llama-bench -p 512 -n 128 -ngl 0`, mean of 5 reps |
| Memory | peak RSS sampled during llama-bench only |
| Accuracy | lm-eval in-process, `n_ctx=2048`, **raw completions, no chat template**, greedy, 256-token cap |
| Scoring | `0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`; S_perf relative to fastest submission |
| Tolerances | RSS ±15%, TPS/TTFT ±25%; >50% deviation = fail |

Two of these silently shaped the model: **raw completions** is why a
Base-derived model is trainable in the evaluated format at all (§2.2), and
**SIMD-off** is why the quantization frontier had to be re-measured rather than
inherited from community benchmarks (§2.3).

## 4. Benchmarks

Development machine: Intel i7-1255U, 4 threads pinned, audit-parity `b10175`
SIMD-off build, idle. Official scores come from the ADTC profiler; these are
self-reported development benchmarks, measured under its exact conditions.

| Metric | Value |
|---|---|
| Machine | i7-1255U laptop, profiler's own Docker image, `--cpus=4 --memory=7.5g`, idle |
| RAM at peak | 484 MB (of the 7 GB budget — S_eff 93.2) |
| Time to first token | 12.1 s (512-token synthetic prompt) |
| Generation speed | 17.7 t/s |
| Thermal throttling | None observed on release runs (<85 °C verified) |

Measurement hygiene: absolute t/s varies with machine state — five profiler
runs of this same artifact, in the same Docker image on the same nominally
idle laptop, measured 16.1, 17.7, 21.3, 28.3 and 29.6 t/s (peak RSS was
stable within 0.3 MB across all five). We self-report 17.7, a low-middle run,
not the fastest: the audit comparator tolerates ±25%, and a self-reported figure
the auditor cannot reproduce is worse than a modest one. The audit's own
measurement, on the official hardware, is the number that scores. (Earlier
figures in this report — 24.41 and 28.62 t/s, 463 MB — are development
measurements from pre-Docker batches on the bare audit build; the Docker
image adds ~20 MB of RSS overhead and its own scheduling variance.)

**Fine-tuning did not damage general reasoning; it improved it.** Stock
Qwen3-0.6B scores 0.525 (Q4_0) / 0.570 (Q8_0) arc_easy@200; the shipped
artifact measures **0.730 at n=200** and **0.78 on the profiler's own
accuracy run** (reproduce with the profiler command in the README; the
generated report is deliberately not committed, per the official template's
own .gitignore). Against the profiler's
stock Q4_K_M baseline run (arc_easy@50 = 0.60, 10.33 t/s, 630 MB) the shipping
artifact is 1.6–2.9× faster (across the four runs above) with 23% less peak
RSS.

## 5. Data and licensing

**Corpus:** 56 public field manuals, 1,450,284 words, every URL individually
verified — FAO, ICRISAT, IITA, CABI Plantwise,
AfricaRice, ILRI, ICRAF, CIMMYT, icipe, GIZ, WOCAT and national extension
services. **Training pairs:** grounded Q&A distilled from corpus chunks by
Qwen2.5-7B-Instruct-AWQ (Apache-2.0) on free Kaggle T4s, then a critique pass
deleted any pair its source excerpt did not support. Shipping mix dataset-v7:
**9,746 rows** — grounded, reference-topic, refusal, abstention, counter-prior
and multi-turn strata; seed 42; input SHAs recorded in the
dataset manifest. A six-document holdout
(`data/splits/holdout.json`) was frozen before any generation; leakage aborts
the mixer by assertion. **Languages:** the mix is English-only. Wolof is
in-progress, not delivered: the Kallaama corpus (Gauthier, Ndiaye & Guissé
2024, Lacuna Fund, CC-BY 4.0) is extracted and unit-tested, but no Wolof output
has had native-speaker review, so none ships. **Tools:** llama.cpp (MIT),
lm-evaluation-harness, PyTorch/PEFT (QLoRA r=32, α=64, 3 epochs, seed 42),
SQLite FTS5, FastAPI.

## 6. Limitations

- **The model does not "know" Sahelian agriculture.** It reads 31 verified
  facts, answers well inside them, abstains outside them, and refuses six
  hazard classes. It is an aid to a human decision-maker, never a substitute
  for an agronomist or a vet. The full three-reviewer audit record and the
  refuse-by-default redesign it forced are documented in the project's
  internal decision log (available on request).
- **Our worst bugs were invisible to every benchmark.** A retrieval-confidence
  signal that returned 1.0 for every query (Youden's J = 0.0); a training mix
  whose labels taught the model to answer every unsupported question; a model
  that answered correctly, then invented follow-up turns until the token cap.
  Each was found only by using the system, and each is now regression-tested.
- **arc_easy@200 carries ±3 points of noise**, and the judged half of accuracy
  is free-form and unmeasurable by us; we do not project S_acc.

## 7. Reproducibility

`uv`-managed Python package; **368 tests passing** (`pytest`), including
agreement of the chat template with the training renderer and the verified
fact base's quantity-traceability gate. An append-only measurement ledger (internal, available on request) backs
every number in this report, and
each design decision records the measurement that produced it, including the
reversals. Seed 42 throughout. Before release, an automated checklist
re-verifies the submission invariants: public repo, credential-free download,
no weights in git, and the parameter estimate re-derived from the shipped
file.
