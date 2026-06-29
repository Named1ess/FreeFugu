# OpenFugu

**An open, runnable reverse-engineering of Sakana AI's Fugu — the "one model to
command them all" LLM orchestrator.**

Fugu is sold as a single model; it is really a *policy over models* — a tiny
coordinator that, per query, routes work to a pool of frontier LLMs and returns
one answer. Sakana's product and trained weights are closed. OpenFugu rebuilds
the mechanism from the two papers + released artifacts, verifies it against real
weights, trains a Conductor of our own, and serves it behind one OpenAI-compatible
endpoint. Four stages, all working: **read → run → train → serve.**

> Independent reimplementation. Not affiliated with Sakana AI. No third-party
> code/weights are redistributed here — `scripts/fetch_artifacts.py` pulls them
> from their licensed sources. See `NOTICE`.

---

## How it works

A ~0.6B backbone (Qwen3-0.6B) never answers the user. It produces one hidden
state at the penultimate token; a **bias-free linear head** scores each worker;
the top worker is dispatched and *its* reply is returned. ~19.5K trainable
numbers (the head + singular-value-fine-tuning offsets on 9 matrices), optimized
gradient-free (sep-CMA-ES). No worker weights are ever touched — it is
macro-level composition over other people's models. Full math, with an
EXEC/CODE/DATA evidence grade on every claim, in `docs/`.

OpenFugu implements **two orchestration lines**, both faithful to the papers:

| Line | Mechanism | Lives in |
|------|-----------|----------|
| **TRINITY** (fugu-mini) | Per-turn picker: hidden-state → linear head → (worker, role) → dispatch → verifier/MaxTurn loop | `openfugu/mini.py` |
| **Fugu-Ultra** (Conductor) | One-shot workflow DAG: a Conductor LM emits 3 equal-length lists (`model_id` / `subtasks` / `access_list`), executed in topological order over a worker pool | `openfugu/ultra.py` |

Both are served behind one OpenAI-compatible endpoint in `openfugu/serve.py`.

---

## Repository layout

```
FreeFugu/
├── openfugu/              # core engine: inference + serving
│   ├── mini.py            #   TRINITY line — FuguRouter (hidden-state→worker/role logits) + Coordinator (step_trinity multi-turn loop)
│   ├── ultra.py           #   Fugu-Ultra line — Conductor emits a workflow DAG (3-list), executed in topo order
│   ├── serve.py           #   OpenAI-compatible /v1/chat/completions over a litellm worker pool (stdlib http.server only)
│   ├── slot_config.py     #   shared LiteLLM slot config: credentials, retry, connectivity preflight, openai-compatible HTTP fallback
│   └── __init__.py
├── docs/                  # reverse-engineering write-ups (evidence-graded)
│   ├── HOW_FUGU_IS_IMPLEMENTED.md   # full math
│   ├── ARCHITECTURE.md              # investigation log
│   └── handoff.md
├── verify/                # prove the reconstruction is faithful to the released checkpoint
│   ├── verify_37.py       #   37-case batch regression vs model_iter_60.npy (hit-rate + class-prior baseline)
│   ├── verify_trinity2.py #   rigorous SVF-order check: apply offsets in state_dict order, reproduce fixture argmax
│   ├── verify_margin.py   #   logit-margin analysis of the 2 misses + independent logit cross-check
│   └── probe_sep_cma.py   #   sep-CMA-ES mechanism probe (what "separable" means in pycma)
├── train/                 # train our own coordinators (no Sakana weights needed)
│   ├── train_trinity.py             # self-train TRINITY head from scratch, sep-CMA-ES (mock — no GPU/API)
│   ├── train_trinity_real.py        # sep-CMA-ES on REAL data: real hidden states + Novita pool + GSM8K answer-match
│   ├── train_trinity_toolscale.py   # sep-CMA-ES on nvidia/ToolScale (multi-domain tool-use)
│   ├── train_trinity_perstep.py     # per-STEP TRINITY training (Fugu's real granularity — fitness = full multi-turn rollout)
│   ├── train_conductor.py           # GRPO a Conductor (Llama-3.2-3B) on ToolScale → openfugu-conductor-3b
│   ├── train_recursion.py           # recursive topology (mock, +9% w/ headroom)
│   ├── train_recursion_real.py      # REAL recursion: round-0 output fed back into round-1 (GRPO subclass)
│   ├── train_adaptive_pool.py       # adaptive k-of-n pool (mock)
│   ├── train_adaptive_pool_perstep.py  # REAL per-step k-of-n: random subset masked each turn, train over varying offered subsets
│   ├── toolscale_data.py            # ToolScale data + reward (format + action)
│   ├── toolscale.yaml               # hydra config for GRPO
│   ├── grpo_smoke.py                # minimal GRPO smoke test
│   └── recovered_training_loop.py
├── eval/                  # does orchestration actually beat the best single model?
│   ├── eval_orchestration.py    # per-question routing vs best single worker (the central Fugu claim)
│   ├── eval_perstep_real.py     # per-step eval on real workers
│   ├── eval_recursion_real.py   # held-out round-0 vs round-1 (honest TIE)
│   ├── serve_e2e.py             # boots the real server, POSTs a GSM8K question, checks the answer
│   └── ultra_e2e.py             # asserts a parsed + executed Fugu-Ultra workflow
├── pipeline/              # one-command end-to-end loops
│   └── e2e_train_serve.py      # train a fresh head → serve THAT head → verify a live request
├── scripts/               # setup utilities
│   └── fetch_artifacts.py      # pull Qwen3-0.6B + model_iter_60.npy + 37-case fixture (not redistributed)
├── webui/                 # browser control panel over the whole stack
│   ├── app.py              #   /api/* backend (deps install, artifact fetch, demos, serve, train, eval jobs, log streaming)
│   ├── index.html          #   workbench UI
│   └── static/             #   app.css / app.js
├── results/               # run logs + reward curve + honest results README (read this for caveats)
├── assets/                # plotting utilities (e.g. plot_reward_curve.py)
├── openspec/              # OpenSpec change/spec documents
├── requirements.txt
├── LICENSE                # Apache-2.0
└── NOTICE                 # attribution + third-party material + Llama 3.2 license note
```

---

## Quickstart

### 1. Setup

```bash
pip install -r requirements.txt           # torch, transformers, trl, litellm, cma, ...
python scripts/fetch_artifacts.py         # pull Qwen3-0.6B + model_iter_60.npy + fixture (not redistributed)

export FUGU_MODEL=$(...Qwen3-0.6B path...)
export FUGU_VECTOR=$PWD/artifacts/model_iter_60.npy
export FUGU_FIXTURE=$PWD/artifacts/qwen_router_prompt_eval_cases.json
```

### 2. Read — the architecture, evidence-graded

```bash
less docs/HOW_FUGU_IS_IMPLEMENTED.md
```

### 3. Verify — prove the reconstruction is faithful to the checkpoint

```bash
python openfugu/mini.py --self-test       # -> 95% agent / 100% role on the 37-case fixture, real weights
python verify/verify_37.py                # 37-case batch regression vs model_iter_60.npy
```

### 4. Run — route queries

```bash
# offline mock pool
python openfugu/mini.py --demo

# live worker pool via litellm
export FUGU_API_KEY=...  FUGU_BASE_URL=...
python openfugu/mini.py --demo --live \
  --slot-models "novita/deepseek/deepseek-v4-flash,novita/zai-org/glm-5,..."
```

> **API URL note:** `FUGU_BASE_URL` / slot `api_base` is a LiteLLM base URL, not
> the full `/chat/completions` endpoint. Whether the base includes `/v1` depends
> on the provider (OpenAI uses `https://api.openai.com/v1`, DeepSeek uses
> `https://api.deepseek.com`). Anthropic-prefixed models are translated by
> LiteLLM. Live runs preflight a tiny request first.

### 5. Train — our own coordinators

```bash
# TRINITY head from scratch (sep-CMA-ES, mock — no GPU/API; chance→optimal in seconds)
python train/train_trinity.py

# per-STEP TRINITY (Fugu's real granularity — fitness = full multi-turn rollout)
python train/train_trinity_perstep.py

# on REAL data: GSM8K + Novita pool  /  ToolScale multi-domain
python train/train_trinity_real.py
python train/train_trinity_toolscale.py

# Conductor (Fugu-Ultra) via GRPO on ToolScale (8x A800-class; HF generation, no vLLM)
python train/train_conductor.py           # reward 1.21 → 1.64, saves checkpoint

# recursive topology (test-time scaling)
python train/train_recursion.py           # mock: +9% over one-shot (toy policy w/ headroom)
python train/train_recursion_real.py      # REAL: round-0 fed back into round-1

# adaptive k-of-n pool — generalize to arbitrary worker subsets (swap the pool)
python train/train_adaptive_pool.py            # mock: +44% over blind, 94% of oracle
python train/train_adaptive_pool_perstep.py    # REAL per-step: random k-of-n masked each turn
```

### 6. Serve — Fugu as one model

```bash
# API worker pool via litellm
python openfugu/serve.py --slot-models "<csv>" --port 8088
curl localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"flatten a nested list in one line"}]}'

# real end-to-end: TRAINED per-step head + REAL local worker pool, no API
python openfugu/serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
  --head trinity_perstep.npy --local-models "<llama dir>,<gemma dir>" --port 8088

# Fugu-Ultra: the Conductor workflow-DAG executor, fully local (no API)
python openfugu/ultra.py --query "..." --local-conductor <conductor dir> \
  --local-models "<llama dir>,<deepseek dir>"
```

### 7. Pipeline — train → serve → verify in one command

```bash
# trains a fresh head, serves THAT head, verifies a live request
python pipeline/e2e_train_serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
  --local-models "<llama dir>,<gemma dir>" --port 8097

# serve+verify only, reusing an existing head
python pipeline/e2e_train_serve.py --skip-train --head trinity_perstep.npy \
  --model <qwen3-0.6b dir> --local-models "<llama dir>,<gemma dir>"
```

### 8. Eval — does orchestration beat the best single model?

```bash
python eval/eval_orchestration.py        # trained coordinator +107% over best single worker
python eval/eval_perstep_real.py         # per-step eval on real workers
python eval/eval_recursion_real.py       # held-out round-0 vs round-1 → TIE (see results/)
python eval/serve_e2e.py                 # boots server, POSTs a real GSM8K question -> answer 72, PASS
python eval/ultra_e2e.py                 # asserts a parsed, executed Fugu-Ultra workflow
```

### 9. Web UI — a browser control panel over everything

```bash
python webui/app.py        # open http://127.0.0.1:7860/
```

The page can install dependencies, fetch artifacts, run TRINITY/Fugu-Ultra demos,
start the OpenAI-compatible server, launch training/eval jobs, stream logs, and
cancel running jobs.

---

## Results (honest)

Every experiment reports what it actually shows, including ties and failures.
Full logs and caveats live in [`results/`](results/README.md). Headlines:

- **Conductor GRPO** (ToolScale): reward **1.21 → 1.64** over 100 steps; format
  reward saturates in ~3 steps, action reward climbs ~0.27 → ~0.6. Curve:
  `results/conductor_grpo_reward.png`.
- **Orchestration vs best single** (mock, per-question routing): trained
  coordinator **+107%** over best single worker, 100% of oracle. *This is
  query-level routing, NOT Fugu's per-step coordination — see the results caveat.*
- **TRINITY on real data (GSM8K)**: coordinator **ties** the best worker
  (0.917 vs 0.92) — GSM8K is too easy for modern workers, no headroom to beat.
- **TRINITY on ToolScale** (multi-domain): coordinator **> best single**
  (0.152 vs 0.142, +7%) — worker complementarity gives routing signal.
- **Per-STEP TRINITY** (real granularity): base rollout 0.750 → sep-CMA
  **1.000** (n=8, in-sample — mechanism proven, scale-up is next).
- **Adaptive k-of-n** (per-step, random subset masked): 0.625 → **1.000** —
  Fugu's "swap the pool / opt out any provider" promise, made concrete.
- **Recursive Conductor** (real, held-out): round-0 0.617 vs round-1 0.616 —
  **TIE**. Mechanism is real; gain is not, and we say so.
- **Real end-to-end serving**: trained head served over a real local pool
  answers a live GSM8K question correctly (**72**, gold 72) — the
  read→run→train→serve loop closed on real artifacts.
- **Fugu-Ultra local**: a workflow is emitted + executed over the local pool
  (PASS). Honest finding: our GRPO `checkpoint-100` speaks the ToolScale
  tool-call DSL, not the 3-list workflow DSL, so it does NOT drive this
  executor (reported, not hidden).

---

## Trained Conductor weights

The Conductor we trained on ToolScale (a fine-tune of Llama-3.2-3B-Instruct) is
published on HuggingFace, **not** in this repo (Llama 3.2 Community License
applies — see `NOTICE`):

    huggingface.co/di-zhang-fdu/openfugu-conductor-3b   (see model card)

---

## License

Apache-2.0 for all OpenFugu code (`LICENSE`). Third-party material is fetched,
not redistributed; trained weights carry the Llama 3.2 license. See `NOTICE`.
