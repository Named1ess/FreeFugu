#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""
eval_perstep_real.py — does a trained per-step TRINITY head actually beat
the best single worker on held-out GSM8K questions, with REAL API workers?

Strategies compared:
  1. each worker ALONE   — every question -> that one worker, single-shot
  2. trained coordinator — FuguRouter + trinity_perstep.npy, full TRINITY loop
  3. oracle (optional)   — try every worker per question, take the best

Usage:
  python eval/eval_perstep_real.py \\
    --model Qwen/Qwen3-0.6B \\
    --vector artifacts/model_iter_identity.npy \\
    --head artifacts/trinity_perstep.npy \\
    --slot-config-env FUGU_SLOT_CONFIG \\
    --n-tasks 16 --skip-train-offset
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
OPENFUGU = ROOT / "openfugu"
TRAIN = ROOT / "train"
sys.path.insert(0, str(OPENFUGU))
sys.path.insert(0, str(TRAIN))

from mini import (  # noqa: E402
    Coordinator,
    FuguRouter,
    HEAD_ROWS,
    HIDDEN,
    LiteLLMWorker,
    N_AGENTS,
    SYSTEM_PROMPT,
    VEC_LEN,
)
from slot_config import SlotSpec, check_litellm_connectivity, load_slot_specs, slot_labels  # noqa: E402
from train_trinity_perstep import (  # noqa: E402
    WorkerBuild,
    build_worker,
    ensure_base_vector,
    gold_answer,
    numeric_answer,
    resolve_router_device,
)


def load_gsm8k_test_tasks(n_tasks: int, skip: int = 0) -> list[tuple[str, str]]:
    from datasets import load_dataset

    end = skip + n_tasks
    ds = load_dataset("openai/gsm8k", "main", split=f"test[{skip}:{end}]")
    tasks = [(row["question"], gold_answer(row["answer"])) for row in ds]
    if not tasks:
        raise ValueError("--n-tasks selected zero tasks")
    return tasks


def run_single_worker(worker, slot_index: int, tasks: list[tuple[str, str]],
                      max_tokens: int) -> tuple[int, int]:
    """Every question -> one fixed worker, single-shot (no TRINITY loop)."""
    scores = run_single_worker_scores(worker, slot_index, tasks, max_tokens)
    return sum(scores), len(scores)


def run_single_worker_scores(worker, slot_index: int, tasks: list[tuple[str, str]],
                             max_tokens: int) -> list[bool]:
    """Every question -> one fixed worker, preserving per-worker question order."""
    scores = []
    msgs_template = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, gold in tasks:
        msgs = msgs_template + [{"role": "user", "content": question}]
        reply = worker("Worker", msgs, slot_index)
        ok = numeric_answer(reply) == gold
        scores.append(ok)
    return scores


def _worker_parallelism(requested: int, n_slots: int) -> int:
    if requested <= 0:
        return max(1, n_slots)
    return max(1, min(requested, n_slots))


def run_single_workers_parallel(worker, n_slots: int, tasks: list[tuple[str, str]],
                                max_tokens: int, max_workers: int) -> list[list[bool]]:
    """Run worker slots in parallel; each slot still walks tasks sequentially."""
    parallelism = _worker_parallelism(max_workers, n_slots)
    scores_by_slot: list[list[bool] | None] = [None] * n_slots
    if parallelism == 1:
        for slot in range(n_slots):
            scores_by_slot[slot] = run_single_worker_scores(worker, slot, tasks, max_tokens)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(run_single_worker_scores, worker, slot, tasks, max_tokens): slot
                for slot in range(n_slots)
            }
            for future in as_completed(futures):
                slot = futures[future]
                scores_by_slot[slot] = future.result()
    missing = [slot for slot, scores in enumerate(scores_by_slot) if scores is None]
    if missing:
        raise RuntimeError(f"missing single-worker results for slots: {missing}")
    return [scores for scores in scores_by_slot if scores is not None]


def run_coordinator(router, worker, tasks: list[tuple[str, str]],
                    max_turns: int) -> tuple[int, int]:
    """Trained coordinator: full TRINITY per-step loop."""
    solved = 0
    for question, gold in tasks:
        coord = Coordinator(router, worker, max_turns=max_turns, sample=False)
        result = coord.run(question)
        if numeric_answer(result.final) == gold:
            solved += 1
    return solved, len(tasks)


def run_oracle(worker, n_slots: int, tasks: list[tuple[str, str]],
               max_tokens: int) -> tuple[int, int]:
    """Ceiling: try every worker per question, take the best."""
    solved = 0
    msgs_template = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, gold in tasks:
        got = False
        for slot in range(n_slots):
            msgs = msgs_template + [{"role": "user", "content": question}]
            reply = worker("Worker", msgs, slot)
            if numeric_answer(reply) == gold:
                got = True
                break
        if got:
            solved += 1
    return solved, len(tasks)


def run_oracle_from_worker_scores(scores_by_slot: list[list[bool]]) -> tuple[int, int]:
    """Ceiling from already-collected per-worker answers."""
    if not scores_by_slot:
        return 0, 0
    total = len(scores_by_slot[0])
    solved = sum(any(slot_scores[i] for slot_scores in scores_by_slot) for i in range(total))
    return solved, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate trained per-step head vs single workers on held-out GSM8K."
    )
    ap.add_argument("--router-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--router-device", default="auto", help="auto, cpu, cuda:0, ...")
    ap.add_argument("--vector", default=str(ROOT / "artifacts" / "model_iter_identity.npy"),
                    help="base 19456-float vector (SVF + head)")
    ap.add_argument("--head", default=str(ROOT / "artifacts" / "trinity_perstep.npy"),
                    help="trained 10240-float head (overrides head in --vector)")
    ap.add_argument("--slot-models", metavar="CSV", help="LiteLLM model ids; max 7")
    ap.add_argument("--slot-config", metavar="JSON", help="JSON file with LiteLLM slot configs")
    ap.add_argument("--slot-config-env", metavar="ENV", help="env var with LiteLLM slot configs JSON")
    ap.add_argument("--local-models", metavar="CSV", help="local HF worker paths")
    ap.add_argument("--n-tasks", type=int, default=16, help="number of held-out GSM8K test questions")
    ap.add_argument("--skip-train-offset", action="store_true", default=True,
                    help="use GSM8K test split (not train) so questions are unseen")
    ap.add_argument("--skip", type=int, default=0,
                    help="offset into GSM8K test split (default 0)")
    ap.add_argument("--max-turns", type=int, default=4, help="coordinator max_turns")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--worker-parallelism", type=int, default=0,
                    help="parallel worker slots for single-worker/oracle baselines; 0 = all slots")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-oracle", action="store_true", help="skip oracle (saves API calls)")
    args = ap.parse_args(argv)

    import torch  # noqa: F401

    vector = ensure_base_vector(args.vector)
    worker_build = build_worker(args)
    worker, labels = worker_build.worker, worker_build.labels
    n_slots = len(labels)
    print(f"[eval] worker slots ({n_slots}): {labels}", flush=True)

    print(f"[eval] loading GSM8K test tasks (n={args.n_tasks}, skip={args.skip}) ...", flush=True)
    tasks = load_gsm8k_test_tasks(args.n_tasks, skip=args.skip)
    print(f"[eval] loaded {len(tasks)} held-out test tasks", flush=True)

    device = resolve_router_device(args.router_device)
    print(f"[eval] loading router: {args.router_model} device={device or 'cpu'}", flush=True)
    router = FuguRouter(args.router_model, vector, device=device, seed=args.seed)
    head_path = Path(args.head)
    if not head_path.exists():
        raise FileNotFoundError(f"trained head not found: {head_path}")
    h = np.load(head_path).astype(np.float64)
    if h.shape != (HEAD_ROWS * HIDDEN,):
        raise ValueError(f"--head must be {HEAD_ROWS * HIDDEN} floats, got {h.shape}")
    router.head = router.torch.from_numpy(h.copy()).float().reshape(HEAD_ROWS, HIDDEN).to(router.device)
    print(f"[eval] applied trained head from {head_path}", flush=True)

    print("\n[eval] === strategy 1: each worker ALONE (single-shot) ===", flush=True)
    single_results: list[tuple[str, float]] = []
    scores_by_slot = run_single_workers_parallel(
        worker,
        n_slots,
        tasks,
        args.max_tokens,
        args.worker_parallelism,
    )
    for slot in range(n_slots):
        solved, total = sum(scores_by_slot[slot]), len(scores_by_slot[slot])
        rate = solved / total
        single_results.append((labels[slot], rate))
        print(f"  {labels[slot]:30s} alone: {rate:.3f}  ({solved}/{total})", flush=True)
    best_single = max(rate for _, rate in single_results)
    best_label = next(label for label, rate in single_results if rate == best_single)

    print("\n[eval] === strategy 2: trained coordinator (full TRINITY loop) ===", flush=True)
    coord_solved, coord_total = run_coordinator(router, worker, tasks, args.max_turns)
    coord_rate = coord_solved / coord_total
    print(f"  coordinator                   : {coord_rate:.3f}  ({coord_solved}/{coord_total})", flush=True)

    oracle_rate: float | None = None
    if not args.no_oracle:
        print("\n[eval] === strategy 3: oracle (try all, take best) ===", flush=True)
        oracle_solved, oracle_total = run_oracle_from_worker_scores(scores_by_slot)
        oracle_rate = oracle_solved / oracle_total
        print(f"  oracle                        : {oracle_rate:.3f}  ({oracle_solved}/{oracle_total})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"held-out evaluation: {len(tasks)} GSM8K test questions", flush=True)
    print("-" * 60, flush=True)
    for label, rate in single_results:
        star = "  <- best single" if rate == best_single else ""
        print(f"  {label:30s} alone : {rate:.3f}{star}", flush=True)
    print(f"  {'trained coordinator':30s}      : {coord_rate:.3f}", flush=True)
    if oracle_rate is not None:
        print(f"  {'oracle (ceiling)':30s}      : {oracle_rate:.3f}", flush=True)

    lift = (coord_rate - best_single) / best_single * 100 if best_single > 0 else float("inf")
    print("-" * 60, flush=True)
    print(f"[result] coordinator {coord_rate:.3f} vs best single {best_single:.3f} "
          f"({best_label})  ->  {lift:+.0f}%", flush=True)
    if oracle_rate is not None:
        frac = coord_rate / oracle_rate * 100 if oracle_rate > 0 else 0
        print(f"[result] coordinator reaches {frac:.0f}% of oracle ceiling", flush=True)

    if coord_rate > best_single + 0.02:
        print("PASS — trained coordinator beats the best single worker on held-out questions", flush=True)
    else:
        print("FAIL — coordinator did not beat the best single worker "
              "(try more training data / iters, or check worker diversity)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
