#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""
train_trinity_perstep.py — train the serving-compatible per-step TRINITY head.

The output is a 10240-float head that can be served with:

  python openfugu/serve.py --model <router> --vector <base-vector> --head <out>

If the base vector is missing, this script creates an identity vector:
SVF offsets are zero, and the initial head is zero. That is the practical path
when the original model_iter_60.npy artifact is unavailable.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
OPENFUGU = ROOT / "openfugu"
sys.path.insert(0, str(OPENFUGU))

from mini import (  # noqa: E402
    Coordinator,
    FuguRouter,
    HEAD_ROWS,
    HIDDEN,
    LiteLLMWorker,
    N_AGENTS,
    VEC_LEN,
)
from slot_config import SlotSpec, check_litellm_connectivity, load_slot_specs, slot_labels  # noqa: E402


DEFAULT_VECTOR = ROOT / "artifacts" / "model_iter_identity.npy"
DEFAULT_HEAD = ROOT / "artifacts" / "trinity_perstep.npy"


def numeric_answer(text: str | None) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return nums[-1] if nums else None


def gold_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def ensure_base_vector(path: str) -> str:
    target = Path(path)
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.zeros(VEC_LEN, dtype=np.float64))
    print(f"[perstep] created identity base vector: {target}", flush=True)
    return str(target)


def resolve_router_device(value: str | None) -> str | None:
    if not value or value == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else None
        except Exception:
            return None
    if value.lower() == "cpu":
        return None
    return value


def parse_local_specs(csv: str) -> list[tuple[str, str, str]]:
    try:
        import torch

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0

    specs: list[tuple[str, str, str]] = []
    for i, entry in enumerate(item.strip() for item in csv.split(",") if item.strip()):
        if "@" in entry:
            path, dev = entry.rsplit("@", 1)
        else:
            path = entry
            dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
        name = os.path.basename(path.rstrip("/\\")) or f"worker-{i}"
        specs.append((name, path, dev))
    return specs


def sanitized_slot_specs(specs: list[SlotSpec] | None) -> list[dict[str, Any]]:
    if not specs:
        return []
    rows = []
    for index, spec in enumerate(specs):
        row: dict[str, Any] = {
            "slot": index,
            "model": spec.model,
            "api_base": spec.api_base or "",
            "has_api_key": bool(spec.api_key),
        }
        if spec.label:
            row["label"] = spec.label
        rows.append(row)
    return rows


def sanitized_csv_slots(models: list[str]) -> list[dict[str, Any]]:
    return [{"slot": index, "model": model} for index, model in enumerate(models)]


def sanitized_local_specs(specs: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [
        {"slot": index, "name": name, "path": path, "device": device}
        for index, (name, path, device) in enumerate(specs)
    ]


class LocalPoolWorker:
    """Local HF worker pool with the same protocol as the serving worker."""

    def __init__(self, specs: list[tuple[str, str, str]], max_new: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not specs:
            raise ValueError("--local-models did not contain any model paths")
        self.torch, self.max_new = torch, max_new
        self.names, self.toks, self.models, self.devs = [], [], [], []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
            self.names.append(name)
            self.toks.append(tk)
            self.models.append(model)
            self.devs.append(dev)

    def __call__(self, role_name: str, messages: list[dict[str, str]], agent_id: int) -> str:
        torch = self.torch
        wid = agent_id % len(self.models)
        tok, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(message["content"] for message in messages)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=self.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


class CachedWorker:
    """Memoize deterministic worker calls across CMA candidates."""

    def __init__(self, worker, n_slots: int):
        self.worker = worker
        self.n_slots = n_slots
        self.cache: dict[tuple[int, str, str], str] = {}

    def __call__(self, role_name: str, messages: list[dict[str, str]], agent_id: int) -> str:
        slot = agent_id % self.n_slots
        key = (slot, role_name, json.dumps(messages, sort_keys=True, ensure_ascii=False))
        if key not in self.cache:
            self.cache[key] = self.worker(role_name, messages, agent_id)
        return self.cache[key]


@dataclass(frozen=True)
class WorkerBuild:
    worker: Any
    labels: list[str]
    source: str
    slots: list[dict[str, Any]]


def build_worker(args: argparse.Namespace) -> WorkerBuild:
    if args.local_models:
        local_specs = parse_local_specs(args.local_models)
        worker = LocalPoolWorker(local_specs, max_new=args.max_tokens)
        labels = [name for name, _, _ in local_specs]
        return WorkerBuild(
            CachedWorker(worker, len(labels)),
            labels,
            "local_models",
            sanitized_local_specs(local_specs),
        )

    specs = load_slot_specs(args.slot_config, args.slot_config_env, min_count=1, max_count=N_AGENTS)
    slots = [item.strip() for item in (args.slot_models or "").split(",") if item.strip()]
    if specs:
        worker = LiteLLMWorker(slot_specs=specs, max_tokens=args.max_tokens, temperature=args.temperature)
        check_litellm_connectivity(
            specs,
            api_key=worker.api_key,
            api_base=worker.api_base,
            label="worker pool",
        )
        labels = slot_labels(specs)
        source = "slot_config"
        slot_rows = sanitized_slot_specs(specs)
    elif slots:
        specs = [SlotSpec(model=model) for model in slots]
        worker = LiteLLMWorker(slot_specs=specs, max_tokens=args.max_tokens, temperature=args.temperature)
        check_litellm_connectivity(
            specs,
            api_key=worker.api_key,
            api_base=worker.api_base,
            label="worker pool",
        )
        labels = slots
        source = "slot_models"
        slot_rows = sanitized_csv_slots(slots)
    else:
        raise ValueError("provide LiteLLM slots via --slot-config-env/--slot-config/--slot-models, or --local-models")

    if len(labels) > N_AGENTS:
        raise ValueError(f"TRINITY supports at most {N_AGENTS} worker slots, got {len(labels)}")
    return WorkerBuild(CachedWorker(worker, len(labels)), labels, source, slot_rows)


def load_gsm8k_tasks(n_train: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n_train}]")
    tasks = [(row["question"], gold_answer(row["answer"])) for row in ds]
    if not tasks:
        raise ValueError("--n-train selected zero tasks")
    return tasks


def write_sidecar_json(
    out: Path,
    args: argparse.Namespace,
    worker_build: WorkerBuild,
    vector: str,
    base_fit: float,
    best_fit: float,
    head_size: int,
) -> Path:
    sidecar = out.with_suffix(".json")
    data = {
        "head_file": str(out),
        "head_floats": head_size,
        "router_model": args.router_model,
        "base_vector": vector,
        "worker_source": worker_build.source,
        "slot_count": len(worker_build.labels),
        "slots": worker_build.slots,
        "training": {
            "dataset": "openai/gsm8k train",
            "n_train": args.n_train,
            "iters": args.iters,
            "max_turns": args.max_turns,
            "sigma0": args.sigma0,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "diagonal_cma": not args.no_diagonal,
        },
        "result": {
            "base_solved": base_fit,
            "best_solved": best_fit,
            "delta": best_fit - base_fit,
        },
        "notes": [
            "API keys are intentionally not stored in this sidecar.",
            "If slot_count is less than 7, TRINITY agent_id maps to slot by agent_id % slot_count.",
        ],
    }
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a serving-compatible per-step TRINITY head.")
    ap.add_argument("--router-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--router-device", default="auto", help="auto, cpu, cuda:0, ...")
    ap.add_argument(
        "--vector",
        default=str(DEFAULT_VECTOR),
        help="base 19456-float vector; created as identity if missing",
    )
    ap.add_argument("--slot-models", metavar="CSV", help="LiteLLM model ids; max 7")
    ap.add_argument("--slot-config", metavar="JSON", help="JSON file with LiteLLM slot configs; max 7")
    ap.add_argument("--slot-config-env", metavar="ENV", help="env var containing LiteLLM slot configs JSON")
    ap.add_argument("--local-models", metavar="CSV", help="local HF worker paths, optionally path@device")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-diagonal", action="store_true", help="use full CMA instead of sep-CMA")
    ap.add_argument("--out", default=str(DEFAULT_HEAD))
    args = ap.parse_args(argv)

    import cma
    import torch

    vector = ensure_base_vector(args.vector)
    worker_build = build_worker(args)
    worker, labels = worker_build.worker, worker_build.labels
    print(f"[perstep] worker slots ({len(labels)}): {labels}", flush=True)

    print("[perstep] loading GSM8K tasks ...", flush=True)
    tasks = load_gsm8k_tasks(args.n_train)
    print(f"[perstep] loaded GSM8K train tasks: {len(tasks)}", flush=True)

    device = resolve_router_device(args.router_device)
    print(f"[perstep] loading router model: {args.router_model} device={device or 'cpu'}", flush=True)
    router = FuguRouter(args.router_model, vector, device=device, seed=args.seed)
    base_head = router.head.detach().cpu().numpy().ravel()
    head_dim = HEAD_ROWS * HIDDEN
    if base_head.shape != (head_dim,):
        raise ValueError(f"base head must be {head_dim} floats, got {base_head.shape}")

    def rollout_solved(head_vec: np.ndarray) -> float:
        router.head = torch.from_numpy(head_vec.copy()).float().reshape(HEAD_ROWS, HIDDEN).to(router.device)
        coord = Coordinator(router, worker, max_turns=args.max_turns, sample=False)
        solved = 0
        for question, gold in tasks:
            result = coord.run(question)
            if numeric_answer(result.final) == gold:
                solved += 1
        return solved / len(tasks)

    base_fit = rollout_solved(base_head)
    print(
        f"[perstep] base rollout solved={base_fit:.3f} "
        f"(n={len(tasks)}, max_turns={args.max_turns})",
        flush=True,
    )

    opts = {"seed": args.seed, "verbose": -9}
    if not args.no_diagonal:
        opts["CMA_diagonal"] = True
    es = cma.CMAEvolutionStrategy(base_head, args.sigma0, opts)
    best_vec, best_fit = base_head, base_fit

    for iteration in range(args.iters):
        candidates = es.ask()
        fits = [rollout_solved(np.asarray(candidate)) for candidate in candidates]
        es.tell(candidates, [-fit for fit in fits])
        best_index = int(np.argmax(fits))
        if fits[best_index] > best_fit:
            best_fit = float(fits[best_index])
            best_vec = np.asarray(candidates[best_index]).copy()
        print(
            f"[iter {iteration}] best_solved={best_fit:.3f} "
            f"(base {base_fit:.3f}, cache={len(worker.cache)})",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, best_vec)
    sidecar = write_sidecar_json(out, args, worker_build, vector, base_fit, best_fit, best_vec.shape[0])
    print(f"\n[result] per-step trained head solved={best_fit:.3f} vs base {base_fit:.3f}")
    print(f"[result] saved {out} ({best_vec.shape[0]} floats)")
    print(f"[result] saved slot metadata {sidecar}")
    if best_fit > base_fit + 0.01:
        print("PASS — per-step sep-CMA improved the router over the rollout baseline")
    else:
        print("NOTE — no improvement over base in this small run; increase --n-train/--iters for a stronger search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
