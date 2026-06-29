#!/usr/bin/env python3
"""Local Web UI for OpenFugu.

The server intentionally uses only the Python standard library. It exposes a
small whitelist of existing repository scripts, runs them as background jobs,
and serves the static HTML/CSS/JS control panel from this directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEBUI = Path(__file__).resolve().parent
STATIC = WEBUI / "static"
PYTHON = sys.executable
LOG_LIMIT = 2000

MASKED_ENV_KEYS = {
    "FUGU_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
}


def _path(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


def _default(name: str, fallback: str) -> str:
    return os.environ.get(name) or fallback


DEFAULT_MODEL = _default("FUGU_MODEL", "Qwen/Qwen3-0.6B")
DEFAULT_VECTOR = _default("FUGU_VECTOR", _path("artifacts", "model_iter_60.npy"))
DEFAULT_FIXTURE = _default(
    "FUGU_FIXTURE", _path("artifacts", "qwen_router_prompt_eval_cases.json")
)
SLOT_CONFIG_ENV = "FUGU_SLOT_CONFIG"


COMMON_MODEL_FIELDS = [
    {
        "name": "model",
        "label": "路由模型",
        "type": "text",
        "flag": "--model",
        "default": DEFAULT_MODEL,
        "placeholder": "Qwen/Qwen3-0.6B 或本地目录",
    },
    {
        "name": "vector",
        "label": "TRINITY 向量",
        "type": "text",
        "flag": "--vector",
        "default": DEFAULT_VECTOR,
        "placeholder": "artifacts/model_iter_60.npy",
    },
]

CUSTOM_SLOT_FIELD = {
    "name": "slot_config",
    "label": "LiteLLM 槽位",
    "type": "slot_config",
    "flag": "--slot-config-env",
    "env": SLOT_CONFIG_ENV,
    "slots": 7,
    "min_slots": 1,
    "max_slots": 7,
}


OPERATIONS: list[dict[str, Any]] = [
    {
        "id": "mini_self_test",
        "group": "run",
        "title": "TRINITY 自检",
        "badge": "验证",
        "description": "复跑 37 个 routing fixture，检查实现是否贴合 checkpoint。",
        "script": _path("openfugu", "mini.py"),
        "fixed_args": ["--self-test"],
        "fields": COMMON_MODEL_FIELDS
        + [
            {
                "name": "fixture",
                "label": "评测 fixture",
                "type": "text",
                "flag": "--fixture",
                "default": DEFAULT_FIXTURE,
            },
            {
                "name": "seed",
                "label": "随机种子",
                "type": "number",
                "flag": "--seed",
                "default": "0",
            },
        ],
    },
    {
        "id": "mini_route",
        "group": "run",
        "title": "查看路由决策",
        "badge": "运行",
        "description": "输入一个问题，查看会路由到哪个 worker 和角色。",
        "script": _path("openfugu", "mini.py"),
        "fields": COMMON_MODEL_FIELDS
        + [
            {
                "name": "route",
                "label": "问题",
                "type": "textarea",
                "flag": "--route",
                "required": True,
                "default": "flatten a nested list in one line",
            },
            {
                "name": "seed",
                "label": "随机种子",
                "type": "number",
                "flag": "--seed",
                "default": "0",
            },
        ],
    },
    {
        "id": "mini_demo",
        "group": "run",
        "title": "TRINITY 演示",
        "badge": "运行",
        "description": "跑一次完整协调循环，可用 mock worker 或 LiteLLM worker。",
        "script": _path("openfugu", "mini.py"),
        "fixed_args": ["--demo"],
        "fields": COMMON_MODEL_FIELDS
        + [
            {
                "name": "query",
                "label": "问题",
                "type": "textarea",
                "flag": "--query",
                "default": "Implement binary search in Python and prove it terminates.",
            },
            {
                "name": "live",
                "label": "使用 LiteLLM worker",
                "type": "checkbox",
                "flag": "--live",
                "default": False,
            },
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
                "placeholder": "openai/gpt-4o-mini,anthropic/claude-3-5-sonnet",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "seed",
                "label": "随机种子",
                "type": "number",
                "flag": "--seed",
                "default": "0",
            },
        ],
    },
    {
        "id": "serve_openai",
        "group": "serve",
        "title": "启动 OpenAI 兼容服务",
        "badge": "服务",
        "description": "启动 /v1/chat/completions，本任务会持续运行直到取消。",
        "script": _path("openfugu", "serve.py"),
        "long_running": True,
        "fields": [
            {
                "name": "model",
                "label": "路由模型",
                "type": "text",
                "flag": "--model",
                "required": True,
                "default": DEFAULT_MODEL,
            },
            {
                "name": "vector",
                "label": "TRINITY 向量",
                "type": "text",
                "flag": "--vector",
                "default": DEFAULT_VECTOR,
            },
            {"name": "head", "label": "训练好的 head", "type": "text", "flag": "--head"},
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
                "placeholder": "pathA@cuda:1,pathB@cuda:2",
            },
            {
                "name": "port",
                "label": "端口",
                "type": "number",
                "flag": "--port",
                "default": "8088",
            },
            {
                "name": "max_turns",
                "label": "最大轮数",
                "type": "number",
                "flag": "--max-turns",
                "default": "5",
            },
        ],
    },
    {
        "id": "ultra_self_test",
        "group": "ultra",
        "title": "Fugu-Ultra 自检",
        "badge": "验证",
        "description": "离线验证 workflow 解析、DAG 顺序和 mock 执行。",
        "script": _path("openfugu", "ultra.py"),
        "fixed_args": ["--self-test"],
        "fields": [],
    },
    {
        "id": "ultra_run",
        "group": "ultra",
        "title": "运行 Fugu-Ultra",
        "badge": "运行",
        "description": "让 Conductor 生成 workflow，再调度 worker 执行。",
        "script": _path("openfugu", "ultra.py"),
        "require_any": [["conductor", "local_conductor"]],
        "fields": [
            {
                "name": "query",
                "label": "问题",
                "type": "textarea",
                "flag": "--query",
                "required": True,
                "default": "Write a Python function that returns the n-th Fibonacci number.",
            },
            {"name": "conductor", "label": "LiteLLM Conductor", "type": "text", "flag": "--conductor"},
            {
                "name": "local_conductor",
                "label": "本地 Conductor",
                "type": "text",
                "flag": "--local-conductor",
            },
            {
                "name": "conductor_device",
                "label": "Conductor 设备",
                "type": "text",
                "flag": "--conductor-device",
                "default": "cuda:0",
            },
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
            },
        ],
    },
    {
        "id": "train_trinity_mock",
        "group": "train",
        "title": "训练 TRINITY mock",
        "badge": "训练",
        "description": "CPU 友好的 sep-CMA-ES mock 训练。",
        "script": _path("train", "train_trinity.py"),
        "fields": [
            {"name": "iters", "label": "迭代数", "type": "number", "flag": "--iters", "default": "60"},
            {
                "name": "n_tasks",
                "label": "任务数",
                "type": "number",
                "flag": "--n-tasks",
                "default": "64",
            },
            {
                "name": "repeats",
                "label": "重复次数",
                "type": "number",
                "flag": "--repeats",
                "default": "4",
            },
            {
                "name": "sigma0",
                "label": "sigma0",
                "type": "number",
                "flag": "--sigma0",
                "default": "0.3",
            },
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "42"},
            {
                "name": "out",
                "label": "输出文件",
                "type": "text",
                "flag": "--out",
                "default": "trinity_mock.npy",
            },
            {
                "name": "no_diagonal",
                "label": "使用 full CMA",
                "type": "checkbox",
                "flag": "--no-diagonal",
                "default": False,
            },
        ],
    },
    {
        "id": "train_adaptive_mock",
        "group": "train",
        "title": "训练自适应 worker 子集",
        "badge": "训练",
        "description": "mock 训练随机 k-of-n worker 子集路由。",
        "script": _path("train", "train_adaptive_pool.py"),
        "fields": [
            {"name": "k", "label": "可用 worker 数", "type": "number", "flag": "--k", "default": "3"},
            {"name": "iters", "label": "迭代数", "type": "number", "flag": "--iters", "default": "50"},
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "42"},
        ],
    },
    {
        "id": "train_perstep_real",
        "group": "train",
        "title": "训练 per-step head",
        "badge": "高级",
        "description": "真实多轮 rollout 训练，支持前端 LiteLLM 槽位或本地 worker。",
        "script": _path("train", "train_trinity_perstep.py"),
        "fields": [
            {
                "name": "router_model",
                "label": "路由模型",
                "type": "text",
                "flag": "--router-model",
                "default": DEFAULT_MODEL,
            },
            {
                "name": "vector",
                "label": "TRINITY 向量",
                "type": "text",
                "flag": "--vector",
                "default": _path("artifacts", "model_iter_identity.npy"),
            },
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
                "placeholder": "pathA@cuda:1,pathB@cuda:2",
            },
            {"name": "n_train", "label": "训练样本", "type": "number", "flag": "--n-train", "default": "8"},
            {"name": "iters", "label": "迭代数", "type": "number", "flag": "--iters", "default": "6"},
            {
                "name": "max_turns",
                "label": "最大轮数",
                "type": "number",
                "flag": "--max-turns",
                "default": "4",
            },
            {
                "name": "sigma0",
                "label": "sigma0",
                "type": "number",
                "flag": "--sigma0",
                "default": "0.3",
            },
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "42"},
            {
                "name": "out",
                "label": "输出 head",
                "type": "text",
                "flag": "--out",
                "default": _path("artifacts", "trinity_perstep.npy"),
            },
            {
                "name": "export_api_keys",
                "label": "metadata JSON 导出 API Key",
                "type": "checkbox",
                "flag": "--export-api-keys",
            },
            {
                "name": "checkpoint",
                "label": "checkpoint 路径",
                "type": "text",
                "flag": "--checkpoint",
                "default": _path("artifacts", "trinity_perstep.ckpt.pkl"),
            },
            {
                "name": "resume",
                "label": "从 checkpoint 断点续训",
                "type": "checkbox",
                "flag": "--resume",
            },
        ],
    },
    {
        "id": "eval_orchestration",
        "group": "eval",
        "title": "评测 orchestration 收益",
        "badge": "评测",
        "description": "比较单 worker、随机路由、训练后 coordinator 和 oracle。",
        "script": _path("eval", "eval_orchestration.py"),
        "fields": [
            {
                "name": "coordinator",
                "label": "coordinator 文件",
                "type": "text",
                "flag": "--coordinator",
                "default": "trinity_mock.npy",
            },
            {
                "name": "n_tasks",
                "label": "评测任务数",
                "type": "number",
                "flag": "--n-tasks",
                "default": "5000",
            },
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "7"},
            {
                "name": "world_seed",
                "label": "world seed",
                "type": "number",
                "flag": "--world-seed",
                "default": "42",
            },
            {
                "name": "train_iters",
                "label": "缺省训练迭代",
                "type": "number",
                "flag": "--train-iters",
                "default": "60",
            },
        ],
    },
    {
        "id": "eval_perstep_real",
        "group": "eval",
        "title": "评测 per-step 真实收益",
        "badge": "高级",
        "description": "真 API 上对比单 worker、训练后 coordinator 和 oracle（GSM8K 测试集，hold-out）。",
        "script": _path("eval", "eval_perstep_real.py"),
        "fields": [
            {
                "name": "router_model",
                "label": "路由模型",
                "type": "text",
                "flag": "--router-model",
                "default": DEFAULT_MODEL,
            },
            {
                "name": "vector",
                "label": "TRINITY 向量",
                "type": "text",
                "flag": "--vector",
                "default": _path("artifacts", "model_iter_identity.npy"),
            },
            {
                "name": "head",
                "label": "训练 head",
                "type": "text",
                "flag": "--head",
                "default": _path("artifacts", "trinity_perstep.npy"),
            },
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
                "placeholder": "pathA@cuda:1,pathB@cuda:2",
            },
            {"name": "n_tasks", "label": "评测题数", "type": "number", "flag": "--n-tasks", "default": "16"},
            {"name": "skip", "label": "测试集偏移", "type": "number", "flag": "--skip", "default": "0"},
            {"name": "max_turns", "label": "最大轮数", "type": "number", "flag": "--max-turns", "default": "4"},
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "42"},
            {
                "name": "no_oracle",
                "label": "跳过 oracle（省 API）",
                "type": "checkbox",
                "flag": "--no-oracle",
            },
        ],
    },
    {
        "id": "serve_e2e",
        "group": "eval",
        "title": "端到端服务验证",
        "badge": "验证",
        "description": "启动真实服务并 POST 一道 GSM8K 题检查结果，支持 LiteLLM 槽位或本地 worker。",
        "script": _path("eval", "serve_e2e.py"),
        "fields": [
            {
                "name": "model",
                "label": "路由模型",
                "type": "text",
                "flag": "--model",
                "required": True,
                "default": DEFAULT_MODEL,
            },
            {
                "name": "vector",
                "label": "TRINITY 向量",
                "type": "text",
                "flag": "--vector",
                "default": DEFAULT_VECTOR,
            },
            {"name": "head", "label": "训练好的 head", "type": "text", "flag": "--head"},
            {
                "name": "slot_models",
                "label": "LiteLLM worker 模型 CSV",
                "type": "textarea",
                "flag": "--slot-models",
            },
            CUSTOM_SLOT_FIELD,
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
                "placeholder": "pathA@cuda:1,pathB@cuda:2",
            },
            {"name": "port", "label": "端口", "type": "number", "flag": "--port", "default": "8099"},
            {
                "name": "max_turns",
                "label": "最大轮数",
                "type": "number",
                "flag": "--max-turns",
                "default": "4",
            },
        ],
    },
    {
        "id": "pipeline_e2e",
        "group": "pipeline",
        "title": "训练到服务一键流水线",
        "badge": "流水线",
        "description": "串起训练 head、启动服务、真实请求验证。",
        "script": _path("pipeline", "e2e_train_serve.py"),
        "fields": [
            {
                "name": "model",
                "label": "路由模型",
                "type": "text",
                "flag": "--model",
                "required": True,
                "default": DEFAULT_MODEL,
            },
            {
                "name": "vector",
                "label": "TRINITY 向量",
                "type": "text",
                "flag": "--vector",
                "default": DEFAULT_VECTOR,
            },
            {
                "name": "local_models",
                "label": "本地 worker 模型",
                "type": "textarea",
                "flag": "--local-models",
                "required": True,
            },
            {"name": "port", "label": "端口", "type": "number", "flag": "--port", "default": "8099"},
            {
                "name": "max_turns",
                "label": "最大轮数",
                "type": "number",
                "flag": "--max-turns",
                "default": "4",
            },
            {"name": "n_train", "label": "训练样本", "type": "number", "flag": "--n-train", "default": "8"},
            {"name": "iters", "label": "训练迭代", "type": "number", "flag": "--iters", "default": "6"},
            {
                "name": "sigma0",
                "label": "sigma0",
                "type": "number",
                "flag": "--sigma0",
                "default": "0.3",
            },
            {"name": "seed", "label": "随机种子", "type": "number", "flag": "--seed", "default": "42"},
            {
                "name": "skip_train",
                "label": "跳过训练",
                "type": "checkbox",
                "flag": "--skip-train",
                "default": False,
            },
            {"name": "head", "label": "已有 head", "type": "text", "flag": "--head"},
        ],
    },
]

GROUPS = [
    {"id": "all", "label": "全部"},
    {"id": "run", "label": "TRINITY"},
    {"id": "serve", "label": "服务"},
    {"id": "ultra", "label": "Ultra"},
    {"id": "train", "label": "训练"},
    {"id": "eval", "label": "评测"},
    {"id": "pipeline", "label": "流水线"},
]


@dataclass
class Job:
    id: str
    operation_id: str
    title: str
    command: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    status: str = "queued"
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > LOG_LIMIT:
                self.logs = self.logs[-LOG_LIMIT:]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_id": self.operation_id,
            "title": self.title,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def detail(self) -> dict[str, Any]:
        data = self.summary()
        with self.lock:
            data["logs"] = list(self.logs)
        return data


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def operation_by_id(op_id: str) -> dict[str, Any] | None:
    return next((op for op in OPERATIONS if op["id"] == op_id), None)


def has_value(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def looks_like_full_api_endpoint(value: str) -> bool:
    url = value.strip().rstrip("/").lower()
    return url.endswith("/chat/completions") or url.endswith("/messages")


def normalize_slot_config(value: Any, field_def: dict[str, Any], errors: list[str]) -> list[dict[str, str]]:
    max_slots = int(field_def.get("max_slots") or field_def.get("slots") or 7)
    min_slots = int(field_def.get("min_slots") or 1)
    label = field_def.get("label", field_def["name"])
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except json.JSONDecodeError as exc:
            errors.append(f"{label} JSON 无效: {exc.msg}")
            return []
    rows = value if isinstance(value, list) else []

    keys = ("model", "api_base", "api_key")
    touched = any(
        isinstance(row, dict) and any(has_value(row.get(key)) for key in keys)
        for row in rows
    )
    if not touched:
        return []

    if len(rows) < min_slots:
        errors.append(f"{label} 至少需要 {min_slots} 个槽位")
        return []
    if len(rows) > max_slots:
        errors.append(f"{label} 最多支持 {max_slots} 个槽位")
        rows = rows[:max_slots]

    specs: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            row = {}
        model = str(row.get("model") or "").strip()
        api_base = str(row.get("api_base") or "").strip()
        api_key = str(row.get("api_key") or "").strip()
        if not model:
            errors.append(f"{label} 槽位 {index} 的模型不能为空")
            continue
        if api_base and looks_like_full_api_endpoint(api_base):
            errors.append(
                f"{label} 槽位 {index} 的 API URL 请填 base URL；"
                "不要填 /chat/completions 或 /messages。是否包含 /v1 取决于服务商。"
            )
            continue
        spec = {"model": model}
        if api_base:
            spec["api_base"] = api_base
        if api_key:
            spec["api_key"] = api_key
        specs.append(spec)
    return specs


def build_command(op: dict[str, Any], values: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    env_overrides: dict[str, str] = {}
    cmd = list(op.get("argv") or [PYTHON, op["script"]])
    cmd.extend(op.get("fixed_args", []))

    for group in op.get("require_any", []):
        if not any(has_value(values.get(name)) for name in group):
            labels = [
                field.get("label", name)
                for name in group
                for field in op.get("fields", [])
                if field.get("name") == name
            ]
            errors.append("至少填写一项: " + " / ".join(labels or group))

    for field_def in op.get("fields", []):
        name = field_def["name"]
        value = values.get(name, field_def.get("default", ""))
        if field_def.get("type") == "slot_config":
            specs = normalize_slot_config(value, field_def, errors)
            if specs:
                env_name = field_def.get("env") or SLOT_CONFIG_ENV
                env_overrides[env_name] = json.dumps(specs, ensure_ascii=False)
                flag = field_def.get("flag")
                if flag:
                    cmd.extend([flag, env_name])
            continue

        if field_def.get("required") and not has_value(value):
            errors.append(f"{field_def.get('label', name)} 不能为空")
            continue

        flag = field_def.get("flag")
        if field_def.get("type") == "checkbox":
            if bool(value) and flag:
                cmd.append(flag)
            continue

        if not has_value(value):
            continue
        if flag:
            cmd.extend([flag, str(value).strip()])
        else:
            cmd.append(str(value).strip())

    if errors:
        raise ValueError("; ".join(errors))
    return cmd, env_overrides


def run_job(job: Job) -> None:
    env = os.environ.copy()
    env.update(job.env)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONFAULTHANDLER"] = "1"
    job.status = "running"
    job.started_at = time.time()
    job.append("$ " + " ".join(job.command) + "\n")
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        job.process = subprocess.Popen(
            job.command,
            cwd=job.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        assert job.process.stdout is not None
        for line in job.process.stdout:
            job.append(line)
        code = job.process.wait()
        if job.status == "cancelling":
            job.status = "cancelled"
        else:
            job.exit_code = code
            job.status = "succeeded" if code == 0 else "failed"
            if code != 0:
                job.append(f"\n[webui] process exited with code {code} ({code:#x})\n")
    except Exception as exc:  # surfaced in the Web UI log panel
        job.exit_code = -1
        job.status = "failed"
        job.append(f"\n[webui] {type(exc).__name__}: {exc}\n")
    finally:
        job.ended_at = time.time()


def start_job(operation_id: str, values: dict[str, Any]) -> Job:
    op = operation_by_id(operation_id)
    if not op:
        raise ValueError(f"unknown operation: {operation_id}")
    command, env_overrides = build_command(op, values)
    job = Job(
        id=uuid.uuid4().hex[:12],
        operation_id=operation_id,
        title=op["title"],
        command=command,
        cwd=str(ROOT),
        env=env_overrides,
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


def cancel_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if not job:
        raise ValueError("job not found")
    if job.process and job.status == "running":
        job.status = "cancelling"
        pid = job.process.pid
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                job.process.terminate()
        except Exception as exc:
            job.append(f"\n[webui] cancel failed: {exc}\n")
    return job


def get_status() -> dict[str, Any]:
    modules = [
        "numpy",
        "torch",
        "transformers",
        "litellm",
        "huggingface_hub",
        "cma",
        "datasets",
        "trl",
        "math_verify",
    ]
    deps = {name: importlib.util.find_spec(name) is not None for name in modules}
    env: dict[str, Any] = {}
    for key in [
        "FUGU_MODEL",
        "FUGU_VECTOR",
        "FUGU_FIXTURE",
        "FUGU_BASE_URL",
        "OPENAI_BASE_URL",
        *sorted(MASKED_ENV_KEYS),
    ]:
        value = os.environ.get(key)
        if key in MASKED_ENV_KEYS:
            env[key] = "已设置" if value else ""
        else:
            env[key] = value or ""

    artifacts_dir = ROOT / "artifacts"
    vector = Path(DEFAULT_VECTOR)
    fixture = Path(DEFAULT_FIXTURE)
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    return {
        "root": str(ROOT),
        "python": {
            "executable": PYTHON,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "env": env,
        "artifacts": {
            "dir": str(artifacts_dir),
            "dir_exists": artifacts_dir.exists(),
            "vector": str(vector),
            "vector_exists": vector.exists(),
            "fixture": str(fixture),
            "fixture_exists": fixture.exists(),
        },
        "dependencies": deps,
        "jobs": {
            "total": len(jobs),
            "running": sum(1 for job in jobs if job.status in {"running", "queued", "cancelling"}),
        },
    }


def post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    port = int(payload.get("port") or 8088)
    message = str(payload.get("message") or "").strip()
    if not message:
        raise ValueError("message required")
    body = json.dumps({"messages": [{"role": "user", "content": message}]}).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def json_response(handler: BaseHTTPRequestHandler, code: int, data: Any) -> None:
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    ctype = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }.get(path.suffix, "application/octet-stream")
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                send_file(self, WEBUI / "index.html")
            elif path.startswith("/static/"):
                target = (STATIC / path.removeprefix("/static/")).resolve()
                if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                    self.send_error(403)
                else:
                    send_file(self, target)
            elif path == "/api/status":
                json_response(self, 200, get_status())
            elif path == "/api/operations":
                json_response(self, 200, {"groups": GROUPS, "operations": OPERATIONS})
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = sorted(JOBS.values(), key=lambda job: job.created_at, reverse=True)
                json_response(self, 200, {"jobs": [job.summary() for job in jobs]})
            elif path.startswith("/api/jobs/"):
                job_id = path.split("/")[3]
                job = JOBS.get(job_id)
                if not job:
                    json_response(self, 404, {"error": "job not found"})
                else:
                    data = job.detail()
                    query = parse_qs(parsed.query)
                    if query.get("tail"):
                        tail = max(1, int(query["tail"][0]))
                        data["logs"] = data["logs"][-tail:]
                    json_response(self, 200, data)
            else:
                self.send_error(404)
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = read_json(self)
            if path == "/api/jobs":
                job = start_job(str(payload.get("operation_id") or ""), payload.get("values") or {})
                json_response(self, 201, job.summary())
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                job = cancel_job(job_id)
                json_response(self, 200, job.summary())
            elif path == "/api/chat":
                json_response(self, 200, post_chat(payload))
            else:
                self.send_error(404)
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[webui] " + fmt % args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenFugu local Web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if not shutil.which(PYTHON):
        print(f"[webui] Python executable not found: {PYTHON}", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[webui] OpenFugu Web UI: {url}")
    print(f"[webui] repo root: {ROOT}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
