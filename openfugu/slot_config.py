"""Shared LiteLLM slot configuration helpers."""
from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_RETRY_INTERVAL = 5.0
_DEFAULT_HTTP_TIMEOUT = 180.0
_RETRYABLE_HTTP_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_CODES
    if isinstance(
        exc,
        (
            http.client.HTTPException,
            urllib.error.URLError,
            ssl.SSLError,
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    ):
        return True
    msg = str(exc).lower()
    if any(
        k in msg
        for k in (
            "ssl",
            "eof",
            "timeout",
            "timed out",
            "connection",
            "reset",
            "unreachable",
            "incompleteread",
            "incomplete read",
            "chunk",
            "chunked",
        )
    ):
        return True
    return any(f"http {code}" in msg for code in _RETRYABLE_HTTP_CODES)


def _retry_forever(fn, *, label: str = "api", interval: float = _RETRY_INTERVAL, max_retries: int | None = None):
    """Call fn() with unlimited retries on transient/network errors.

    Prints a one-line status on every failure and on recovery so the WebUI log
    panel surfaces "连不上" / "已恢复" in real time. Returns fn()'s return value."""
    attempt = 0
    while True:
        try:
            result = fn()
        except Exception as exc:
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                raise
            if not _is_retryable(exc):
                raise
            kind = type(exc).__name__
            detail = str(getattr(exc, "reason", exc))[:200]
            print(
                f"[retry] {label} 连不上: {kind}: {detail} — "
                f"第 {attempt} 次重试，{interval:.0f}s 后再试",
                flush=True,
            )
            time.sleep(interval)
            continue
        if attempt > 0:
            print(f"[retry] {label} 已恢复 — 第 {attempt + 1} 次尝试成功", flush=True)
        return result


@dataclass(frozen=True)
class SlotSpec:
    model: str
    api_base: str | None = None
    api_key: str | None = None
    label: str | None = None


OPENAI_COMPAT_HINT = (
    "API URL is passed to LiteLLM as api_base. For OpenAI-compatible servers, "
    "enter the provider base URL, not the full /chat/completions endpoint. "
    "Some providers use /v1 in the base URL and some do not."
)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _provider_name(model: str | None = None, api_base: str | None = None) -> str:
    model_text = (model or "").lower()
    base_text = (api_base or "").lower()
    if "/" in model_text:
        return model_text.split("/", 1)[0]
    if model_text.startswith("gemini") or "generativelanguage.googleapis.com" in base_text:
        return "gemini"
    if model_text.startswith("deepseek") or "deepseek.com" in base_text:
        return "deepseek"
    if model_text.startswith("claude") or "anthropic.com" in base_text:
        return "anthropic"
    if "api.openai.com" in base_text:
        return "openai"
    if "openrouter.ai" in base_text:
        return "openrouter"
    if "api.groq.com" in base_text:
        return "groq"
    if "novita" in base_text:
        return "novita"
    return ""


def _provider_key_env(model: str | None = None, api_base: str | None = None) -> str | None:
    provider = _provider_name(model, api_base)
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "novita": "NOVITA_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)


def default_api_key(
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
) -> str | None:
    explicit_key = _clean(api_key)
    if explicit_key:
        return explicit_key

    fugu_key = _clean(os.environ.get("FUGU_API_KEY"))
    if fugu_key:
        return fugu_key

    provider_env = _provider_key_env(model, api_base)
    if provider_env:
        key = _clean(os.environ.get(provider_env))
        if key:
            return key
    return _clean(os.environ.get("OPENAI_API_KEY"))


def default_api_base(api_base: str | None = None) -> str | None:
    return (
        _clean(api_base)
        or _clean(os.environ.get("FUGU_BASE_URL"))
        or _clean(os.environ.get("OPENAI_BASE_URL"))
    )


def litellm_credentials(
    spec: SlotSpec | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, str]:
    """Return the credential kwargs used by every LiteLLM caller."""
    model = spec.model if spec else None
    base = (spec.api_base if spec else None) or default_api_base(api_base)
    key = (spec.api_key if spec else None) or default_api_key(
        api_key,
        model=model,
        api_base=base,
    )
    out: dict[str, str] = {}
    if key:
        out["api_key"] = key
    if base:
        out["api_base"] = base
    return out


def uses_openai_compatible_http(model: str, api_base: str | None = None) -> bool:
    return bool(_clean(api_base)) and "/" not in (_clean(model) or "")


def _chat_completions_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.lower().endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def openai_compatible_completion_content(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    api_base: str,
    max_tokens: int,
    temperature: float,
    timeout: float | None = None,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _chat_completions_url(api_base),
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        request_timeout = _DEFAULT_HTTP_TIMEOUT if timeout is None else timeout
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    try:
        return payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat completion response: {payload}") from exc


def completion_content(
    spec: SlotSpec,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: float | None = None,
    label: str | None = None,
    max_retries: int | None = None,
) -> str:
    credentials = litellm_credentials(spec, api_key=api_key, api_base=api_base)
    base = credentials.get("api_base")
    tag = label or f"slot {spec.model}"

    def _call() -> str:
        if uses_openai_compatible_http(spec.model, base):
            return openai_compatible_completion_content(
                model=spec.model,
                messages=messages,
                api_key=credentials.get("api_key"),
                api_base=base,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )

        import litellm

        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        kwargs.update(credentials)
        return litellm.completion(**kwargs).choices[0].message.content or ""

    return _retry_forever(_call, label=tag, max_retries=max_retries)


def describe_api_format(model: str, api_base: str | None = None) -> str:
    provider = _provider_name(model, api_base) or "openai-compatible"
    if provider == "anthropic":
        return (
            "Anthropic model via LiteLLM. The code still passes OpenAI-style "
            "chat messages to LiteLLM, which translates to Anthropic's native "
            "messages API. api_base should be a provider base URL, not the full "
            "/messages endpoint."
        )
    if provider in {"openai", "azure", "openrouter", "deepseek", "novita", "groq"}:
        return OPENAI_COMPAT_HINT
    if api_base:
        return OPENAI_COMPAT_HINT
    return (
        "Provider is selected by the LiteLLM model prefix. If you set API URL, "
        "it is treated as api_base rather than a full request endpoint."
    )


def _safe_url(value: str | None) -> str:
    if not value:
        return "(provider default)"
    return value.split("?", 1)[0]


def _looks_like_full_endpoint(api_base: str | None) -> bool:
    if not api_base:
        return False
    value = api_base.rstrip("/").lower()
    return value.endswith("/chat/completions") or value.endswith("/messages")


def _looks_like_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "401",
            "unauthorized",
            "authentication",
            "api key",
            "invalid key",
            "invalid_request_error",
        )
    )


def _credential_hint(spec: SlotSpec, api_base: str | None = None) -> str:
    env_name = _provider_key_env(spec.model, api_base)
    if env_name:
        return f"Check the slot api_key, {env_name}, or FUGU_API_KEY."
    return "Check the slot api_key, provider API key env var, or FUGU_API_KEY."


def connectivity_check_enabled() -> bool:
    value = os.environ.get("FUGU_SKIP_CONNECTIVITY_CHECK", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _connectivity_parallelism(n_checks: int) -> int:
    raw = os.environ.get("FUGU_CONNECTIVITY_PARALLELISM", "").strip()
    if not raw:
        return max(1, n_checks)
    try:
        requested = int(raw)
    except ValueError:
        return max(1, n_checks)
    if requested <= 0:
        return max(1, n_checks)
    return max(1, min(requested, n_checks))


def check_litellm_connectivity(
    specs: Iterable[SlotSpec],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 20,
    label: str = "litellm",
) -> None:
    """Fail fast if configured remote model slots cannot answer a tiny request."""
    specs = list(specs)
    if not specs:
        return
    if not connectivity_check_enabled():
        print("[preflight] LiteLLM connectivity check skipped by FUGU_SKIP_CONNECTIVITY_CHECK", flush=True)
        return

    print(f"[preflight] checking {label} connectivity ({len(specs)} slot(s)) ...", flush=True)
    seen: set[tuple[str, str, bool, str]] = set()
    checks: list[tuple[int, SlotSpec, dict[str, str], str | None, str]] = []
    for index, spec in enumerate(specs):
        credentials = litellm_credentials(spec, api_key=api_key, api_base=api_base)
        base = credentials.get("api_base")
        backend = "http" if uses_openai_compatible_http(spec.model, base) else "litellm"
        dedupe_key = (spec.model, base or "", bool(credentials.get("api_key")), backend)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if _looks_like_full_endpoint(base):
            raise RuntimeError(
                f"slot {index} ({spec.model}) API URL looks like a full endpoint: "
                f"{_safe_url(base)}. {describe_api_format(spec.model, base)}"
            )
        checks.append((index, spec, credentials, base, backend))

    def _check_one(item: tuple[int, SlotSpec, dict[str, str], str | None, str]):
        index, spec, credentials, base, backend = item

        try:
            completion_content(
                spec,
                [{"role": "user", "content": "Reply with OK."}],
                api_key=api_key,
                api_base=api_base,
                max_tokens=2,
                temperature=0,
                timeout=timeout,
                max_retries=0,
            )
        except Exception as exc:
            if _looks_like_auth_error(exc):
                raise RuntimeError(
                    f"LiteLLM connectivity check reached slot {index} "
                    f"({spec.model}, backend={backend}, "
                    f"api_base={_safe_url(base)}) but authentication failed. "
                    f"{_credential_hint(spec, base)} "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            raise RuntimeError(
                f"LiteLLM connectivity check failed for slot {index} "
                f"({spec.model}, backend={backend}, "
                f"api_base={_safe_url(base)}). "
                f"{describe_api_format(spec.model, base)} "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return index, spec, backend, base

    results = []
    parallelism = _connectivity_parallelism(len(checks))
    if parallelism == 1:
        for item in checks:
            results.append(_check_one(item))
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {executor.submit(_check_one, item): item[0] for item in checks}
            for future in as_completed(futures):
                results.append(future.result())

    for index, spec, backend, base in sorted(results, key=lambda item: item[0]):
        print(
            f"[preflight] ok slot {index}: {spec.label or spec.model} "
            f"model={spec.model} backend={backend} "
            f"api_base={_safe_url(base)}",
            flush=True,
        )


def _slot_from_raw(raw: Any, index: int) -> SlotSpec:
    if isinstance(raw, str):
        model = _clean(raw)
        if not model:
            raise ValueError(f"slot {index} model is empty")
        return SlotSpec(model=model)
    if not isinstance(raw, dict):
        raise ValueError(f"slot {index} must be an object or model string")

    model = _clean(raw.get("model") or raw.get("model_name"))
    if not model:
        raise ValueError(f"slot {index} model is empty")

    return SlotSpec(
        model=model,
        api_base=_clean(raw.get("api_base") or raw.get("base_url") or raw.get("url")),
        api_key=_clean(raw.get("api_key") or raw.get("key")),
        label=_clean(raw.get("label")),
    )


def _decode_config(raw: str) -> list[Any]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("slot config must be a JSON list")
    return data


def load_slot_specs(
    config_path: str | None = None,
    env_name: str | None = None,
    required_count: int | None = None,
    min_count: int | None = None,
    max_count: int | None = None,
) -> list[SlotSpec] | None:
    """Load per-slot LiteLLM config from a JSON file or environment variable."""
    raw: str | None = None
    source = ""
    if config_path:
        raw = Path(config_path).read_text(encoding="utf-8")
        source = config_path
    elif env_name:
        raw = os.environ.get(env_name)
        source = f"${env_name}"
    elif os.environ.get("FUGU_SLOT_CONFIG"):
        raw = os.environ["FUGU_SLOT_CONFIG"]
        source = "$FUGU_SLOT_CONFIG"

    if raw is None or raw.strip() == "":
        return None

    specs = [_slot_from_raw(item, i) for i, item in enumerate(_decode_config(raw))]
    if required_count is not None and len(specs) != required_count:
        raise ValueError(
            f"slot config from {source or 'input'} must contain exactly "
            f"{required_count} slots, got {len(specs)}"
        )
    if min_count is not None and len(specs) < min_count:
        raise ValueError(
            f"slot config from {source or 'input'} must contain at least "
            f"{min_count} slots, got {len(specs)}"
        )
    if max_count is not None and len(specs) > max_count:
        raise ValueError(
            f"slot config from {source or 'input'} must contain at most "
            f"{max_count} slots, got {len(specs)}"
        )
    return specs


def slot_labels(specs: list[SlotSpec]) -> list[str]:
    return [spec.label or spec.model for spec in specs]
