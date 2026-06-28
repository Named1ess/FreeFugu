"""Shared LiteLLM slot configuration helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SlotSpec:
    model: str
    api_base: str | None = None
    api_key: str | None = None
    label: str | None = None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
