from __future__ import annotations

import json
import time
from typing import Any

import requests


def list_models(api_key: str, base_url: str) -> list[str]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    return [item["id"] for item in payload.get("data", []) if "id" in item]


def resolve_model(api_key: str, cfg: dict[str, Any]) -> str:
    requested = cfg["openrouter"]["requested_model"]
    base_url = cfg["openrouter"]["base_url"]
    models = list_models(api_key, base_url)
    if requested in models:
        return requested
    matches = [model for model in models if "deepseek" in model.lower()]
    hint = ", ".join(matches[:8]) or "no deepseek models found"
    raise RuntimeError(
        f"requested OpenRouter model {requested!r} is not live. "
        f"no fallback is allowed for this run. nearby models: {hint}"
    )


def chat_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: int,
    max_retries: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Mantissagithub/transformer-lab",
        "X-Title": "lean-starcoder-sft",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"transient OpenRouter status {resp.status_code}")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {last_error}")
