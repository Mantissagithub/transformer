from __future__ import annotations

from typing import Any

from .schema import REQUIRED_FIELDS, LeanExample, normalize_code


def build_example(raw: dict[str, Any], *, expected_phase: str, model: str) -> LeanExample:
    missing = REQUIRED_FIELDS - set(raw)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    phase = str(raw["phase"]).strip()
    if phase != expected_phase:
        raise ValueError(f"expected phase {expected_phase}, got {phase}")
    instruction = str(raw["instruction"]).strip()
    response = str(raw["response"]).strip()
    lean_code = normalize_code(str(raw["lean_code"]))
    flags = [str(flag) for flag in raw.get("quality_flags", [])]
    rejection = rejection_reason(instruction, response, lean_code)
    if rejection:
        raise ValueError(rejection)
    return LeanExample(
        phase=phase,  # type: ignore[arg-type]
        instruction=instruction,
        response=response,
        lean_code=lean_code,
        source=str(raw.get("source") or "synthetic_openrouter"),
        generator_model=str(raw.get("generator_model") or model),
        quality_flags=flags,
    )


def rejection_reason(instruction: str, response: str, lean_code: str) -> str | None:
    if not instruction:
        return "empty instruction"
    if not response:
        return "empty response"
    if not lean_code:
        return "empty lean_code"
    if "```" in response or "```" in lean_code:
        return "markdown fence"
    if len(response) > 6000:
        return "response too long"
    lean_markers = ("theorem ", "lemma ", "def ", "example ", "inductive ", "structure ", "namespace ")
    if not any(marker in lean_code for marker in lean_markers):
        return "no lean declaration marker"
    if lean_code.count("sorry") > 0:
        return "contains sorry"
    return None
