from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Phase = Literal["syntax", "proof"]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class LeanExample:
    phase: Phase
    instruction: str
    response: str
    lean_code: str
    source: str
    generator_model: str
    quality_flags: list[str]
    split: Split = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_FIELDS = {
    "phase",
    "instruction",
    "response",
    "lean_code",
    "source",
    "generator_model",
    "quality_flags",
}


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def format_for_sft(row: dict[str, Any]) -> str:
    instruction = str(row["instruction"]).strip()
    response = str(row["response"]).strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
