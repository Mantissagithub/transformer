from __future__ import annotations


JSON_RULES = """Return one strict JSON object with:
phase, instruction, response, lean_code, source, quality_flags.
No markdown fences. No extra prose."""


def syntax_prompt(seed: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You create small correct Lean 4 syntax training examples for a code model.",
        },
        {
            "role": "user",
            "content": f"""{JSON_RULES}

Create one Lean 4 syntax example. Cover imports, definitions, inductives, structures, namespaces, theorem statements, tactics, or term syntax.
The response should explain the pattern briefly, then give compileable Lean code.
Use seed {seed} to vary the topic.
Set phase to syntax and source to synthetic_openrouter.""",
        },
    ]


def proof_prompt(seed: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You create short Lean 4 proof-completion training examples for a code model.",
        },
        {
            "role": "user",
            "content": f"""{JSON_RULES}

Create one simple Lean 4 proof example. Prefer elementary propositions, Nat/Int arithmetic where feasible, lists, options, equality rewriting, conjunctions, implications, or small tactic proofs.
The instruction should ask for the proof. The response should include only the final Lean proof code plus a short note if helpful.
Use seed {seed} to vary the theorem.
Set phase to proof and source to synthetic_openrouter.""",
        },
    ]
