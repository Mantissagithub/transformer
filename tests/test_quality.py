from lean_sft.quality import build_example


def test_build_example_accepts_syntax_row():
    row = {
        "phase": "syntax",
        "instruction": "write a definition",
        "response": "def x : Nat := 1",
        "lean_code": "def x : Nat := 1",
        "source": "test",
        "generator_model": "stub",
        "quality_flags": [],
    }
    example = build_example(row, expected_phase="syntax", model="stub")
    assert example.lean_code == "def x : Nat := 1"


def test_build_example_rejects_sorry():
    row = {
        "phase": "proof",
        "instruction": "prove it",
        "response": "theorem x : True := by\n  sorry",
        "lean_code": "theorem x : True := by\n  sorry",
        "source": "test",
        "generator_model": "stub",
        "quality_flags": [],
    }
    try:
        build_example(row, expected_phase="proof", model="stub")
    except ValueError as exc:
        assert "sorry" in str(exc)
    else:
        raise AssertionError("expected rejection")
