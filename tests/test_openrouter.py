from lean_sft.openrouter import resolve_model


def test_resolve_model_uses_requested(monkeypatch):
    monkeypatch.setattr("lean_sft.openrouter.list_models", lambda *_: ["deepseek/deepseek-v4-flash"])
    cfg = {"openrouter": {"requested_model": "deepseek/deepseek-v4-flash", "base_url": "https://example.test"}}
    assert resolve_model("key", cfg) == "deepseek/deepseek-v4-flash"


def test_resolve_model_rejects_fallback(monkeypatch):
    monkeypatch.setattr("lean_sft.openrouter.list_models", lambda *_: ["deepseek/deepseek-v3.2"])
    cfg = {"openrouter": {"requested_model": "deepseek/deepseek-v4-flash", "base_url": "https://example.test"}}
    try:
        resolve_model("key", cfg)
    except RuntimeError as exc:
        assert "no fallback" in str(exc)
    else:
        raise AssertionError("expected fallback rejection")
