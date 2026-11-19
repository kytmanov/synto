from synto.pricing import estimate_cost_usd, lookup


def test_claude_list_prices():
    assert estimate_cost_usd("claude-opus-4-7", 1_000_000, 1_000_000) == 30.0
    assert estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0
    assert estimate_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == 6.0


def test_local_and_unknown_models():
    assert estimate_cost_usd("gemma4:e4b", 1_000_000, 1_000_000) == 0.0
    assert lookup("my-local-model") is None
    assert estimate_cost_usd("my-local-model", 1, 1) is None
