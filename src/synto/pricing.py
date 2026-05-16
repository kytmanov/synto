"""Conservative pricing lookup for offline stats cost estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float | None
    output_per_mtok: float | None
    is_cloud: bool


_DEFAULT_PRICES: dict[str, Price] = {
    "gemma4:e4b": Price(0.0, 0.0, is_cloud=False),
    "google/gemma-4-e4b": Price(0.0, 0.0, is_cloud=False),
    "qwen2.5:14b": Price(0.0, 0.0, is_cloud=False),
    "claude-opus-4-7": Price(15.0, 75.0, is_cloud=True),
    "claude-sonnet-4-6": Price(3.0, 15.0, is_cloud=True),
    "claude-haiku-4-5-20251001": Price(0.8, 4.0, is_cloud=True),
}

_CLOUD_PROVIDERS = {"anthropic", "azure-openai", "groq", "openai", "together"}


def lookup(model_name: str) -> Price | None:
    return _DEFAULT_PRICES.get(model_name)


def is_cloud_provider(provider_name: str) -> bool:
    return provider_name.lower() in _CLOUD_PROVIDERS


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = lookup(model)
    if price is None or price.input_per_mtok is None or price.output_per_mtok is None:
        return None
    return (input_tokens * price.input_per_mtok / 1_000_000) + (
        output_tokens * price.output_per_mtok / 1_000_000
    )
