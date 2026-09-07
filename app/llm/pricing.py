from __future__ import annotations

from app.core.settings import ModelPricingSettings
from app.llm.schemas import LLMUsage


def estimate_cost_usd(
    model: str,
    usage: LLMUsage,
    pricing: dict[str, ModelPricingSettings],
) -> float:
    """Estimate a call's cost in USD from per-1M-token prices.

    Unpriced models are treated as cost 0.0 (best-effort estimate; pricing
    is an opt-in [llm].pricing configuration, not a hard dependency).
    """
    price = pricing.get(model)
    if price is None:
        return 0.0
    input_cost = usage.prompt_tokens / 1_000_000 * price.input_per_million
    output_cost = usage.completion_tokens / 1_000_000 * price.output_per_million
    return round(input_cost + output_cost, 6)
