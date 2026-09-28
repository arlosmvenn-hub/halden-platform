"""What a model call costs, from the provider's token counts."""

from dataclasses import dataclass

from halden.ports.llm import Usage


# [start:prices]
@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_write: float  # 5-minute cache
    cache_read: float


# Published list prices, checked September 2026. Prices change:
# keep this table in configuration you can update without a
# deploy, and reconcile it monthly against the provider's invoice.
PRICES: dict[str, Price] = {
    "claude-haiku-4-5-20251001": Price(1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-5": Price(2.00, 10.00, 2.50, 0.20),
    "claude-opus-5-5": Price(4.00, 20.00, 5.00, 0.20),
}
# [end:prices]


# [start:cost]
def cost_usd(model: str, usage: Usage) -> float | None:
    """Cost of one call, or None for an unknown model. Unknown is
    not zero: a missing price should show up as a gap on the
    dashboard, not as a free model."""
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
    ) / 1_000_000


# [end:cost]
