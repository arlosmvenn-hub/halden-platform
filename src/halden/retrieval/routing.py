"""Decide where a question should go, and which filters apply."""

from typing import Literal

from pydantic import BaseModel, Field

from halden.generation.structured import complete_structured
from halden.ports.llm import LLMClient, Message


# [start:models]
class SearchFilters(BaseModel):
    """Relevance filters the model may propose. Deliberately has no
    field for permissions, tenants, or groups: those come only from
    the authenticated user, never from the question text."""

    product_family: Literal["PX", "FM", "HX"] | None = None
    doc_type: (
        Literal["manual", "policy", "ticket", "wiki", "contract"] | None
    ) = None
    region: Literal["US", "DE", "CA"] | None = None


class RouteDecision(BaseModel):
    destination: Literal["documents", "live_data", "out_of_scope"]
    filters: SearchFilters = Field(default_factory=SearchFilters)
    reason: str = Field(max_length=200)


# [end:models]

# [start:router]
ROUTER_SYSTEM = """\
You route questions for Ask Halden, the internal assistant of
Halden Instruments (industrial sensors: PX pressure transmitters,
FM flow meters, HX temperature sensors).

destination:
- "documents": answered by reading manuals, policies, wiki pages,
  past support tickets, or contracts.
- "live_data": needs current values or counts from systems:
  stock levels, order status, how many tickets match something,
  who is on call now.
- "out_of_scope": unrelated to Halden work.

filters: set only when the question clearly implies them;
otherwise leave null. Keep "reason" to one short sentence."""


async def route_question(
    llm: LLMClient, question: str
) -> RouteDecision:
    return await complete_structured(
        llm,
        RouteDecision,
        [Message.user(question)],
        system=ROUTER_SYSTEM,
    )


# [end:router]
