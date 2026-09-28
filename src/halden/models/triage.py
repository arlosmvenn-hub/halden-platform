"""Support-ticket triage: the Chapter 3 structured-output example."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SAFETY = "safety"  # always routed to a human


class TicketTriage(BaseModel):
    product_line: str = Field(
        pattern=r"^(PX|FM|HX)-\d{3,4}$",
        description="Halden part number, e.g. PX-200",
    )
    severity: Severity
    summary: str = Field(max_length=200)

    @property
    def needs_human(self) -> bool:
        # A rule in code, not a judgment left to the model.
        return self.severity in {Severity.HIGH, Severity.SAFETY}
