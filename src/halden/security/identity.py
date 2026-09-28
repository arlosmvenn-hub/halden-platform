"""Who is asking. Every request carries one of these."""

from pydantic import BaseModel


class Identity(BaseModel):
    user_id: str
    groups: list[str]  # drives every retrieval permission filter
    tenant: str = "halden"  # the customer organization (Ch 26)
