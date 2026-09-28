"""Query transformations: condense, expand, and imagine."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from halden.generation.structured import complete_structured
from halden.ports.embedder import Embedder
from halden.ports.llm import LLMClient, Message
from halden.retrieval.fusion import reciprocal_rank_fusion
from halden.retrieval.pg_retriever import PgRetriever, RetrievedChunk


# [start:condense]
class Turn(BaseModel):
    role: str  # "user" or "assistant"
    text: str


CONDENSE_SYSTEM = """\
Rewrite the user's latest message as a standalone search query for
Halden Instruments' internal documents. Resolve pronouns and
references using the conversation. Keep product names, part
numbers, and error codes exactly as written. Output only the
query, with no quotes or explanation."""


async def condense_question(
    llm: LLMClient, history: Sequence[Turn], question: str
) -> str:
    """Turn a follow-up ("and how do I send it back?") into a
    query that retrieval can use on its own."""
    if not history:
        return question
    convo = "\n".join(f"{t.role}: {t.text}" for t in history[-6:])
    prompt = f"Conversation:\n{convo}\n\nLatest message: {question}"
    resp = await llm.complete(
        [Message.user(prompt)], system=CONDENSE_SYSTEM, max_tokens=100
    )
    return resp.text.strip() or question


# [end:condense]


# [start:variants]
class Variants(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


VARIANTS_SYSTEM = """\
You help search the internal documents of Halden Instruments, a
maker of pressure transmitters (PX), flow meters (FM), and
temperature sensors (HX). Rewrite the question as three different
search queries: one using precise technical terms, one using the
words a manual would use, and one short keyword query. Keep part
numbers and error codes exactly."""


async def generate_variants(llm: LLMClient, question: str) -> list[str]:
    v = await complete_structured(
        llm,
        Variants,
        [Message.user(question)],
        system=VARIANTS_SYSTEM,
    )
    return v.queries


async def multi_query(
    retriever: PgRetriever,
    queries: Sequence[str],
    groups: Sequence[str],
    k: int = 10,
) -> list[RetrievedChunk]:
    """Retrieve for each query variant, then fuse with RRF."""
    runs = [await retriever.hybrid(q, groups, k=20) for q in queries]
    by_id = {c.chunk_id: c for run in runs for c in run}
    fused = reciprocal_rank_fusion(
        [[c.chunk_id for c in run] for run in runs]
    )
    return [by_id[cid] for cid, _ in fused[:k]]


# [end:variants]


# [start:hyde]
HYDE_SYSTEM = """\
Write a short passage (2-4 sentences) that could appear in a Halden
Instruments manual, policy, or support ticket and would answer the
question. Write it in the style of that document. It is used only
to search for the real passage, so plausible detail is fine."""


async def hyde(
    llm: LLMClient,
    embedder: Embedder,
    retriever: PgRetriever,
    question: str,
    groups: Sequence[str],
    k: int = 10,
) -> list[RetrievedChunk]:
    """Hypothetical Document Embeddings: embed an imagined answer
    *as a document* and search with it; lexical search still uses
    the user's real words."""
    resp = await llm.complete(
        [Message.user(question)], system=HYDE_SYSTEM, max_tokens=200
    )
    vec = (await embedder.embed_documents([resp.text]))[0]
    return await retriever.hybrid(question, groups, k=k, qvec=vec)


# [end:hyde]
