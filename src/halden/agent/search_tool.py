"""Document search exposed as a tool, with session-wide citations."""

from collections.abc import Sequence
from typing import Any

from halden.agent.loop import Tool
from halden.generation.context import Context, Source
from halden.ports.llm import ToolSpec
from halden.retrieval.pg_retriever import PgRetriever


# [start:tool]
class SearchSession:
    """Runs searches for one agent session and numbers every
    source it returns, so citations stay unique across searches
    and can be checked against everything the model was shown."""

    def __init__(
        self,
        retriever: PgRetriever,
        groups: Sequence[str],
        per_search: int = 4,
    ) -> None:
        self._retriever = retriever
        self._groups = list(groups)  # from identity, never the model
        self._per_search = per_search
        self.sources: list[Source] = []
        self._label_of: dict[str, int] = {}

    async def search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()[:300]
        if not query:
            raise ValueError("query must be a non-empty string")
        hits = await self._retriever.hybrid(
            query, self._groups, k=self._per_search
        )
        lines = []
        for h in hits:
            if h.chunk_id not in self._label_of:
                label = len(self.sources) + 1
                self._label_of[h.chunk_id] = label
                self.sources.append(
                    Source(
                        label,
                        h.chunk_id,
                        h.title,
                        h.heading_path,
                        h.text,
                    )
                )
            where = " > ".join([h.title, *h.heading_path])
            lines.append(
                f"[{self._label_of[h.chunk_id]}] {where}\n{h.text}"
            )
        return "\n\n".join(lines) or "No results."

    def context(self) -> Context:
        return Context(sources=self.sources, tokens=0)

    def tool(self) -> Tool:
        return Tool(
            spec=ToolSpec(
                name="search_documents",
                description=(
                    "Search Halden's manuals, policies, wiki, and "
                    "support tickets. Returns numbered sources. "
                    "Use one focused query per sub-question; call "
                    "again with different words if results miss."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            handler=self.search,
        )


# [end:tool]
