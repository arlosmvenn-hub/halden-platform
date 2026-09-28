"""The evaluation runner against a real database, scripted models."""

import json
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from halden.adapters.fake_llm import ScriptedLLM
from halden.adapters.local_embedder import HashingEmbedder
from halden.config.settings import Settings
from halden.eval.dataset import EvalCase
from halden.eval.report import summarize
from halden.eval.runner import run_case
from halden.ingestion.service import IngestionService
from halden.ports.llm import Message, TextBlock
from halden.services.ask import AskService
from halden.store.documents import DocumentStore

DATA = Path(__file__).parents[2] / "data"


async def test_run_case_grades_retrieval_answer_and_judge(
    pool: AsyncConnectionPool,
) -> None:
    emb = HashingEmbedder(dimensions=384)
    ingest = IngestionService(
        DocumentStore(pool), emb, lambda t: len(t.split()), 60
    )
    await ingest.ingest(
        doc_id="m:px",
        system="m",
        filename="px.md",
        data=(DATA / "manuals/px200_manual.md").read_bytes(),
        acl=["everyone"],
        source_version="1",
    )
    answer = ScriptedLLM(["HART needs at least 250 ohms [1]."])
    judge = ScriptedLLM(
        [
            json.dumps(
                {
                    "claims": [
                        {
                            "claim": "HART needs 250 ohms",
                            "reason": "Stated in [1].",
                            "supported": True,
                        }
                    ]
                }
            ),
            json.dumps({"reasoning": "Matches.", "verdict": "correct"}),
        ]
    )
    svc = AskService(pool, emb, answer, Settings(), None)
    case = EvalCase(
        id="c1",
        question="What loop resistance does HART need? 250 ohms",
        evidence=["loop resistance of at least 250 ohms"],
        reference="At least 250 ohms.",
    )
    r = await run_case(svc, case, generate=True, judge=judge)
    assert r.hit == 1.0 and r.context_recall == 1.0
    assert r.abstention_ok and r.citations_valid
    assert r.faithfulness == 1.0 and r.correctness == "correct"
    # The judge saw the question, the sources, and the answer.
    first: Message = judge.calls[0]["messages"][0]  # type: ignore[index]
    block = first.content[0]
    assert isinstance(block, TextBlock)
    prompt = block.text
    assert "<question>" in prompt and "[1]" in prompt
    assert summarize([r])["correct"] == 1.0


async def test_unanswerable_case_expects_abstention(
    pool: AsyncConnectionPool,
) -> None:
    svc = AskService(
        pool,
        HashingEmbedder(dimensions=384),
        ScriptedLLM([]),
        Settings(),
    )
    case = EvalCase(
        id="u", question="Who is the CEO?", answerable=False
    )
    r = await run_case(svc, case, generate=True)
    # Empty index: no permitted sources, abstain without a model call.
    assert r.abstained and r.abstention_ok
    assert summarize([r])["missed_abstain"] == 0.0
