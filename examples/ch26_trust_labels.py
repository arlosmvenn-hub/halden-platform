"""Chapter 26: how the context builder labels an external source.

Renders two sources the way the model receives them: a curated
manual section and an inert injection fixture from a support
ticket. Texts are cut after their first sentence for print.
"""

from pathlib import Path

from halden.generation.context import Context, Source

ROOT = Path(__file__).resolve().parent.parent


def first_sentence(path: Path, marker: str) -> str:
    for line in path.read_text("utf-8").splitlines():
        if line.startswith(marker):
            return line.split(". ")[0] + ". ..."
    raise SystemExit(f"{marker!r} not found in {path}")


def title(path: Path) -> str:
    return path.read_text("utf-8").splitlines()[0].lstrip("# ")


def main() -> None:
    manual = ROOT / "data" / "manuals" / "px200_manual.md"
    ticket = ROOT / "eval" / "injection" / "ticket-9003.md"
    ctx = Context(
        sources=[
            Source(
                label=1,
                chunk_id="manuals:px200_manual#3.2",
                title=title(manual),
                heading_path=[
                    "3 Installation",
                    "3.2 Electrical connection",
                ],
                text=first_sentence(manual, "Connect the loop"),
            ),
            Source(
                label=2,
                chunk_id="tickets:ticket-9003#0",
                title="Support Ticket 9003",
                heading_path=["(open)"],
                text=first_sentence(ticket, "Note from"),
                untrusted=True,
            ),
        ],
        tokens=0,
    )
    print(ctx.render())


if __name__ == "__main__":
    main()
