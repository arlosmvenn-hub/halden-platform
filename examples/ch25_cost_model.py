"""Chapter 25: a monthly cost model for Ask Halden.

Token counts are the averages measured by Chapter 23's evaluation
run; traffic is Halden's estimate. Change the assumptions and rerun.
"""

from halden.observability.cost import cost_usd
from halden.ports.llm import Usage

# [start:assumptions]
USERS = 400  # employees who use Ask Halden
QUESTIONS_PER_DAY = 6  # per user, on a working day
WORKDAYS = 21
ANSWER = Usage(input_tokens=1_180, output_tokens=68)
CONDENSE = Usage(input_tokens=350, output_tokens=25)
FOLLOW_UP_SHARE = 0.4  # questions asked inside a conversation
# [end:assumptions]

SONNET, HAIKU = "claude-sonnet-5", "claude-haiku-4-5-20251001"


# [start:model]
def monthly(
    answer_model: str, cache_hit_rate: float = 0.0
) -> tuple[float, float]:
    """(cost per question, cost per month) in USD. Cache hits skip
    the answer call; follow-ups add a small-model condense call."""
    per_answer = cost_usd(answer_model, ANSWER) or 0.0
    per_condense = cost_usd(HAIKU, CONDENSE) or 0.0
    per_question = (
        per_answer * (1 - cache_hit_rate)
        + per_condense * FOLLOW_UP_SHARE
    )
    questions = USERS * QUESTIONS_PER_DAY * WORKDAYS
    return per_question, per_question * questions


# [end:model]


def main() -> None:
    questions = USERS * QUESTIONS_PER_DAY * WORKDAYS
    print(f"{questions:,} questions per month\n")
    print(f"{'scenario':<34}{'per question':>14}{'per month':>12}")
    for label, model, hit in [
        ("Sonnet 5 answers", SONNET, 0.0),
        ("Sonnet 5, 20% answer-cache hits", SONNET, 0.2),
        ("Haiku 4.5 answers", HAIKU, 0.0),
    ]:
        each, month = monthly(model, hit)
        print(f"{label:<34}{each:>14.5f}{month:>12,.2f}")
    budget = 50_000
    per_user_day = budget / (ANSWER.input_tokens + ANSWER.output_tokens)
    print(
        f"\nA {budget:,}-token daily budget allows about "
        f"{per_user_day:.0f} questions per user per day."
    )


if __name__ == "__main__":
    main()
