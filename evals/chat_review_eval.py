"""
LLM evaluation of the chat review agent across Saudi dialects.

    uv run python evals/chat_review_eval.py

Needs GEMINI_API_KEY. Each case is checked two ways:
  1. FlagsMatch: the boolean flags must equal the labels in datasets/chat_cases.jsonl
  2. LLMJudge: a second model grades whether the rationale quotes the right phrases
The report is grouped by case name, which starts with the dialect, so you can see
which dialects the model handles worst. Compare models by changing DEFAULT_GOOGLE_MODEL.
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_evals import Case, Dataset  # noqa: E402
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge  # noqa: E402

from cc_copilot.agents.chat_review_agent import review_chat  # noqa: E402
from cc_copilot.env import get_default_model_choice, get_judge_model_choice  # noqa: E402
from cc_copilot.types.results import ChatReviewResult  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "chat_cases.jsonl"


@dataclass
class FlagsMatch(Evaluator[str, ChatReviewResult]):
    expected: dict[str, bool]

    def evaluate(self, ctx: EvaluatorContext[str, ChatReviewResult]) -> EvaluationReason:
        wrong = [k for k, v in self.expected.items() if getattr(ctx.output, k) != v]
        return EvaluationReason(value=not wrong, reason=f"wrong flags: {wrong}" if wrong else "all flags correct")


def build_dataset() -> Dataset[str, ChatReviewResult, dict]:
    judge = get_judge_model_choice().model
    cases = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            Case(
                name=row["id"],
                inputs=row["transcript"],
                metadata={"dialect": row["dialect"]},
                evaluators=(
                    FlagsMatch(expected=row["expected"]),
                    LLMJudge(
                        rubric="The rationale correctly explains each flag and quotes the agent's words as evidence.",
                        model=judge,
                    ),
                ),
            )
        )
    return Dataset(cases=cases)


async def run_case(transcript: str) -> ChatReviewResult:
    return await review_chat(get_default_model_choice(), transcript)


def main() -> None:
    report = asyncio.run(build_dataset().evaluate(run_case, max_concurrency=2))
    report.print(include_input=False, include_output=False, include_durations=True)


if __name__ == "__main__":
    main()
