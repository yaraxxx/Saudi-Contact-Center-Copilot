from pydantic_ai import Agent

from cc_copilot.pii.saudi_patterns import mask_pii
from cc_copilot.rag.retriever import BM25Retriever
from cc_copilot.types.model_choice import ModelChoice
from cc_copilot.types.results import PolicyCheckResult

POLICY_INSTRUCTIONS = """
<role>
You are a compliance reviewer at Rimal Telecom, a Saudi telecom operator.
</role>

<input>
A customer message, the agent's reply, and excerpts from company policy. Each excerpt starts with its ID in brackets.
</input>

<instructions>
- Decide whether the agent's reply complies with the policy excerpts.
- Use only the excerpts provided. If none are relevant, the reply is compliant and cited_policy_ids is empty.
- Only use policy IDs that appear in the excerpts.
- Write suggested_reply in the same language and dialect as the agent's reply.
</instructions>
"""

policy_agent = Agent(
    output_type=PolicyCheckResult,
    instructions=POLICY_INSTRUCTIONS,
)


def build_prompt(customer_message: str, agent_reply: str, retriever: BM25Retriever, top_k: int) -> tuple[str, set[str]]:
    hits = retriever.search(f"{customer_message} {agent_reply}", top_k=top_k)
    excerpts = "\n\n".join(f"[{h.chunk.policy_id}] {h.chunk.title}\n{h.chunk.text}" for h in hits)
    prompt = (
        f"<customer_message>\n{customer_message}\n</customer_message>\n\n"
        f"<agent_reply>\n{agent_reply}\n</agent_reply>\n\n"
        f"<policy_excerpts>\n{excerpts or '(no relevant policy found)'}\n</policy_excerpts>"
    )
    return prompt, {h.chunk.policy_id for h in hits}


async def check_policy(
    model_choice: ModelChoice,
    retriever: BM25Retriever,
    customer_message: str,
    agent_reply: str,
    top_k: int = 3,
) -> PolicyCheckResult:
    masked_customer, _ = mask_pii(customer_message)
    masked_reply, _ = mask_pii(agent_reply)
    prompt, retrieved_ids = build_prompt(masked_customer, masked_reply, retriever, top_k)

    result = await policy_agent.run(prompt, model=model_choice.model, model_settings=model_choice.model_settings)
    output = result.output

    # Guard against hallucinated citations: keep only IDs we actually retrieved.
    output.cited_policy_ids = [p for p in output.cited_policy_ids if p in retrieved_ids]
    output.violated_policy_ids = [p for p in output.violated_policy_ids if p in retrieved_ids]
    return output
