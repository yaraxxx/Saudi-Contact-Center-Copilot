"""
QA manager and trainee UI. Calls the FastAPI backend over HTTP (like the original project),
except the training simulator, which runs the customer agent in-process.
"""

import asyncio

import gradio as gr
import requests

from cc_copilot.agents.customer_sim_agent import DIALECT_HINTS, CustomerPersona, customer_reply
from cc_copilot.env import API_BASE_URL, get_default_model_choice, get_user_api_key
from cc_copilot.fusion import build_report
from cc_copilot.types.results import ChatReviewResult


def _post(path: str, **kwargs) -> dict:
    response = requests.post(
        f"{API_BASE_URL}{path}", headers={"Authorization": f"Bearer {get_user_api_key()}"}, timeout=120, **kwargs
    )
    response.raise_for_status()
    return response.json()


def mask_text(text: str) -> tuple[str, list]:
    result = _post("/api/v1/pii/mask", json={"text": text})
    return result["masked_text"], result["findings"]


def review_transcript(transcript: str) -> tuple[dict, dict]:
    chat = ChatReviewResult.model_validate(_post("/api/v1/review/chat", json={"transcript": transcript}))
    return chat.model_dump(mode="json"), build_report(chat=chat).model_dump(mode="json")


def review_call_file(path: str) -> dict:
    with open(path, "rb") as f:
        return _post("/api/v1/review/call", files={"file": f})


def check_reply(customer_message: str, agent_reply: str) -> dict:
    return _post("/api/v1/policy/check", json={"customer_message": customer_message, "agent_reply": agent_reply})


def train_step(message: str, chat_history: list, dialect: str, state: dict):
    # Review the trainee's message before the simulated customer sees it.
    transcript = "\n".join(f"{'Agent' if m['role'] == 'user' else 'Customer'}: {m['content']}" for m in chat_history)
    review = ChatReviewResult.model_validate(
        _post("/api/v1/review/chat", json={"transcript": f"{transcript}\nAgent: {message}"})
    )
    report = build_report(chat=review)
    if report.risk_level == "high":
        feedback = "Blocked. " + "; ".join(report.blocking_issues)
        return chat_history, feedback, state

    reply, state["history"] = asyncio.run(
        customer_reply(get_default_model_choice(), CustomerPersona(dialect=dialect), message, state.get("history"))
    )
    chat_history = chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    feedback = "; ".join(report.warnings) or "Looks good."
    return chat_history, feedback, state


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Contact-Center QA Copilot") as demo:
        gr.Markdown("# Contact-Center QA Copilot\nArabic-first QA and PDPL compliance for Rimal Telecom (fictional).")

        with gr.Tab("PII masker"):
            text_in = gr.Textbox(label="Text", lines=4, rtl=True)
            masked_out = gr.Textbox(label="Masked", lines=4, rtl=True)
            findings_out = gr.JSON(label="Findings")
            gr.Button("Mask").click(mask_text, text_in, [masked_out, findings_out])

        with gr.Tab("Chat review"):
            transcript_in = gr.Textbox(label="Transcript (Customer: / Agent: lines)", lines=10, rtl=True)
            review_out, report_out = gr.JSON(label="Review"), gr.JSON(label="Report")
            gr.Button("Review").click(review_transcript, transcript_in, [review_out, report_out])

        with gr.Tab("Call QA"):
            audio_in = gr.Audio(type="filepath", label="Call recording")
            call_out = gr.JSON(label="QA result")
            gr.Button("Score call").click(review_call_file, audio_in, call_out)

        with gr.Tab("Policy check"):
            cust_in = gr.Textbox(label="Customer message", rtl=True)
            reply_in = gr.Textbox(label="Agent reply", rtl=True)
            policy_out = gr.JSON(label="Policy check")
            gr.Button("Check").click(check_reply, [cust_in, reply_in], policy_out)

        with gr.Tab("Training simulator"):
            dialect_in = gr.Dropdown(list(DIALECT_HINTS), value="najdi", label="Customer dialect")
            chatbot = gr.Chatbot(rtl=True)
            msg_in = gr.Textbox(label="Your reply as the agent", rtl=True)
            feedback_out = gr.Textbox(label="Coach feedback")
            state = gr.State({})
            msg_in.submit(train_step, [msg_in, chatbot, dialect_in, state], [chatbot, feedback_out, state])

    return demo


def main():
    build_ui().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
