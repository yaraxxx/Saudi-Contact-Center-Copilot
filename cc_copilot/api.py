import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Form
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import asyncio

from cc_copilot.agents.call_qa_agent import review_call
from cc_copilot.agents.chat_review_agent import review_chat
from cc_copilot.agents.document_agent import review_document
from cc_copilot.agents.policy_agent import check_policy, last_exchange
from cc_copilot.fusion import build_report
from cc_copilot.env import POLICY_DIR, get_default_model_choice, get_user_api_key
from cc_copilot.pii.saudi_patterns import mask_pii
from cc_copilot.rag.retriever import BM25Retriever, load_policy_chunks
from cc_copilot.types.model_choice import ModelChoice
from cc_copilot.types.results import CallQAResult, ChatReviewResult, DocumentReviewResult, PIIFinding, PolicyCheckResult, InteractionReview
from cc_copilot.utils import detect_file_type

# Standard auth scheme using -H "Authorization: Bearer <api_key>" header.
security = HTTPBearer()


# NOTE: a static key is enough for a demo. Production would use per-client keys or OAuth.
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != get_user_api_key():
        raise HTTPException(status_code=401, detail="Invalid user API key")
    return credentials.credentials


# Dependencies are functions so tests can override them with pydantic-ai's TestModel.
def get_model_choice() -> ModelChoice:
    try:
        return get_default_model_choice()
    except ValueError as e:
        # Missing GEMINI_API_KEY / DEFAULT_GOOGLE_MODEL: the offline endpoints (e.g. /pii/mask) still work.
        raise HTTPException(status_code=503, detail=f"LLM not configured: {e}")


@lru_cache
def get_retriever() -> BM25Retriever:
    return BM25Retriever(load_policy_chunks(POLICY_DIR))


class TextRequest(BaseModel):
    text: str


class ChatReviewRequest(BaseModel):
    transcript: str


class PolicyCheckRequest(BaseModel):
    customer_message: str
    agent_reply: str


class MaskResponse(BaseModel):
    masked_text: str
    findings: list[PIIFinding]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("ENABLE_TRACING", "1") == "1":
        from cc_copilot.tracing import setup_tracing

        setup_tracing()
    yield


app = FastAPI(title="Contact-Center QA Copilot", dependencies=[Depends(validate_api_key)], lifespan=lifespan)


async def _read_upload(file: UploadFile, expected_prefix: str) -> tuple[bytes, str]:
    data = await file.read()
    try:
        mime = detect_file_type(data, context=file.filename or "uploaded file")
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    if not mime.startswith(expected_prefix):
        raise HTTPException(status_code=415, detail=f"Expected {expected_prefix}* file, got {mime}")
    return data, mime


@app.post("/api/v1/pii/mask", response_model=MaskResponse)
async def mask_endpoint(request: TextRequest):
    masked, matches = mask_pii(request.text)
    return MaskResponse(masked_text=masked, findings=[PIIFinding.from_match(m) for m in matches])


@app.post("/api/v1/review/chat", response_model=ChatReviewResult)
async def review_chat_endpoint(request: ChatReviewRequest, model_choice: ModelChoice = Depends(get_model_choice)):
    return await review_chat(model_choice, request.transcript)


@app.post("/api/v1/review/call", response_model=CallQAResult)
async def review_call_endpoint(file: UploadFile = File(...), model_choice: ModelChoice = Depends(get_model_choice)):
    data, mime = await _read_upload(file, "audio/")
    return await review_call(model_choice, data, mime)


@app.post("/api/v1/review/document", response_model=DocumentReviewResult)
async def review_document_endpoint(file: UploadFile = File(...), model_choice: ModelChoice = Depends(get_model_choice)):
    data, mime = await _read_upload(file, "image/")
    return await review_document(model_choice, data, mime)


@app.post("/api/v1/policy/check", response_model=PolicyCheckResult)
async def policy_check_endpoint(
    request: PolicyCheckRequest,
    model_choice: ModelChoice = Depends(get_model_choice),
    retriever: BM25Retriever = Depends(get_retriever),
):
    return await check_policy(model_choice, retriever, request.customer_message, request.agent_reply)

@app.post("/api/v1/review/interaction", response_model=InteractionReview)
async def review_interaction_endpoint(
    transcript: str | None = Form(None),
    call: UploadFile | None = File(None),
    documents: list[UploadFile] | None = File(None),
    model_choice: ModelChoice = Depends(get_model_choice),
    retriever: BM25Retriever = Depends(get_retriever),
):
    """Review a chat transcript, a call and any images from the same interaction, then fuse the results."""
    documents = documents or []
    if not (transcript or call or documents):
        raise HTTPException(status_code=422, detail="Send at least one of: transcript, call, documents")
    # Read and check every upload first, so a bad file fails before any Gemini call is paid for.
    call_upload = await _read_upload(call, "audio/") if call else None
    doc_uploads = [await _read_upload(d, "image/") for d in documents]

    # Every reviewer is independent, so run them all at the same time.
    tasks = {}
    if transcript:
        tasks["chat"] = review_chat(model_choice, transcript)
        if exchange := last_exchange(transcript):
            tasks["policy"] = check_policy(model_choice, retriever, *exchange)
    if call_upload:
        tasks["call"] = review_call(model_choice, *call_upload)
    for i, upload in enumerate(doc_uploads):
        tasks[f"document_{i}"] = review_document(model_choice, *upload)

    results = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    docs = [results[f"document_{i}"] for i in range(len(doc_uploads))]

    report = build_report(chat=results.get("chat"), call=results.get("call"), documents=docs, policy=results.get("policy"))
    return InteractionReview(
        report=report, chat=results.get("chat"), policy=results.get("policy"), call=results.get("call"), documents=docs
    )


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("cc_copilot.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
