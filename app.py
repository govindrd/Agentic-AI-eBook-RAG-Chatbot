"""FastAPI HTTP interface."""

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_openai.chat_models.base import OpenAIRateLimitError
from pydantic import BaseModel, Field

from src.config import Settings
from src.graph import ChatResponse, create_production_graph
from src.ingestion import (
    KnowledgeBaseNotReadyError,
    inspect_index,
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


def create_app(
    answer_graph: Callable[[str], ChatResponse] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Agentic AI eBook RAG Chatbot",
        version="1.0.0",
        description="Answers questions using only the supplied Agentic AI eBook.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str | int]:
        settings = Settings.from_env()
        status = inspect_index(settings)
        if not status.ready:
            raise KnowledgeBaseNotReadyError(status.detail)
        return {
            "status": "ready",
            "index": status.index_name,
            "namespace": status.namespace,
            "vector_count": status.namespace_vector_count,
        }

    @app.exception_handler(KnowledgeBaseNotReadyError)
    async def knowledge_base_not_ready(_, error: KnowledgeBaseNotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(OpenAIRateLimitError)
    async def openai_rate_limit(_, error: OpenAIRateLimitError):
        error_body = error.body if isinstance(error.body, dict) else {}
        provider_error = error_body.get("error", {})
        provider_code = (
            provider_error.get("code") if isinstance(provider_error, dict) else None
        )
        if provider_code in {"insufficient_quota", "credit_balance_exhausted"}:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "OpenAI API credits are exhausted. Add API billing credits or "
                        "set CHAT_PROVIDER=local in .env and restart Uvicorn."
                    )
                },
            )
        return JSONResponse(
            status_code=429,
            content={"detail": "The configured chat provider rate limit was exceeded."},
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        nonlocal answer_graph
        if answer_graph is None:
            answer_graph = create_production_graph(Settings.from_env())
        return answer_graph(request.query)

    return app


app = create_app()
