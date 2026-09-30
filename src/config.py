"""Environment-backed application settings."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    pinecone_api_key: str
    chat_provider: str = "local"
    local_chat_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    local_max_new_tokens: int = 64
    pinecone_index: str = "agentic-ai-ebook-local"
    pinecone_namespace: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chat_model: str = "gpt-4o-mini"
    pdf_path: str = "data/Ebook-Agentic-AI.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 6
    retrieval_min_score: float = 0.3
    max_generation_attempts: int = 2
    pinecone_ready_timeout_seconds: int = 120
    pinecone_upsert_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not pinecone_api_key:
            raise ValueError("Missing required environment variable: PINECONE_API_KEY")

        settings = cls(
            openai_api_key=openai_api_key or None,
            pinecone_api_key=pinecone_api_key,
            chat_provider=os.getenv("CHAT_PROVIDER", "local").strip().lower(),
            local_chat_model=os.getenv(
                "LOCAL_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
            ),
            local_max_new_tokens=_positive_int("LOCAL_MAX_NEW_TOKENS", 64),
            pinecone_index=os.getenv("PINECONE_INDEX", "agentic-ai-ebook-local"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dimension=_positive_int("EMBEDDING_DIMENSION", 384),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            pdf_path=os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf"),
            chunk_size=_positive_int("CHUNK_SIZE", 1000),
            chunk_overlap=_nonnegative_int("CHUNK_OVERLAP", 150),
            retrieval_k=_positive_int("RETRIEVAL_K", 5),
            retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3")),
            max_generation_attempts=_positive_int("MAX_GENERATION_ATTEMPTS", 2),
            pinecone_ready_timeout_seconds=_positive_int(
                "PINECONE_READY_TIMEOUT_SECONDS", 120
            ),
            pinecone_upsert_timeout_seconds=_positive_int(
                "PINECONE_UPSERT_TIMEOUT_SECONDS", 60
            ),
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if settings.chat_provider not in {"local", "openai"}:
            raise ValueError("CHAT_PROVIDER must be either 'local' or 'openai'")
        if settings.chat_provider == "openai" and not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when CHAT_PROVIDER=openai")
        if not 0.0 <= settings.retrieval_min_score <= 1.0:
            raise ValueError("RETRIEVAL_MIN_SCORE must be between 0 and 1")
        return settings

    def require_openai_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for chat generation and groundedness grading"
            )
        return self.openai_api_key


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
