"""PDF loading, page-aware splitting, and Pinecone setup."""

import hashlib
from pathlib import Path
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings

if TYPE_CHECKING:
    from langchain_pinecone import PineconeVectorStore
    from src.graph import ContextChunk


class KnowledgeBaseNotReadyError(RuntimeError):
    """The configured Pinecone index cannot currently serve this app."""


@dataclass(frozen=True)
class IndexReadiness:
    index_name: str
    namespace: str
    exists: bool
    ready: bool
    dimension: int | None
    namespace_vector_count: int
    total_vector_count: int
    detail: str


def _field(value, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def inspect_index(settings: Settings, client=None) -> IndexReadiness:
    """Inspect index existence, readiness, dimension, and configured namespace count."""
    if client is None:
        from pinecone import Pinecone

        client = Pinecone(api_key=settings.pinecone_api_key)

    index_names = client.list_indexes().names()
    if settings.pinecone_index not in index_names:
        return IndexReadiness(
            index_name=settings.pinecone_index,
            namespace=settings.pinecone_namespace,
            exists=False,
            ready=False,
            dimension=None,
            namespace_vector_count=0,
            total_vector_count=0,
            detail=(
                f"Pinecone index {settings.pinecone_index!r} does not exist. "
                "Run `python -m src.ingestion` to create it and index the PDF."
            ),
        )

    description = client.describe_index(settings.pinecone_index)
    dimension = _field(description, "dimension")
    status = _field(description, "status", {})
    index_ready = bool(_field(status, "ready", False))
    if not index_ready:
        return IndexReadiness(
            index_name=settings.pinecone_index,
            namespace=settings.pinecone_namespace,
            exists=True,
            ready=False,
            dimension=dimension,
            namespace_vector_count=0,
            total_vector_count=0,
            detail=f"Pinecone index {settings.pinecone_index!r} exists but is not ready yet.",
        )

    stats = client.Index(settings.pinecone_index).describe_index_stats()
    namespaces = _field(stats, "namespaces", {}) or {}
    namespace_stats = namespaces.get(settings.pinecone_namespace, {})
    namespace_count = int(_field(namespace_stats, "vector_count", 0) or 0)
    total_count = int(_field(stats, "total_vector_count", 0) or 0)

    if dimension != settings.embedding_dimension:
        detail = (
            f"Pinecone index {settings.pinecone_index!r} has dimension {dimension}; "
            f"the configured embedding dimension is {settings.embedding_dimension}."
        )
    elif namespace_count == 0 and total_count:
        detail = (
            f"Pinecone index {settings.pinecone_index!r} contains {total_count} vectors, "
            f"but namespace {settings.pinecone_namespace!r} contains none. Set "
            "PINECONE_NAMESPACE to the namespace used during ingestion, or ingest into "
            "the configured namespace with `python -m src.ingestion`."
        )
    elif namespace_count == 0:
        detail = (
            f"Pinecone index {settings.pinecone_index!r} is empty in namespace "
            f"{settings.pinecone_namespace!r}. Run `python -m src.ingestion` and check "
            "the ingestion output before querying."
        )
    else:
        detail = (
            f"Pinecone index {settings.pinecone_index!r} is ready with "
            f"{namespace_count} vectors in namespace {settings.pinecone_namespace!r}."
        )

    return IndexReadiness(
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        exists=True,
        ready=index_ready and dimension == settings.embedding_dimension and namespace_count > 0,
        dimension=dimension,
        namespace_vector_count=namespace_count,
        total_vector_count=total_count,
        detail=detail,
    )


def require_retrievable_index(settings: Settings, client=None) -> IndexReadiness:
    readiness = inspect_index(settings, client)
    if not readiness.ready:
        raise KnowledgeBaseNotReadyError(readiness.detail)
    return readiness


def normalize_page_number(value) -> int | None:
    """Accept integral numeric/string metadata without silently losing valid matches."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        page = value
    elif isinstance(value, float) and value.is_integer():
        page = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        page = int(value.strip())
    else:
        return None
    return page if page > 0 else None


def wait_for_upserted_vectors(
    index,
    vector_ids: list[str],
    namespace: str,
    timeout_seconds: int,
) -> None:
    """Wait until Pinecone can fetch every vector acknowledged by the upsert."""
    deadline = time.monotonic() + timeout_seconds
    pending = set(vector_ids)
    while pending and time.monotonic() < deadline:
        pending_ids = list(pending)
        for offset in range(0, len(pending_ids), 1000):
            batch = pending_ids[offset : offset + 1000]
            response = index.fetch(ids=batch, namespace=namespace or None)
            found = _field(response, "vectors", {}) or {}
            pending.difference_update(found.keys())
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError(
            f"Pinecone acknowledged the upsert but did not make "
            f"{len(pending)} of {len(vector_ids)} vectors visible in namespace "
            f"{namespace!r} within {timeout_seconds} seconds. Retry ingestion and "
            "check Pinecone index status."
        )


def load_and_split_pdf(
    pdf_path: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF not found at {path}. Place the supplied eBook at this path "
            "or set PDF_PATH in .env."
        )

    from pypdf import PdfReader

    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": page_number},
        )
        for page_number, page in enumerate(PdfReader(str(path)).pages, start=1)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(pages)


def ensure_pinecone_index(settings: Settings) -> None:
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = client.list_indexes().names()
    if settings.pinecone_index in existing_indexes:
        description = client.describe_index(settings.pinecone_index)
        if description.dimension != settings.embedding_dimension:
            raise ValueError(
                f"Pinecone index {settings.pinecone_index!r} has dimension "
                f"{description.dimension}; expected {settings.embedding_dimension}."
            )
    else:
        client.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )

    deadline = time.monotonic() + settings.pinecone_ready_timeout_seconds
    while time.monotonic() < deadline:
        description = client.describe_index(settings.pinecone_index)
        if description.dimension != settings.embedding_dimension:
            raise ValueError(
                f"Pinecone index {settings.pinecone_index!r} has dimension "
                f"{description.dimension}; expected {settings.embedding_dimension}."
            )
        if description.status.ready:
            return
        time.sleep(2)
    raise TimeoutError(
        f"Pinecone index {settings.pinecone_index!r} did not become ready within "
        f"{settings.pinecone_ready_timeout_seconds} seconds."
    )


def create_vector_store(
    settings: Settings, embeddings: Embeddings
) -> "PineconeVectorStore":
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    client = Pinecone(api_key=settings.pinecone_api_key)
    return PineconeVectorStore(
        index=client.Index(settings.pinecone_index),
        embedding=embeddings,
        namespace=settings.pinecone_namespace or None,
    )


def create_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class PineconeRetriever:
    def __init__(self, settings: Settings):
        require_retrievable_index(settings)
        embeddings = create_embeddings(settings)
        self.store = create_vector_store(settings, embeddings)
        self.k = settings.retrieval_k
        self.namespace = settings.pinecone_namespace
        self.settings = settings
        self.minimum_score = settings.retrieval_min_score

    def retrieve(self, query: str) -> list["ContextChunk"]:
        from src.graph import ContextChunk

        matches = self.store.similarity_search_with_score(
            query, k=self.k, namespace=self.namespace or None
        )
        if not matches:
            require_retrievable_index(self.settings)
            return []
        chunks = []
        for document, score in matches:
            if float(score) < self.minimum_score:
                continue
            page = normalize_page_number(document.metadata.get("page"))
            if page is None:
                raise ValueError(
                    "Pinecone returned a matching chunk without a valid positive "
                    "'page' metadata value. Re-index the PDF with "
                    "`python -m src.ingestion` to restore page metadata."
                )
            chunks.append(
                ContextChunk(
                    text=document.page_content,
                    source=str(document.metadata.get("source", "Ebook-Agentic-AI.pdf")),
                    page=page,
                )
            )
        return chunks


def ingest(settings: Settings) -> int:
    documents = load_and_split_pdf(
        settings.pdf_path, settings.chunk_size, settings.chunk_overlap
    )
    if not documents:
        raise ValueError(f"No text could be extracted from {settings.pdf_path}")
    ensure_pinecone_index(settings)
    embeddings = create_embeddings(settings)
    store = create_vector_store(settings, embeddings)
    ids = [
        hashlib.sha256(
            f"{document.metadata['source']}:{document.metadata['page']}:"
            f"{document.metadata.get('start_index', 0)}:{document.page_content}".encode()
        ).hexdigest()
        for document in documents
    ]
    upserted_ids = store.add_documents(
        documents,
        ids=ids,
        namespace=settings.pinecone_namespace or None,
    )
    from pinecone import Pinecone

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index)
    wait_for_upserted_vectors(
        index,
        upserted_ids,
        settings.pinecone_namespace,
        settings.pinecone_upsert_timeout_seconds,
    )
    readiness = inspect_index(settings)
    print(
        f"Upserted {len(documents)} chunks to index {settings.pinecone_index!r}, "
        f"namespace {settings.pinecone_namespace!r}. "
        f"Index reports {readiness.namespace_vector_count} vectors there."
    )
    return len(documents)


def main() -> None:
    count = ingest(Settings.from_env())
    print(f"Indexed {count} chunks from the configured PDF into Pinecone.")


if __name__ == "__main__":
    main()
