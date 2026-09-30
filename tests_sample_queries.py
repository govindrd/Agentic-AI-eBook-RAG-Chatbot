"""Mocked graph tests and representative eBook questions (no paid API calls)."""

from collections.abc import Sequence

from unittest.mock import Mock

from fastapi.testclient import TestClient
from langchain_openai.chat_models.base import OpenAIRateLimitError

from app import create_app
from src.graph import (
    REFUSAL_ANSWER,
    ChatResponse,
    ContextChunk,
    GroundingAssessment,
    LocalAnswerGenerator,
    LocalGroundingGrader,
    build_graph,
)
from src.config import Settings
from src.ingestion import (
    IndexReadiness,
    inspect_index,
    normalize_page_number,
    require_retrievable_index,
    wait_for_upserted_vectors,
)

SAMPLE_QUERIES = [
    "How does the eBook define an AI agent?",
    "What planning strategies are described?",
    "How do tools extend an agent's capabilities?",
    "What role does memory play in an agentic system?",
    "How should an agent handle a task it cannot complete?",
    "What limitations or risks does the eBook identify?",
]


class FakeRetriever:
    def __init__(self, chunks: list[ContextChunk]):
        self.chunks = chunks

    def retrieve(self, query: str) -> list[ContextChunk]:
        return self.chunks


class FakeGenerator:
    def __init__(self, answers: Sequence[str]):
        self.answers = list(answers)
        self.calls: list[bool] = []

    def generate(self, query: str, chunks: list[ContextChunk], retry: bool) -> str:
        self.calls.append(retry)
        return self.answers.pop(0)


class FakeGrader:
    def __init__(self, assessments: Sequence[GroundingAssessment]):
        self.assessments = list(assessments)

    def grade(
        self, query: str, answer: str, chunks: list[ContextChunk]
    ) -> GroundingAssessment:
        return self.assessments.pop(0)


CHUNK = ContextChunk(text="Agents can plan and use tools.", source="ebook.pdf", page=7)


def test_sample_queries_are_six_distinct_questions() -> None:
    assert len(SAMPLE_QUERIES) == 6
    assert len(set(SAMPLE_QUERIES)) == 6


def test_settings_allow_local_ingestion_without_openai_key(monkeypatch) -> None:
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")

    settings = Settings.from_env()

    assert settings.openai_api_key is None
    assert settings.embedding_dimension == 384
    assert settings.pinecone_index == "agentic-ai-ebook-local"


def test_chat_requires_openai_key() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test-pinecone-key")

    try:
        settings.require_openai_api_key()
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)
    else:
        raise AssertionError("Expected chat settings to require an OpenAI API key")


def test_chat_provider_defaults_to_local_without_openai_key(monkeypatch) -> None:
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")

    settings = Settings.from_env()

    assert settings.chat_provider == "local"
    assert settings.openai_api_key is None
    assert settings.local_chat_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_openai_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("CHAT_PROVIDER", "openai")

    try:
        Settings.from_env()
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)
    else:
        raise AssertionError("OpenAI provider must require an API key")


class FakeLocalLanguageModel:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_local_generator_only_supplies_retrieved_evidence_and_page_citations() -> None:
    settings = Settings(
        openai_api_key=None,
        pinecone_api_key="test",
        local_max_new_tokens=64,
    )
    model = FakeLocalLanguageModel(["Supported claim [page 7]."])
    generator = LocalAnswerGenerator(model, settings)

    answer = generator.generate("What can agents do?", [CHUNK], retry=False)

    assert answer == "Supported claim [page 7]."
    assert "[ebook.pdf, page 7]" in model.prompts[0]
    assert CHUNK.text in model.prompts[0]
    assert "Never use outside knowledge" in model.prompts[0]


def test_local_generator_extracts_definition_from_retrieved_page() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    model = FakeLocalLanguageModel([])
    generator = LocalAnswerGenerator(model, settings)
    definition_chunk = ContextChunk(
        text=(
            "A Journey into the Heart of Autonomous Intelligence\n\n"
            "Agentic AI refers to systems capable of autonomous decision-making "
            "and action in pursuit of specific objectives."
        ),
        source="Ebook-Agentic-AI.pdf",
        page=18,
    )

    answer = generator.generate("What is Agentic AI?", [definition_chunk], retry=False)

    assert answer == (
        "Agentic AI refers to systems capable of autonomous decision-making "
        "and action in pursuit of specific objectives. [page 18]"
    )
    assert model.prompts == []


def test_local_grader_accepts_exact_cited_definition_without_model_call() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    model = FakeLocalLanguageModel([])
    grader = LocalGroundingGrader(model, settings)
    definition_chunk = ContextChunk(
        text=(
            "Agentic AI refers to systems capable of autonomous decision-making "
            "and action in pursuit of specific objectives."
        ),
        source="Ebook-Agentic-AI.pdf",
        page=18,
    )

    assessment = grader.grade(
        "What is Agentic AI?",
        "Agentic AI refers to systems capable of autonomous decision-making "
        "and action in pursuit of specific objectives. [page 18]",
        [definition_chunk],
    )

    assert assessment.grounded is True
    assert assessment.confidence_score == 0.99
    assert model.prompts == []


def test_local_grounding_grader_parses_structured_result() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    model = FakeLocalLanguageModel(
        [
            '{"grounded": true, "confidence_score": 0.91, '
            '"explanation": "The claim appears in page 7."}'
        ]
    )
    grader = LocalGroundingGrader(model, settings)

    assessment = grader.grade(
        "What can agents do?", "Agents can plan [page 7].", [CHUNK]
    )

    assert assessment.grounded is True
    assert assessment.confidence_score == 0.91
    assert "eBook excerpts" in model.prompts[0]


def test_local_grounding_grader_fails_closed_on_malformed_output() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    grader = LocalGroundingGrader(
        FakeLocalLanguageModel(["I think this is supported."]), settings
    )

    assessment = grader.grade("Question?", "Answer.", [CHUNK])

    assert assessment.grounded is False
    assert assessment.confidence_score == 0.0
    assert "invalid JSON" in assessment.explanation


def test_openai_quota_error_returns_actionable_service_response() -> None:
    response = Mock(status_code=429, request=Mock())
    error = OpenAIRateLimitError(
        "quota exhausted",
        response=response,
        body={"error": {"code": "credit_balance_exhausted"}},
    )
    client = TestClient(create_app(answer_graph=lambda query: (_ for _ in ()).throw(error)))

    result = client.post("/chat", json={"query": "A sample question?"})

    assert result.status_code == 503
    assert "CHAT_PROVIDER=local" in result.json()["detail"]


class FakeIndex:
    def __init__(self, vector_count: int):
        self.vector_count = vector_count

    def describe_index_stats(self):
        return {
            "total_vector_count": self.vector_count,
            "namespaces": {"": {"vector_count": self.vector_count}},
        }


class FakePineconeClient:
    def __init__(self, *, index_names, dimension=384, ready=True, vector_count=0):
        self.index_names = index_names
        self.dimension = dimension
        self.ready = ready
        self.vector_count = vector_count

    def list_indexes(self):
        return self

    def names(self):
        return self.index_names

    def describe_index(self, index_name):
        return {
            "dimension": self.dimension,
            "status": {"ready": self.ready},
        }

    def Index(self, index_name):
        return FakeIndex(self.vector_count)


def test_index_diagnostics_catches_missing_index_and_names_ingest_command() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")

    status = inspect_index(settings, FakePineconeClient(index_names=[]))

    assert status.ready is False
    assert status.exists is False
    assert "python -m src.ingestion" in status.detail


def test_index_diagnostics_distinguishes_empty_index() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")

    status = inspect_index(
        settings,
        FakePineconeClient(
            index_names=[settings.pinecone_index],
            dimension=settings.embedding_dimension,
            vector_count=0,
        ),
    )

    assert status.ready is False
    assert "is empty" in status.detail


def test_index_diagnostics_reports_embedding_dimension_mismatch() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")

    status = inspect_index(
        settings,
        FakePineconeClient(
            index_names=[settings.pinecone_index],
            dimension=settings.embedding_dimension + 1,
            vector_count=10,
        ),
    )

    assert status.ready is False
    assert "dimension" in status.detail


def test_require_index_ready_accepts_populated_matching_namespace() -> None:
    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    client = FakePineconeClient(
        index_names=[settings.pinecone_index],
        dimension=settings.embedding_dimension,
        vector_count=8,
    )

    status = require_retrievable_index(settings, client)

    assert status.ready is True
    assert status.namespace_vector_count == 8


def test_index_diagnostics_detect_namespace_mismatch() -> None:
    settings = Settings(
        openai_api_key=None,
        pinecone_api_key="test",
        pinecone_namespace="configured",
    )
    client = FakePineconeClient(
        index_names=[settings.pinecone_index],
        dimension=settings.embedding_dimension,
        vector_count=8,
    )

    status = inspect_index(settings, client)

    assert status.ready is False
    assert "namespace 'configured' contains none" in status.detail


def test_upsert_verification_waits_for_expected_vectors() -> None:
    class FetchIndex:
        def fetch(self, ids, namespace):
            return {"vectors": {vector_id: {} for vector_id in ids}}

    wait_for_upserted_vectors(FetchIndex(), ["one", "two"], "", timeout_seconds=1)


def test_upsert_verification_fails_clearly_when_vectors_never_appear() -> None:
    class EmptyFetchIndex:
        def fetch(self, ids, namespace):
            return {"vectors": {}}

    try:
        wait_for_upserted_vectors(
            EmptyFetchIndex(), ["one"], "docs", timeout_seconds=0
        )
    except TimeoutError as error:
        assert "did not make" in str(error)
        assert "namespace 'docs'" in str(error)
    else:
        raise AssertionError("Missing vectors must fail ingestion verification")


def test_page_metadata_normalizes_common_pinecone_numeric_encodings() -> None:
    assert normalize_page_number(7) == 7
    assert normalize_page_number(7.0) == 7
    assert normalize_page_number(" 7 ") == 7
    assert normalize_page_number(0) is None
    assert normalize_page_number(7.5) is None
    assert normalize_page_number(True) is None


def test_retriever_keeps_integral_float_page_metadata() -> None:
    from src.ingestion import PineconeRetriever

    retriever = object.__new__(PineconeRetriever)

    class SearchStore:
        def similarity_search_with_score(self, query, k, namespace):
            class Match:
                page_content = "Evidence"
                metadata = {"page": 7.0, "source": "ebook.pdf"}

            return [(Match(), 0.8)]

    retriever.store = SearchStore()
    retriever.k = 5
    retriever.namespace = ""
    retriever.settings = Settings(openai_api_key=None, pinecone_api_key="test")
    retriever.minimum_score = 0.3

    chunks = retriever.retrieve("question")

    assert len(chunks) == 1
    assert chunks[0].page == 7


def test_retriever_reports_invalid_page_metadata_instead_of_dropping_match() -> None:
    from src.ingestion import PineconeRetriever

    retriever = object.__new__(PineconeRetriever)

    class SearchStore:
        def similarity_search_with_score(self, query, k, namespace):
            class Match:
                page_content = "Evidence"
                metadata = {"source": "ebook.pdf"}

            return [(Match(), 0.8)]

    retriever.store = SearchStore()
    retriever.k = 5
    retriever.namespace = ""
    retriever.settings = Settings(openai_api_key=None, pinecone_api_key="test")
    retriever.minimum_score = 0.3

    try:
        retriever.retrieve("question")
    except ValueError as error:
        assert "valid positive 'page' metadata" in str(error)
    else:
        raise AssertionError("Invalid page metadata must not silently drop a match")


def test_retriever_filters_low_similarity_matches() -> None:
    from src.ingestion import PineconeRetriever

    retriever = object.__new__(PineconeRetriever)

    class SearchStore:
        def similarity_search_with_score(self, query, k, namespace):
            class Match:
                page_content = "Unrelated content"
                metadata = {"page": 32, "source": "ebook.pdf"}

            return [(Match(), 0.09)]

    retriever.store = SearchStore()
    retriever.k = 5
    retriever.namespace = ""
    retriever.settings = Settings(openai_api_key=None, pinecone_api_key="test")
    retriever.minimum_score = 0.3

    assert retriever.retrieve("What is the capital of France?") == []


def test_empty_pinecone_search_surfaces_unready_knowledge_base(monkeypatch) -> None:
    from src.ingestion import KnowledgeBaseNotReadyError, PineconeRetriever

    retriever = object.__new__(PineconeRetriever)

    class EmptySearchStore:
        def similarity_search_with_score(self, query, k, namespace):
            return []

    def report_unready(settings):
        raise KnowledgeBaseNotReadyError("No indexed vectors in configured namespace.")

    monkeypatch.setattr("src.ingestion.require_retrievable_index", report_unready)
    retriever.store = EmptySearchStore()
    retriever.k = 5
    retriever.namespace = ""
    retriever.settings = Settings(openai_api_key=None, pinecone_api_key="test")
    retriever.minimum_score = 0.3

    try:
        retriever.retrieve("question")
    except KnowledgeBaseNotReadyError as error:
        assert "No indexed vectors" in str(error)
    else:
        raise AssertionError("Empty index must not be confused with an out-of-scope query")


def test_grounded_answer_contains_retrieved_evidence_and_confidence() -> None:
    answer = build_graph(
        FakeRetriever([CHUNK]),
        FakeGenerator(["Agents can plan and use tools [page 7]."]),
        FakeGrader(
            [
                GroundingAssessment(
                    grounded=True,
                    confidence_score=0.94,
                    explanation="Supported by page 7.",
                )
            ]
        ),
    )

    response = answer("What can agents do?")

    assert response.final_answer == "Agents can plan and use tools [page 7]."
    assert response.retrieved_context_chunks == [CHUNK.text]
    assert response.confidence_score == 0.94


def test_unsupported_answer_retries_then_returns_grounded_draft() -> None:
    generator = FakeGenerator(["Unsupported claim.", "Supported answer [page 7]."])
    grader = FakeGrader(
        [
            GroundingAssessment(
                grounded=False, confidence_score=0.2, explanation="Not in context."
            ),
            GroundingAssessment(
                grounded=True, confidence_score=0.88, explanation="Supported."
            ),
        ]
    )
    answer = build_graph(FakeRetriever([CHUNK]), generator, grader, max_attempts=2)

    response = answer("What can agents do?")

    assert response.final_answer == "Supported answer [page 7]."
    assert generator.calls == [False, True]
    assert response.confidence_score == 0.88


def test_empty_retrieval_refuses_without_calling_generator() -> None:
    generator = FakeGenerator([])
    answer = build_graph(FakeRetriever([]), generator, FakeGrader([]))

    response = answer("What does the eBook say?")

    assert response.final_answer == REFUSAL_ANSWER
    assert response.retrieved_context_chunks == []
    assert response.confidence_score == 0.0
    assert generator.calls == []


def test_repeated_ungrounded_drafts_end_in_refusal() -> None:
    generator = FakeGenerator(["Unsupported first.", "Unsupported again."])
    grader = FakeGrader(
        [
            GroundingAssessment(
                grounded=False, confidence_score=0.1, explanation="Unsupported."
            ),
            GroundingAssessment(
                grounded=False, confidence_score=0.1, explanation="Still unsupported."
            ),
        ]
    )
    answer = build_graph(FakeRetriever([CHUNK]), generator, grader, max_attempts=2)

    response = answer("Tell me something unsupported.")

    assert response.final_answer == REFUSAL_ANSWER
    assert generator.calls == [False, True]
    assert response.confidence_score == 0.0


def test_api_exposes_health_and_structured_chat_response() -> None:
    def fake_answer(query: str) -> ChatResponse:
        return ChatResponse(
            query=query,
            final_answer="Supported [page 7].",
            retrieved_context_chunks=[CHUNK.text],
            confidence_score=0.9,
        )

    client = TestClient(create_app(answer_graph=fake_answer))

    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/chat", json={"query": "A sample question?"})

    assert response.status_code == 200
    assert set(response.json()) == {
        "query",
        "final_answer",
        "retrieved_context_chunks",
        "confidence_score",
    }
    assert response.json()["retrieved_context_chunks"] == [CHUNK.text]


def test_ready_endpoint_explains_empty_index(monkeypatch) -> None:
    import app as app_module

    settings = Settings(openai_api_key=None, pinecone_api_key="test")
    monkeypatch.setattr(app_module.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(
        app_module,
        "inspect_index",
        lambda _: IndexReadiness(
            index_name=settings.pinecone_index,
            namespace=settings.pinecone_namespace,
            exists=True,
            ready=False,
            dimension=settings.embedding_dimension,
            namespace_vector_count=0,
            total_vector_count=0,
            detail="Index is empty. Run `python -m src.ingestion`.",
        ),
    )
    client = TestClient(create_app(answer_graph=lambda query: None))

    response = client.get("/ready")

    assert response.status_code == 503
    assert "python -m src.ingestion" in response.json()["detail"]
