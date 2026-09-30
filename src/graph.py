"""LangGraph retrieval, generation, groundedness grading, retry, and refusal."""

import json
import re
from typing import Protocol, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config import Settings

REFUSAL_ANSWER = "I can't answer that from the provided Agentic AI eBook."


class ContextChunk(BaseModel):
    text: str
    source: str
    page: int = Field(ge=1)


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class GroundingAssessment(BaseModel):
    grounded: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[ContextChunk]: ...


class AnswerGenerator(Protocol):
    def generate(self, query: str, chunks: list[ContextChunk], retry: bool) -> str: ...


class GroundingGrader(Protocol):
    def grade(
        self, query: str, answer: str, chunks: list[ContextChunk]
    ) -> GroundingAssessment: ...


class LocalTextModel(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class HuggingFaceTextModel:
    """One shared local instruct model for generation and groundedness grading."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = result[0]["generated_text"]
        if not isinstance(generated, str):
            raise TypeError("The local language model returned non-text content")
        return generated.strip()


class GraphState(TypedDict):
    query: str
    chunks: list[ContextChunk]
    answer: str
    confidence: float
    attempts: int
    grounded: bool


class OpenAIAnswerGenerator:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.require_openai_api_key(),
            temperature=0,
        )

    def generate(self, query: str, chunks: list[ContextChunk], retry: bool) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        context = "\n\n".join(
            f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        retry_instruction = (
            "Your previous draft was not fully supported. Remove every unsupported claim."
            if retry
            else ""
        )
        result = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "Answer only with facts explicitly supported by the supplied eBook "
                        "excerpts. Do not use outside knowledge or infer missing details. "
                        "Cite each supported claim using its source page, such as [page 3]. "
                        "If the excerpts do not answer the question, reply exactly: "
                        f"{REFUSAL_ANSWER}\n{retry_instruction}"
                    )
                ),
                HumanMessage(content=f"Question: {query}\n\neBook excerpts:\n{context}"),
            ]
        )
        if not isinstance(result.content, str):
            raise TypeError("The chat model returned non-text content")
        return result.content.strip()


class OpenAIGroundingGrader:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.require_openai_api_key(),
            temperature=0,
        )
        self.model = model.with_structured_output(GroundingAssessment)

    def grade(
        self, query: str, answer: str, chunks: list[ContextChunk]
    ) -> GroundingAssessment:
        from langchain_core.messages import HumanMessage, SystemMessage

        context = "\n\n".join(
            f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        return self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "Assess whether every factual claim in the answer is supported "
                        "directly by the supplied excerpts. An explicit refusal with no "
                        "factual claims is grounded. Return grounded=false for unsupported "
                        "claims, outside knowledge, or invented citations. Confidence is "
                        "your 0-to-1 confidence in this assessment."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {query}\nAnswer: {answer}\n\n"
                        f"Retrieved excerpts:\n{context}"
                    )
                ),
            ]
        )


class LocalAnswerGenerator:
    def __init__(self, model: LocalTextModel, settings: Settings):
        self.model = model
        self.max_new_tokens = settings.local_max_new_tokens

    def generate(self, query: str, chunks: list[ContextChunk], retry: bool) -> str:
        if not retry:
            definition = _extract_definition(query, chunks)
            if definition:
                return definition

        context = "\n\n".join(
            f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        retry_note = (
            "This is a retry. Correct the previous unsupported draft and include only "
            "claims explicitly supported by the excerpts.\n"
            if retry
            else ""
        )
        prompt = (
            "You answer questions using only the supplied eBook excerpts. Never use "
            "outside knowledge. Cite supported claims with the provided page number. "
            f"If the excerpts do not answer the question, reply exactly: {REFUSAL_ANSWER}\n"
            f"{retry_note}\nQuestion:\n{query}\n\nExcerpts:\n{context}\n\nAnswer:"
        )
        return self.model.generate(prompt, self.max_new_tokens)


class LocalGroundingGrader:
    def __init__(self, model: LocalTextModel, settings: Settings):
        self.model = model
        self.max_new_tokens = min(settings.local_max_new_tokens, 192)

    def grade(
        self, query: str, answer: str, chunks: list[ContextChunk]
    ) -> GroundingAssessment:
        citation = re.search(r"\s+\[page (\d+)\]\s*$", answer)
        if citation:
            cited_page = int(citation.group(1))
            answer_text = _normalize_whitespace(answer[: citation.start()])
            for chunk in chunks:
                if (
                    chunk.page == cited_page
                    and answer_text.casefold()
                    in _normalize_whitespace(chunk.text).casefold()
                ):
                    return GroundingAssessment(
                        grounded=True,
                        confidence_score=0.99,
                        explanation="The cited answer is an exact excerpt from the cited page.",
                    )

        context = "\n\n".join(
            f"[{chunk.source}, page {chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        prompt = (
            "Judge whether every factual claim in the answer is directly supported by "
            "the supplied eBook excerpts. Do not use outside knowledge. "
            "A plain refusal with no factual "
            "claims is grounded. Return only one JSON object with exactly these fields: "
            '{"grounded": boolean, "confidence_score": number from 0 to 1, '
            '"explanation": string}. Set grounded=false if any claim is unsupported or '
            "has an invented citation.\n\n"
            f"Question:\n{query}\n\nAnswer:\n{answer}\n\nExcerpts:\n{context}\n\nJSON:"
        )
        result = self.model.generate(prompt, self.max_new_tokens)
        start = result.find("{")
        end = result.rfind("}")
        if start < 0 or end < start:
            return GroundingAssessment(
                grounded=False,
                confidence_score=0.0,
                explanation="The local grounding model returned invalid JSON.",
            )
        try:
            assessment = GroundingAssessment.model_validate_json(result[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            return GroundingAssessment(
                grounded=False,
                confidence_score=0.0,
                explanation="The local grounding model returned an invalid assessment.",
            )
        return assessment


def _extract_definition(query: str, chunks: list[ContextChunk]) -> str | None:
    match = re.match(r"\s*(?:what is|define)\s+(.+?)[?.]?\s*$", query, re.IGNORECASE)
    if not match:
        return None

    subject = re.split(
        r"\s+(?:as outlined|according to|in the|from the)\b",
        match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    subject = re.sub(r"^(?:the\s+)?(?:core\s+)?definition\s+of\s+", "", subject, flags=re.I)
    ignored = {"a", "an", "the", "of", "what", "is", "define", "core", "definition"}
    terms = {
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9]+", subject)
        if term.casefold() not in ignored
    }
    if not terms:
        return None

    cues = ("refers to", "is defined as", "means", "is about", "is like")
    candidates: list[tuple[int, int, str, int]] = []
    for chunk in chunks:
        for paragraph in re.split(r"\n\s*\n", chunk.text):
            text = _normalize_whitespace(paragraph)
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                folded = sentence.casefold()
                if terms.issubset(set(re.findall(r"[a-z0-9]+", folded))):
                    cue_rank = next(
                        (
                            len(cues) - index
                            for index, cue in enumerate(cues)
                            if cue in folded
                        ),
                        0,
                    )
                    candidates.append((cue_rank, len(terms), sentence, chunk.page))

    if not candidates:
        return None
    _, _, sentence, page = max(candidates, key=lambda item: (item[0], item[1]))
    return f"{sentence} [page {page}]"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_graph(
    retriever: Retriever,
    generator: AnswerGenerator,
    grader: GroundingGrader,
    max_attempts: int = 2,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    def retrieve_node(state: GraphState) -> dict:
        return {"chunks": retriever.retrieve(state["query"])}

    def generate_node(state: GraphState) -> dict:
        attempt = state["attempts"] + 1
        return {
            "answer": generator.generate(
                state["query"], state["chunks"], retry=attempt > 1
            ),
            "attempts": attempt,
        }

    def grade_node(state: GraphState) -> dict:
        assessment = grader.grade(state["query"], state["answer"], state["chunks"])
        return {
            "grounded": assessment.grounded,
            "confidence": assessment.confidence_score,
        }

    def refuse_node(_: GraphState) -> dict:
        return {"answer": REFUSAL_ANSWER, "confidence": 0.0, "grounded": True}

    def after_retrieval(state: GraphState) -> str:
        return "generate" if state["chunks"] else "refuse"

    def after_grade(state: GraphState) -> str:
        if state["grounded"]:
            return "finish"
        return "generate" if state["attempts"] < max_attempts else "refuse"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("refuse", refuse_node)
    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve", after_retrieval, {"generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("generate", "grade")
    workflow.add_conditional_edges(
        "grade", after_grade, {"finish": END, "generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("refuse", END)
    compiled = workflow.compile()

    def answer(query: str) -> ChatResponse:
        final_state = compiled.invoke(
            {
                "query": query,
                "chunks": [],
                "answer": "",
                "confidence": 0.0,
                "attempts": 0,
                "grounded": False,
            }
        )
        return ChatResponse(
            query=query,
            final_answer=final_state["answer"],
            retrieved_context_chunks=[
                chunk.text for chunk in final_state["chunks"]
            ],
            confidence_score=final_state["confidence"],
        )

    return answer


def create_production_graph(settings: Settings):
    from src.ingestion import PineconeRetriever

    if settings.chat_provider == "openai":
        generator = OpenAIAnswerGenerator(settings)
        grader = OpenAIGroundingGrader(settings)
    else:
        local_model = HuggingFaceTextModel(settings.local_chat_model)
        generator = LocalAnswerGenerator(local_model, settings)
        grader = LocalGroundingGrader(local_model, settings)

    return build_graph(
        retriever=PineconeRetriever(settings),
        generator=generator,
        grader=grader,
        max_attempts=settings.max_generation_attempts,
    )
