import asyncio
import json
import tqdm
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from student.models import (UnansweredQuestion,
                            AnsweredQuestion,
                            MinimalSource,
                            StudentSearchResults,
                            StudentSearchResultsAndAnswer,
                            MinimalAnswer)
from student.searching import Search

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen3:0.6b"
PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


class Answer:
    """Generate natural language answers using an LLM."""

    def __init__(self) -> None:
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="unused")
        self.async_client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL, api_key="unused")

    def _build_messages(
            self, question: str,
            context: str) -> list[ChatCompletionUserMessageParam]:
        return [ChatCompletionUserMessageParam(
            role="user",
            content=PROMPT_TEMPLATE.format(context=context, question=question)
        )]

    def answer(self, question: UnansweredQuestion, k: int,
               sources: list[MinimalSource] | None = None) -> AnsweredQuestion:
        if not sources:
            sources = Search().search(question, k).retrieved_sources
        context = "\n\n".join(s.text for s in sources)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self._build_messages(question.question, context)
        )
        return AnsweredQuestion(
            **question.model_dump(),
            sources=sources,
            answer=str(response.choices[0].message.content)
        )

    async def _answer_async(self, question: UnansweredQuestion,
                            sources: list[MinimalSource]) -> AnsweredQuestion:
        context = "\n\n".join(s.text for s in sources)
        response = await self.async_client.chat.completions.create(
            model=MODEL,
            messages=self._build_messages(question.question, context)
        )
        return AnsweredQuestion(
            **question.model_dump(),
            sources=sources,
            answer=str(response.choices[0].message.content)
        )

    def answer_dataset(self, student_search_result_path: str,
                       save_directory: str) -> None:
        try:
            with open(student_search_result_path, "r") as file:
                data = StudentSearchResults.model_validate(json.load(file))
        except (FileNotFoundError, PermissionError) as e:
            print(e)
            print(f"Cannot read {student_search_result_path}")
            return

        async def run_all() -> list[AnsweredQuestion]:
            tasks = [
                self._answer_async(
                    UnansweredQuestion(
                        question_id=s.question_id, question=s.question),
                    s.retrieved_sources
                )
                for s in data.search_results
            ]
            results = await tqdm.asyncio.tqdm.gather(*tasks)
            return list(results)

        out = asyncio.run(run_all())
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_dir / "student_answers.json", "w") as file:
                file.write(json.dumps(
                    StudentSearchResultsAndAnswer(
                        search_results=[MinimalAnswer(
                            question_id=a.question_id,
                            question=a.question,
                            retrieved_sources=a.sources,
                            answer=a.answer
                        ) for a in out],
                        k=data.k
                    ).model_dump(),
                    indent=4
                ))
        except (FileNotFoundError, PermissionError) as e:
            print(e)
            print(f"Cannot write to {save_directory}")
