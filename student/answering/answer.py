import asyncio
import json
import tqdm
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from student.models import (UnansweredQuestion,
                            AnsweredQuestion,
                            MinimalSource,
                            StudentSearchResults,
                            StudentSearchResultsAndAnswer,
                            MinimalAnswer)
from student.searching import Search


class Answer:
    """Generate natural language answers using an LLM.

    Uses context retrieved from the BM25 knowledge base.
    """

    def __init__(self) -> None:
        """Initialize the LLM, prompt template, and chain."""
        self.llm = ChatOllama(model="qwen3:0.6b")
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm

    def answer(self, question: UnansweredQuestion, k: int,
               sources: list[MinimalSource] | None = None) -> AnsweredQuestion:
        """Answer a single question using retrieved context.

        Args:
            question: The question to answer.
            k: Number of sources to retrieve if sources is not provided.
            sources: Pre-retrieved sources; fetched automatically if None.

        Returns:
            An AnsweredQuestion with sources and generated answer.
        """
        if (not sources):
            sources = Search().search(question, k).retrieved_sources
        context = "\n\n".join(s.text for s in sources)
        answer = self.chain.invoke({
            "context": context,
            "question": question.question
        })
        return (AnsweredQuestion(
            **question.model_dump(),
            sources=sources,
            answer=str(answer.content)
        ))

    async def _answer_async(self, question: UnansweredQuestion,
                            sources: list[MinimalSource]) -> AnsweredQuestion:
        """Async version of answer for concurrent batch processing.

        Args:
            question: The question to answer.
            sources: Pre-retrieved sources to use as context.

        Returns:
            An AnsweredQuestion with sources and generated answer.
        """
        context = "\n\n".join(s.text for s in sources)
        answer = await self.chain.ainvoke({
            "context": context,
            "question": question.question
        })
        return AnsweredQuestion(
            **question.model_dump(),
            sources=sources,
            answer=str(answer.content)
        )

    def answer_dataset(self, student_search_result_path: str,
                       save_directory: str) -> None:
        """Generate answers for all questions in a search results file.

        Args:
            student_search_result_path: Path to the search results JSON.
            save_directory: Directory where answered results will be saved.
        """
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
