import hashlib
import json
import tqdm
from pathlib import Path
from openai import OpenAI
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
        self.cache: dict[str, AnsweredQuestion] = {}
        self.cache_path = Path("data/processed/cache")
        try:
            with open(self.cache_path, "r") as file:
                self.cache = {
                    k: AnsweredQuestion.model_validate(v)
                    for k, v in json.load(file).items()
                }
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            pass

    def _build_messages(
            self, question: str,
            context: str) -> list[ChatCompletionUserMessageParam]:
        return [ChatCompletionUserMessageParam(
            role="user",
            content=PROMPT_TEMPLATE.format(context=context, question=question)
        )]

    def _cache_key(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()

    def answer(self, question: UnansweredQuestion, k: int,
               sources: list[MinimalSource] | None = None,
               cache: bool = True) -> AnsweredQuestion:
        key = self._cache_key(question.question)
        if (cache and self.cache.get(key)):
            print("LOADED FROM CACHE")
            return (self.cache[key])
        if not sources:
            sources = Search().search(question, k).retrieved_sources
        context = "\n\n".join(s.text for s in sources)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self._build_messages(question.question, context)
        )
        result = AnsweredQuestion(
            **question.model_dump(),
            sources=sources,
            answer=str(response.choices[0].message.content)
        )
        if (cache):
            self.cache[key] = result
            try:
                with open(self.cache_path, "w") as file:
                    file.write(json.dumps(
                        {k: v.model_dump() for k, v in self.cache.items()},
                        indent=4
                    ))
            except (FileNotFoundError, PermissionError):
                print("Cache file not writable")
        return (result)


    def answer_dataset(self, student_search_result_path: str,
                       save_directory: str,
                       cache: bool) -> None:
        try:
            with open(student_search_result_path, "r") as file:
                data = StudentSearchResults.model_validate(json.load(file))
        except (FileNotFoundError, PermissionError) as e:
            print(e)
            print(f"Cannot read {student_search_result_path}")
            return

        out = []
        for s in tqdm.tqdm(data.search_results):
            out.append(self.answer(
                UnansweredQuestion(
                    question_id=s.question_id, question=s.question),
                data.k,
                sources=s.retrieved_sources,
                cache=cache
            ))
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
        if (cache):
            try:
                with open(self.cache_path, "w") as file:
                    file.write(json.dumps(
                        {k: v.model_dump() for k, v in self.cache.items()}
                    ))
            except (FileNotFoundError, PermissionError):
                print("Cache file not writable")
