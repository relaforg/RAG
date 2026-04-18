import bm25s
import json
import tqdm
from pathlib import Path

from student.models import (
    MinimalSource, MinimalSearchResults, RagDataset, UnansweredQuestion,
    StudentSearchResults
)
from student.color import RESET, RED
from .query_expander import QueryExpander


class Search:
    """Retrieves relevant chunks from the BM25 index."""

    def __init__(self) -> None:
        """Load the BM25 index and chunks from disk."""
        self.expander = QueryExpander()
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")
        try:
            with open("data/processed/chunks", "r") as file:
                self.chunks = json.load(file)
        except (FileNotFoundError, PermissionError):
            raise ValueError

    def _unit_search(self, question: UnansweredQuestion,
                     k: int) -> MinimalSearchResults:
        """Retrieve top-k sources for a single question.

        Args:
            question: The question to search for.
            k: Number of results to retrieve.

        Returns:
            Search results with the top-k retrieved sources.
        """
        tokenized_prompt = bm25s.tokenize(question.question)

        idxs, _ = self.retriever.retrieve(tokenized_prompt, k=k)
        sources = [MinimalSource(**self.chunks[i]) for i in idxs[0]]

        return (MinimalSearchResults(
            **question.model_dump(),
            retrieved_sources=sources
        ))

    def _expanded_search(self, question: UnansweredQuestion,
                         k: int) -> MinimalSearchResults:
        questions = [question.question] + \
            self.expander.expand(question.question)

        score: dict[int, float] = {}
        for q in questions:
            tokenized = bm25s.tokenize(q)
            idxs, _ = self.retriever.retrieve(tokenized)
            for rank, i in enumerate(idxs[0]):
                score[i] = score.get(i, 0) + 1 / (rank + 1)

        top_k = sorted(score, key=score.__getitem__, reverse=True)[:k]
        sources = [MinimalSource(**self.chunks[i]) for i in top_k]

        return (MinimalSearchResults(
            **question.model_dump(),
            retrieved_sources=sources
        ))

    def search(self, question: UnansweredQuestion,
               k: int,
               query_expansion: bool) -> MinimalSearchResults:
        if (query_expansion):
            return (self._expanded_search(question, k))
        return (self._unit_search(question, k))

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str, query_expansion: bool) -> None:
        """Search all questions in a dataset and save results to disk.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of results to retrieve per question.
            save_directory: Directory where results will be saved.
        """
        if (query_expansion):
            search = self._expanded_search
        else:
            search = self._unit_search
        try:
            with open(dataset_path, "r") as file:
                rag_dataset = RagDataset(
                    **json.load(file)
                )
        except (FileNotFoundError, PermissionError):
            print(f"{dataset_path + RED} does not exists or cannnot "
                  f"be written {RESET}")
            return
        out = []
        for data in tqdm.tqdm(rag_dataset.rag_questions):
            out.append(search(data, k))
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_dir / dataset_path.split("/")[-1], "w") as file:
                file.write(json.dumps(
                    StudentSearchResults(
                        search_results=out,
                        k=k
                    ).model_dump(),
                    indent=4
                ))
        except (FileNotFoundError, PermissionError):
            print(f"{str(save_dir / 'student_search_results.json') + RED} does"
                  f" not exists or cannnot be written {RESET}")
