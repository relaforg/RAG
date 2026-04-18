import bm25s
import json
import tqdm
from pathlib import Path
import chromadb

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
        self.client = chromadb.PersistentClient(path="data/chroma")
        self.collection = self.client.get_or_create_collection("chunks")

    def _rrf(self, rankings: list[list[str]], k: int) -> list[MinimalSource]:
        score: dict[str, float] = {}
        for ranked_ids in rankings:
            for rank, doc_id in enumerate(ranked_ids):
                score[doc_id] = score.get(doc_id, 0) + 1 / (rank + 1)
        top_k = sorted(score, key=score.__getitem__, reverse=True)[:k]
        return [MinimalSource(**self.chunks[int(i)]) for i in top_k]

    def _bm25_ids(self, query: str, k: int) -> list[str]:
        tokenized = bm25s.tokenize(query)
        idxs, _ = self.retriever.retrieve(tokenized, k=k)
        return [str(i) for i in idxs[0]]

    def _chroma_ids(self, query: str, k: int) -> list[str]:
        return self.collection.query(
            query_texts=[query], n_results=k)["ids"][0]

    def search(self, question: UnansweredQuestion,
               k: int,
               query_expansion: bool = False,
               hybrid: bool = False) -> MinimalSearchResults:
        queries = (
            [question.question] + self.expander.expand(question.question)
            if query_expansion else [question.question]
        )
        rankings = []
        for q in queries:
            rankings.append(self._bm25_ids(q, k))
            if hybrid:
                rankings.append(self._chroma_ids(q, k))

        if len(rankings) == 1:
            sources = [MinimalSource(**self.chunks[int(i)])
                       for i in rankings[0]]
        else:
            sources = self._rrf(rankings, k)

        return MinimalSearchResults(
            **question.model_dump(),
            retrieved_sources=sources
        )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str, query_expansion: bool,
                       hybrid: bool = False) -> None:
        """Search all questions in a dataset and save results to disk.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of results to retrieve per question.
            save_directory: Directory where results will be saved.
        """
        def _search(q: UnansweredQuestion, k: int) -> MinimalSearchResults:
            return self.search(q, k, query_expansion, hybrid)
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
            out.append(_search(data, k))
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
