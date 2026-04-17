import bm25s
import json

from student.models import (
    MinimalSource, MinimalSearchResults, RagDataset, UnansweredQuestion
)
from student.color import RESET, RED


class Search:
    def __init__(self):
        self.retriever = bm25s.BM25.load("data/processed/bm25_index")
        try:
            with open("data/processed/chunks", "r") as file:
                self.chunks = json.load(file)
        except (FileNotFoundError, PermissionError):
            raise ValueError

    def search(self, question: UnansweredQuestion,
               k: int) -> MinimalSearchResults:
        tokenized_prompt = bm25s.tokenize(question.question)

        idxs, _ = self.retriever.retrieve(tokenized_prompt, k=k)
        sources = [MinimalSource(**self.chunks[i]) for i in idxs[0]]

        return (MinimalSearchResults(
            **question.model_dump(),
            retrieved_sources=sources
        ))

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str):
        try:
            with open(dataset_path, "r") as file:
                rag_dataset = RagDataset(
                    **json.load(file)
                )
        except (FileNotFoundError, PermissionError):
            print(f"{dataset_path + RED} does not exists or cannnot "
                  f"be written {RESET}")
        out = []
        for data in rag_dataset.rag_questions:
            out.append(self.search(data, k))
        return (out)
