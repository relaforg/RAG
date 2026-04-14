import json
import bm25s
from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from pathlib import Path

from student.models import MinimalSource
from student.color import GREEN, RESET, BRIGHT_BLACK


class Index:
    def __init__(self, max_chunk_size: int):
        self.dataset_path = "data/raw/vllm-0.10.1"
        self.max_chunk_size = max_chunk_size
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=0
        )
        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=0
        )
        self.corpus: list[str] = []

    def _split_mardowns(self, out: list[MinimalSource]):
        md_files = Path(self.dataset_path).rglob("*.md")
        for md in tqdm(md_files, desc="Indexing .md files", unit=" file"):
            idx = 0
            for chunck in self.markdown_splitter.split_text(md.read_text()):
                self.corpus.append(chunck)
                out.append(MinimalSource(
                    file_path=str(md),
                    text=chunck,
                    first_character_index=idx,
                    last_character_index=idx + len(chunck) - 1
                ))
                idx += len(chunck)

    def _split_pythons(self, out: list[MinimalSource]):
        py_files = Path(self.dataset_path).rglob("*.py")
        for py in tqdm(py_files, desc="Indexing .py files", unit=" file"):
            idx = 0
            for chunck in self.python_splitter.split_text(py.read_text()):
                self.corpus.append(chunck)
                out.append(MinimalSource(
                    file_path=str(py),
                    text=chunck,
                    first_character_index=idx,
                    last_character_index=idx + len(chunck) - 1
                ))
                idx += len(chunck)

    def index(self) -> None:
        out: list[MinimalSource] = []
        self._split_mardowns(out)
        self._split_pythons(out)
        try:
            chunks_file_path = Path() / "data" / "processed"
            chunks_file_path.mkdir(parents=True, exist_ok=True)
            with open(chunks_file_path / "chunks", "w") as file:
                file.write(
                    json.dumps(
                        [chunk.model_dump() for chunk in out],
                        indent=4
                    )
                )
            print(f"{GREEN}Ingestion complete! Indices saved"
                  f" under data/processed/{BRIGHT_BLACK} ({len(out)} chunks)"
                  + RESET)
        except (PermissionError, FileNotFoundError) as e:
            print(e)
        tokenized = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        retriever.save("data/processed/bm25_index")
