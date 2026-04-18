import json
import bm25s
from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from pathlib import Path
import chromadb

from student.models import MinimalSource
from student.color import GREEN, RESET, BRIGHT_BLACK


class Index:
    """Builds a BM25 searchable index from the vLLM repository."""

    def __init__(self, max_chunk_size: int):
        """Initialize splitters and corpus storage.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
        """
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
        self.client = chromadb.PersistentClient(path="data/chroma")
        self.client.delete_collection("chunks")
        self.collection = self.client.get_or_create_collection("chunks")

    def _split_mardowns(self, out: list[MinimalSource], chroma: bool,
                        chunk_id: int = 0) -> int:
        """Chunk all Markdown files and append sources to out.

        Args:
            out: List to append MinimalSource chunks to.
            chunk_id: Starting Chroma ID offset.

        Returns:
            Next available chunk_id.
        """
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
                if (chroma):
                    self.collection.add(
                        documents=[chunck],
                        ids=[str(chunk_id)]
                    )
                    chunk_id += 1
        return chunk_id

    def _split_pythons(self, out: list[MinimalSource], chroma: bool,
                       chunk_id: int = 0) -> None:
        """Chunk all Python files and append sources to out.

        Args:
            out: List to append MinimalSource chunks to.
            chunk_id: Starting Chroma ID offset.
        """
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
                if (chroma):
                    self.collection.add(
                        documents=[chunck],
                        ids=[str(chunk_id)]
                    )
                    chunk_id += 1

    def index(self, chroma: bool) -> None:
        """Index all repository files and save the BM25 index to disk."""
        out: list[MinimalSource] = []
        next_id = self._split_mardowns(out, chroma)
        self._split_pythons(out, chroma, chunk_id=next_id)
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
