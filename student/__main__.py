from fire import Fire
from student.indexing import Index
from student.searching import Search
from student.models import UnansweredQuestion
from student.answering import Answer
from student.evaluate import Evaluate


DEFAULT_ANSWER_DIR = "data/output/search_results_and_answer"


def index(max_chunck_size: int = 2000) -> None:
    """Index the vLLM repository and save the BM25 index to disk.

    Args:
        max_chunck_size: Maximum number of characters per chunk.
    """
    Index(max_chunck_size).index()


def search(prompt: str, k: int = 10) -> None:
    """Search the index for a single query and print results.

    Args:
        prompt: The search query.
        k: Number of results to retrieve.
    """
    question = UnansweredQuestion(question=prompt)
    answer = Search().search(question, k)
    print(answer)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None:
    """Search all questions in a dataset and save results to disk.

    Args:
        dataset_path: Path to the dataset JSON file.
        k: Number of results to retrieve per question.
        save_directory: Directory where results will be saved.
    """
    Search().search_dataset(dataset_path, k, save_directory)


def answer(prompt: str, k: int = 10) -> None:
    """Answer a single question using retrieved context and print the result.

    Args:
        prompt: The question to answer.
        k: Number of sources to retrieve.
    """
    question = UnansweredQuestion(question=prompt)
    answer = Answer().answer(question, k)
    print(question.question)
    print("\nAnswer:")
    print(answer.answer)


def answer_dataset(student_search_result_path: str,
                   save_directory: str = DEFAULT_ANSWER_DIR) -> None:
    """Generate answers for all questions in a search results file.

    Args:
        student_search_result_path: Path to the search results JSON.
        save_directory: Directory where answered results will be saved.
    """
    Answer().answer_dataset(student_search_result_path, save_directory)


def evaluate(student_answer_path: str, dataset_path: str) -> None:
    """Evaluate search results against ground truth and print recall@k scores.

    Args:
        student_answer_path: Path to the student search results JSON.
        dataset_path: Path to the answered questions dataset JSON.
    """
    Evaluate().evaluate(student_answer_path, dataset_path)


if (__name__ == "__main__"):
    Fire()
