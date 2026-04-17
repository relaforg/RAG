from fire import Fire
from student.indexing import Index
from student.searching import Search
from student.models import UnansweredQuestion
from student.answering import Answer
from student.evaluate import Evaluate


def index(max_chunck_size: int = 2000) -> None:
    Index(max_chunck_size).index()


def search(prompt: str, k: int = 10) -> None:
    question = UnansweredQuestion(question=prompt)
    answer = Search().search(question, k)
    print(answer)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None:
    Search().search_dataset(dataset_path, k, save_directory)


def answer(prompt: str, k: int = 10) -> None:
    question = UnansweredQuestion(question=prompt)
    answer = Answer().answer(question, k)
    print(question.question)
    print("\nAnswer:")
    print(answer.answer)


def answer_dataset(student_search_result_path: str,
                   save_directory: str = "data/output/search_results_and_answer") -> None:
    Answer().answer_dataset(student_search_result_path, save_directory)


def evaluate(student_answer_path: str, dataset_path: str) -> None:
    Evaluate().evaluate(student_answer_path, dataset_path)


if (__name__ == "__main__"):
    Fire()
