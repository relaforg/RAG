from fire import Fire
from student.indexing import Index
from student.searching import Search
from student.models import UnansweredQuestion
from student.answering import Answer


def index(max_chunck_size: int = 2000) -> None:
    Index(max_chunck_size).index()


def search(prompt: str, k: int = 10) -> None:
    question = UnansweredQuestion(question=prompt)
    answer = Search().search(question, k)
    print(answer)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None:
    answer = Search().search_dataset(dataset_path, k, save_directory)
    print(answer)


def answer(prompt: str, k: int = 10) -> None:
    question = UnansweredQuestion(question=prompt)
    answer = Answer().answer(question, k)
    print(question.question)
    print("\nAnswer:")
    print(answer.answer)


if (__name__ == "__main__"):
    Fire()
