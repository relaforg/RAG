from fire import Fire
from student.indexing import Index
from student.searching import Search


def index(max_chunck_size: int = 2000):
    Index(max_chunck_size).index()


def search(prompt: str, k: int = 10):
    answer = Search().search(prompt, k)
    print(answer)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None:
    answer = Search().search_dataset(dataset_path, k, save_directory)
    print(answer)


if (__name__ == "__main__"):
    Fire()
