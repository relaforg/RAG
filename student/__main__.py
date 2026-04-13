from fire import Fire
from student.indexing import Index


def index(max_chunck_size: int = 2000):
    Index(max_chunck_size).index()


if (__name__ == "__main__"):
    Fire()
