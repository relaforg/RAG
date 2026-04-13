from pydantic import BaseModel
from .minimal_answer import MinimalAnswer


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int
