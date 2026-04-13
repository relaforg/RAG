from pydantic import BaseModel
from .minimal_search_results import MinimalSearchResults


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
