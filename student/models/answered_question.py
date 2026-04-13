from .unanswered_question import UnansweredQuestion
from .minimal_source import MinimalSource


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
