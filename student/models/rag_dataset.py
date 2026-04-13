from pydantic import BaseModel
from .answered_question import AnsweredQuestion
from .unanswered_question import UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
