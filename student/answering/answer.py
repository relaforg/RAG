from student.models import UnansweredQuestion, AnsweredQuestion
from student.searching import Search

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class Answer:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:0.6b", think=False)
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm

    def answer(self, question: UnansweredQuestion, k: int) -> AnsweredQuestion:
        sources = Search().search(question, k)
        context = "\n\n".join(s.text for s in sources.retrieved_sources)
        answer = self.chain.invoke({
            "context": context,
            "question": question.question
        })
        return (AnsweredQuestion(
            **question.model_dump(),
            sources=sources.retrieved_sources,
            answer=answer.content
        ))
