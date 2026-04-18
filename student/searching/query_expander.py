from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryExpander:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:0.6b")
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "Generate 3 search query variants for: {question}"
             "\nReturn one per line, no numbering.")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def expand(self, prompt: str) -> list[str]:
        return (self.chain.invoke({"question": prompt}).splitlines())
