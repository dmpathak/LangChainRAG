from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from services.prompt import SYSTEM_PROMPT
from services.AI_models import get_llm_model


class LLMService:

    def __init__(self):
        self.chain = (
                ChatPromptTemplate.from_messages(
                    [
                        ("system", SYSTEM_PROMPT),
                        ("human", """Context: {context}  Question: {question}"""),
                    ]
                )
                | get_llm_model()
                | StrOutputParser()
        )

    def get_response(
            self,
            query: str,
            context: str,
    ):
        return self.chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )
