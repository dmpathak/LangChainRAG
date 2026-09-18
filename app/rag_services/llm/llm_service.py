import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.rag_services.llm.AI_models import get_llm_model
from app.rag_services.llm.prompt import RAG_USER_PROMPT, SYSTEM_PROMPT


class LLMService:
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", RAG_USER_PROMPT),
            ])
            | get_llm_model()
            | StrOutputParser()
        )

    def get_response(self, query, context, max_retries=5, initial_delay=2.0):
        for attempt in range(max_retries):
            try:
                return self.chain.invoke({"context": context, "question": query})
            except Exception as error:
                message = str(error).lower()
                is_rate_limit = "429" in message or "rate limit" in message
                if not is_rate_limit or attempt == max_retries - 1:
                    raise
                time.sleep(initial_delay * (2 ** attempt))
