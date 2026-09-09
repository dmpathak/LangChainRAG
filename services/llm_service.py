import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from services.prompt import RAG_USER_PROMPT, SYSTEM_PROMPT
from services.AI_models import get_llm_model


class LLMService:
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", RAG_USER_PROMPT),
                ]
            )
            | get_llm_model()
            | StrOutputParser()
        )

    def get_response(
        self,
        query: str,
        context: str,
        max_retries: int = 5,
        initial_delay: float = 2.0,
    ):
        for attempt in range(max_retries):
            try:
                return self.chain.invoke({"context": context, "question": query})
            except Exception as error:
                error_name = type(error).__name__
                error_message = str(error).lower()
                is_rate_limit = (
                    "TooManyRequests" in error_name
                    or "429" in error_message
                    or "rate limit" in error_message
                )

                if not is_rate_limit or attempt == max_retries - 1:
                    raise

                delay = initial_delay * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
