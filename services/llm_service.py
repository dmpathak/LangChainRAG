import time
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
            max_retries: int = 5,
            initial_delay: float = 2.0,
    ):
        for attempt in range(max_retries):
            try:
                return self.chain.invoke(
                    {
                        "context": context,
                        "question": query,
                    }
                )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                if "TooManyRequests" in error_type or "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        print(f"Rate limited for choosen model. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                raise
