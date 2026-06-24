"""
===============================================================================
LLM & EMBEDDING INITIALIZATION NOTES
===============================================================================

This project uses LangChain for both chat models and embedding models.

===============================================================================
CHAT MODELS
===============================================================================

Purpose
-------
Chat models generate text responses.

Example:

    response = llm.invoke("Explain vector databases")


Ways to Initialize Chat Models
==============================

1. Provider-Specific Client

Examples:

    ChatOpenAI(...)
    ChatOpenRouter(...)
    ChatAnthropic(...)

Pros:
    - Access to provider-specific features.
    - Simple setup.

Cons:
    - Application becomes tied to a specific provider.
    - Switching providers may require code changes.


2. Generic Factory

Example:

    init_chat_model(
        model=MODEL_NAME,
        model_provider=PROVIDER_NAME
    )

Pros:
    - Provider can be changed through configuration.
    - Business logic remains provider-independent.
    - Easier provider migration.

Cons:
    - Some provider-specific features may not be exposed.


Project Decision
================

We use:

    init_chat_model()

Reason:

    - Provider may change in the future.
    - Configuration controls provider selection.
    - Keeps application code provider-independent.
    - Easier maintenance and migration.

Current implementation:

    @lru_cache(maxsize=1)
    def get_llm():
        return init_chat_model(
            model=LLM_MODEL,
            model_provider=LLM_MODEL_PROVIDER,
            temperature=0.2,
        )


===============================================================================
EMBEDDING MODELS
===============================================================================

Purpose
-------

Embedding models convert text into vectors.

Example:

    "best gaming laptop"

becomes:

    [0.12, -0.45, 0.83, ...]

Used for:

    - RAG
    - Vector Search
    - Similarity Search
    - Recommendations


Ways to Initialize Embedding Models
===================================

1. LangChain Embedding Client

Examples:

    OpenAIEmbeddings(...)
    HuggingFaceEmbeddings(...)
    VoyageAIEmbeddings(...)

Pros:

    - Native LangChain integration.
    - Less custom code.
    - Works seamlessly with retrievers and vector stores.

Cons:

    - Depends on LangChain abstractions.


2. Custom Embedding Service

Example:

    class EmbeddingService:
        def get_embedding(self, text):
            ...

Pros:

    - Full control over implementation.
    - Easy customization and debugging.

Cons:

    - More code to maintain.


Can OpenAIEmbeddings Be Used With OpenRouter?
=============================================

Yes.

Example:

    OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

Important:

    OpenAIEmbeddings does NOT mean requests go to OpenAI.

    The class implements the OpenAI-compatible Embeddings API.

    Requests are sent to whichever endpoint is configured
    through the base_url parameter.

Example flow:

    Application
        ->
    OpenAIEmbeddings
        ->
    OpenRouter API
        ->
    OpenRouter Embedding Model


Why Not Use ChatOpenRouter For Embeddings?
==========================================

Chat models and embedding models serve different purposes.

Chat Models:
    Generate text responses.

Embedding Models:
    Generate vector representations.

Incorrect:

    ChatOpenRouter(...)

Correct:

    OpenAIEmbeddings(...)
    HuggingFaceEmbeddings(...)
    Custom EmbeddingService(...)


Project Decision
================

We use:

    OpenAIEmbeddings()

Reason:

    - Project already uses LangChain extensively.
    - Less custom code.
    - Easy integration with retrievers and vector stores.
    - OpenRouter supports OpenAI-compatible embedding APIs.
    - Simpler than maintaining a custom embedding layer.

Current implementation:

    @lru_cache(maxsize=1)
    def get_embedding():
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )


===============================================================================
CACHING
===============================================================================

Both chat and embedding clients are cached using:

    @lru_cache(maxsize=1)

Reason:

    - Create the client only once.
    - Reuse the same instance throughout the application.
    - Reduce initialization overhead.
    - Improve performance.

Effectively behaves like a singleton within the current process.

===============================================================================
"""