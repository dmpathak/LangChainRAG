"""
This File Only used for --> @app.post("/invoke") endpoint.
Which is demo for direct llm call.

******* Not Related To RAG & lagchain flow *******
"""

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")
