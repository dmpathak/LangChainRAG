"""Response schema for the direct LLM demo endpoint."""

from pydantic import BaseModel, Field


class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")
