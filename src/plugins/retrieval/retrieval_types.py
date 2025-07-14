from pydantic import BaseModel, Field
from typing import List, Optional

class VectorIndexRetrievalResult(BaseModel):
    """
    Result of a vector index search operation.
    """
    result: str = Field(..., description="Result of the vector index search")
    error: Optional[str] = Field(None, description="Error message, if any")

class MultimodalVectorIndexRetrievalResult(BaseModel):
    """
    Result of a multimodal (text + images) vector index search.
    """
    texts: List[str] = Field(..., description="List of retrieved text excerpts")
    images: List[List[str]] = Field(
        ...,
        description="List of lists of image URLs; each inner list corresponds to images related to a document"
    )
    captions: List[List[str]] = Field(
        ...,
        description="List of lists of captions; each inner list corresponds to captions related to a document"
    )
    error: Optional[str] = Field(None, description="Error message, if any")

class DataPointsResult(BaseModel):
    """
    Result containing data points extracted from a chat log.
    """
    data_points: List[str] = Field(..., description="List of data points extracted from the chat log")