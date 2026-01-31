from typing import Optional

from pydantic import BaseModel, Field


class Tool(BaseModel):
    id: str = Field(..., description="Unique tool id")
    name: str
    description: str
    target_function: str
    signature: Optional[str] = None
    output_format: Optional[str] = None
    purpose: Optional[str] = None
    tags: list[str]
