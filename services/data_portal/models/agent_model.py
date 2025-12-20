from typing import Optional

from pydantic import BaseModel, Field


class Agent(BaseModel):
    id: str = Field(..., description="Unique agent id")
    name: str
    full_prompt: str
    role_description: str
    goal_description: str
    background_information: str
    output_format: Optional[str] = None
    tools_list: Optional[list[str]] = None
    tags: list[str]
