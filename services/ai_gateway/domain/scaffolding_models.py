"""Pydantic models for the agent scaffolding graph."""

from operator import add
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class AgentCreationTask(BaseModel):
    """Task definition for agent generation."""

    id: str = Field(description="Task ID")
    name: str = Field(description="Agent name")
    purpose: str = Field(description="Agent's purpose")
    tags: list[str] = Field(default_factory=list, description="Agent tags")
    is_manager: bool = Field(default=False, description="Whether the agent is the manager")
    tools: list[str] = Field(default_factory=list, description="Tools available for the agent")
    system_prompt: str = Field(default="", description="System prompt of the agent")


class AgentCreationTaskResult(BaseModel):
    """Result of executing a task."""

    task_id: str = Field(description="ID of the task")
    name: str = Field(description="Task name")
    output: str = Field(description="Task output")
    passed_qa: bool = Field(default=False, description="QA review status")
    feedback: str = Field(default="", description="QA feedback if failed")
    retry_count: int = Field(default=0, description="Number of retries")


class PlanTask(BaseModel):
    """Plan task with QA tracking for the overall plan."""

    plan: list[AgentCreationTask] = Field(default_factory=list, description="List of agent tasks in the plan")
    plan_json: str = Field(default="", description="JSON representation of the plan")
    passed_qa: bool = Field(default=False, description="Whether the plan passed QA review")
    feedback: str = Field(default="", description="QA feedback if plan failed review")
    retry_count: int = Field(default=0, description="Number of times the plan has been retried")


class GraphState(BaseModel):
    """Graph state with runtime validation."""

    user_request: str = Field(default="", description="Original user request")

    # Plan tracking with QA and retry logic
    plan_task: PlanTask | None = Field(default=None, description="Current plan with QA tracking")

    # Task results (all results including retries for tracking)
    task_results: Annotated[list[AgentCreationTaskResult], add] = Field(default_factory=list, description="All task results (for retry tracking)")

    final_output: str = Field(default="", description="Final formatted output")

    class Config:
        arbitrary_types_allowed = True


class PlannerAgentSpec(BaseModel):
    """Structured planner output for a single agent spec."""

    id: str = Field(description="Unique agent id")
    name: str = Field(description="Agent name")
    purpose: str = Field(description="Agent purpose")
    is_manager: bool = Field(description="Whether this agent is the manager")
    tags: list[str] = Field(default_factory=list, description="Agent tags")


class PlannerOutput(BaseModel):
    """Structured planner output for the full plan."""

    user_request: str = Field(description="Original user request")
    agents: list[PlannerAgentSpec] = Field(description="Planned agents")


class PlanQAOutput(BaseModel):
    """Structured output for plan QA."""

    overall_decision: Literal["approved", "needs_revision"]
    feedback: str | dict[str, Any] | list[Any]


class AgentGeneratorOutput(BaseModel):
    """Structured output for agent_generator."""

    agent_id: str
    agent_name: str
    tools: list[str]
    system_prompt: str
    sub_agents: list[str]


class AgentQAOutput(BaseModel):
    """Structured output for agent QA."""

    overall_decision: Literal["approved", "needs_revision"]
    feedback: str | dict[str, Any] | list[Any]
