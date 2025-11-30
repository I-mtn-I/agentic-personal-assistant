from typing import Any, Dict, Optional

from langchain.agents import create_agent as lc_agent
from langchain.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ai_gateway.config import APP_CONFIG


class Agent:
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: Optional[list[Any]] = None,
        middleware: Optional[list[Any]] = None,  # TODO: check middleware config in langchain
        state_schema: Optional[Any] = None,
        context_schema: Optional[Dict[str, Any]] = None,
        checkpointer: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.tools = tools if tools else []
        self.middleware = middleware if middleware else ()
        self.state_schema = state_schema
        self.context_schema = context_schema if context_schema else {}
        self.checkpointer = checkpointer
        self.agent: Optional[CompiledStateGraph[Any, None, Any, Any]] = None

    def create_agent(self) -> "Agent":
        _model = ChatOllama(
            model=getattr(APP_CONFIG, "LLM_MODEL", ""),
            base_url=getattr(APP_CONFIG, "LLM_HOST", None),
        )

        agent_kwargs: Dict[str, Any] = {
            "model": _model,
            "name": self.name,
            "tools": self.tools,
            "system_prompt": self.prompt,
            "middleware": self.middleware,
            "state_schema": self.state_schema,
            "context_schema": self.context_schema,
            "checkpointer": self.checkpointer,
        }
        self.agent = lc_agent(**agent_kwargs)

        return self

    def ask(self, query: str) -> str:
        """
        Send a user query to the built agent and return the final response text.
        """
        if self.agent is None:
            raise RuntimeError(
                "Agent not initialised - call ``create_agent()`` before ``invoke()``."
            )

        response = self.agent.invoke(  # pyright: ignore
            {
                "messages": [{"role": "user", "content": query}],
            }
        )
        # ``response`` follows the LangGraph schema; the last message holds the answer.
        return response["messages"][-1].content

    def get_agent_as_tool(self) -> BaseTool:
        """
        Use the agent as a tool for another agent, making this a subagent.
        """
        if self.agent is None:
            raise ValueError("No agent found, did you forget to call create_agent()?")

        class AgentInput(BaseModel):
            """Input for the agent tool."""

            query: str = Field(description="The query to pass to the agent")

        @tool(args_schema=AgentInput)
        def agent_tool(query: str) -> str:
            """Call the agent to handle specialized tasks."""
            if self.agent:
                response = self.agent.invoke({"messages": [{"role": "user", "content": query}]})  # pyright: ignore
                return response["messages"][-1].content
            else:
                raise ValueError(
                    "No agent found, did you forget to create it? (Agent.create_agent)"
                )

        agent_tool.name = f"{self.name}_tool"
        agent_tool.description = f"Delegate tasks to the {self.name} agent. {self.prompt}"

        return agent_tool
