"""Entry point for the agent scaffolding graph."""

import asyncio

from data_portal.helpers.colored_logger import ColoredLogger

from ai_gateway.config import APP_CONFIG
from ai_gateway.domain.agent import Agent
from ai_gateway.domain.agent_factory import AgentFactory
from ai_gateway.domain.agent_persistence import AgentConfigRepository
from ai_gateway.domain.agent_scaffolding_graph import AgentScaffoldingGraph

logger = ColoredLogger(level="DEBUG")


async def run_agent_scaffolding(
    user_request: str,
    agents: dict[str, Agent],
    persistence: AgentConfigRepository | None = None,
) -> str:
    graph = AgentScaffoldingGraph(agents=agents, persistence=persistence, logger=logger)
    return await graph.run(user_request)


async def main() -> None:
    """Main entry point."""
    logger.info("Initializing agents...")
    default_agents = AgentFactory.generate_default_agents()

    agents = {
        "planner": default_agents.planner,
        "plan_qa": default_agents.plan_qa,
        "agent_generator": default_agents.agent_generator,
        "qa": default_agents.qa,
    }

    persistence = AgentConfigRepository.from_app_config(APP_CONFIG)

    user_request = "I need a team to perform a search the web on a given subject and create a report."

    await run_agent_scaffolding(user_request, agents, persistence)


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 70)
    logger.info("DYNAMIC AGENT GENERATOR")
    logger.info("=" * 70)
    logger.info("")

    asyncio.run(main())
