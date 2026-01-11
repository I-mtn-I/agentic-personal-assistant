import asyncio
import random
import uuid

from ai_gateway.config import APP_CONFIG
from ai_gateway.domain.agent_factory import AgentFactory, ToolFactory
from ai_gateway.domain.agent_persistence import AgentConfigRepository


def generate_hilton_guest_password():
    num = random.randint(0, 999)
    return f"Hilton!Guest{num:03d}"


def main():
    # Option A: Multi Agent Pattern with supervisor agent using sub-agent as tool
    ## 1) Create a tool using Python Callable
    # tool_wifi_pass = create_tool_from_callable(
    #     target=generate_hilton_guest_password, description="Wifi Password Generator"
    # )

    # ## 2) Create an Agent and provide the tool to it
    # it_agent = Agent(
    #     "it",
    #     "You are an IT specialist of the Hilton 'H'otel supporting front desk.",
    #     tools=[tool_wifi_pass],
    # ).create_agent()

    # ## 3) Create a supervisor Agent that uses sub-agent as tool
    # supervisor_agent = Agent(
    #     "reception_supervisor",
    #     "You are the supervisor at the frontdesk of Hotel Hilton.\
    #       You are responsible of all reception operations.",
    #     tools=[it_agent.get_agent_as_tool()],
    # ).create_agent()

    # ## 4) invoke the supervisor agent
    # response = asyncio.run(supervisor_agent.ask("I'm at room 451, What is the Wi-Fi password?"))
    # print(response)

    # Option B: Same Multi Agent Pattern using pre-configured agents
    # default_agents = AgentFactory.generate_default_agents()
    # supervisor_agent = default_agents.reception_supervisor.extend_agent_with_subagent(
    #     default_agents.it,
    #     "Useful to get it related answers such as wifi password",
    # )
    # response = asyncio.run(supervisor_agent.ask("I'm at room 451, What is the Wi-Fi password?"))
    # print(response)

    # default_agents = AgentFactory.generate_default_agents()
    # response = asyncio.run(
    #     default_agents.researcher.ask(
    #         "What are the recent developments in Asia that are related to climate change?"
    #     )
    # )
    # print(response)

    repo = AgentConfigRepository.from_app_config(APP_CONFIG)

    team_id_raw = "0fbb40a9-9ea1-4db3-bf9b-f8c66b30ac0e"
    if team_id_raw:
        team_record = repo.get_team_config(team_id=uuid.UUID(team_id_raw))
    else:
        team_record = repo.get_latest_team_config()

    if not team_record:
        raise ValueError("No team configuration found in database.")

    manager_record = repo.get_agent_config_by_id(agent_config_id=team_record.manager_agent_id)
    if not manager_record:
        raise ValueError("Manager agent config not found for the selected team.")

    manager_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in manager_record.tools]
    manager_agent = AgentFactory.build_agent(
        manager_record.agent_name, manager_record.system_prompt, manager_tools
    )

    for agent_config_id in team_record.agent_config_ids:
        if agent_config_id == team_record.manager_agent_id:
            continue
        sub_record = repo.get_agent_config_by_id(agent_config_id=agent_config_id)
        if not sub_record:
            raise ValueError(f"Sub-agent config not found: {agent_config_id}")

        sub_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in sub_record.tools]
        sub_agent = AgentFactory.build_agent(
            sub_record.agent_name, sub_record.system_prompt, sub_tools
        )
        manager_agent = manager_agent.extend_agent_with_subagent(
            sub_agent,
            f"Sub-agent for: {sub_record.purpose}",
        )

    user_prompt = "What are the recent news about Elon Musk? (Jan 2026)"
    response = asyncio.run(manager_agent.ask(user_prompt))
    print(response)


if __name__ == "__main__":
    main()
