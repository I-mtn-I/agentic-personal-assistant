import asyncio
import random
import sys
import traceback
import uuid

from ai_gateway.config.settings import APP_CONFIG
from ai_gateway.domain import AgentConfigRepository, AgentFactory, ToolFactory
from ai_gateway.utils.streaming import build_streaming_session


def generate_hilton_guest_password():
    num = random.randint(0, 999)
    return f"Hilton!Guest{num:03d}"


def main():
    # -----------------------------------------------------------------------------
    # Option A: Multi Agent Pattern with supervisor agent using sub-agent as tool
    # -----------------------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Option B: Same Multi Agent Pattern using pre-configured agents
    # ----------------------------------------------------------------
    # default_agents = AgentFactory.generate_default_agents()
    # supervisor_agent = default_agents.reception_supervisor.extend_agent_with_subagent(
    #     default_agents.it,
    #     "Useful to get it related answers such as wifi password",
    # )
    # response = asyncio.run(supervisor_agent.ask("I'm at room 451, What is the Wi-Fi password?"))
    # print(response)

    # Temp tool test for web search:
    # default_agents = AgentFactory.generate_default_agents()
    # response = asyncio.run(
    #     default_agents.researcher.ask(
    #         "What was the last statement from Donald Trump? indicate the source and date time."
    #     )
    # )
    # print(response)

    # -----------------------------------------------------------------------------
    # Option C: Use Scaffolding Technique to Spawn generated agents from DB
    # -----------------------------------------------------------------------------
    repo = AgentConfigRepository.from_app_config(APP_CONFIG)

    team_id_raw = "73b13f09-85f9-4a8b-b232-eed488e19084"
    if team_id_raw:
        team_record = repo.get_team_config(team_id=uuid.UUID(team_id_raw))
    else:
        team_record = repo.get_latest_team_config()

    if not team_record:
        raise ValueError("No team configuration found in database.")

    manager_record = repo.get_agent_config_by_id(agent_config_id=team_record.manager_agent_id)
    if not manager_record:
        raise ValueError("Manager agent config not found for the selected team.")

    stream_response = True
    manager_stream = build_streaming_session(manager_record.agent_name, is_subagent=False) if stream_response else None
    streaming_callbacks = manager_stream.callbacks if manager_stream else None

    manager_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in manager_record.tools]
    manager_agent = AgentFactory.build_agent(
        manager_record.agent_name,
        manager_record.system_prompt,
        manager_tools,
        streaming=stream_response,
        callbacks=streaming_callbacks,
    )

    for agent_config_id in team_record.agent_config_ids:
        if agent_config_id == team_record.manager_agent_id:
            continue
        sub_record = repo.get_agent_config_by_id(agent_config_id=agent_config_id)
        if not sub_record:
            raise ValueError(f"Sub-agent config not found: {agent_config_id}")

        sub_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in sub_record.tools]
        sub_stream = build_streaming_session(sub_record.agent_name, is_subagent=True) if stream_response else None
        sub_callbacks = sub_stream.callbacks if sub_stream else None
        sub_agent = AgentFactory.build_agent(
            sub_record.agent_name,
            sub_record.system_prompt,
            sub_tools,
            streaming=stream_response,
            callbacks=sub_callbacks,
        )
        manager_agent = manager_agent.extend_agent_with_subagent(
            sub_agent,
            f"Sub-agent for: {sub_record.purpose}",
            streaming=stream_response,
            callbacks=streaming_callbacks,
        )

    user_prompt = "What was the last statement from Donald Trump in 2026? indicate the source and date time."
    response = asyncio.run(manager_agent.ask(user_prompt))
    if not (stream_response and manager_stream and not manager_stream.should_print_final()):
        print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Unhandled exception in ai_gateway.main", file=sys.stderr)
        traceback.print_exc()
        raise
