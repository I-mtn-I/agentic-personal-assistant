import asyncio
import random

from ai_gateway.domain import AgentFactory


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

    # TEST AREA:
    default_agents = AgentFactory.generate_default_agents()
    response = asyncio.run(
        default_agents.researcher.ask(
            "What are the recent developments in Asia that are related to climate change?"
        )
    )
    print(response)


if __name__ == "__main__":
    main()
