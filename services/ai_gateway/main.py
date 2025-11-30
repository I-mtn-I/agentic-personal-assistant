import random

from ai_gateway.domain import Tool
from ai_gateway.domain.agent_factory import AgentFactory


def generate_hilton_guest_password():
    num = random.randint(0, 999)
    return f"Hilton!Guest{num:03d}"


def main():
    # Option A: Multi Agent Pattern with supervisor agent using sub-agent as tool
    ## 1) Create a tool using Python Callable
    tool_wifi_pass = Tool(
        description="Wifi Password Generator", target=generate_hilton_guest_password
    ).create_tool()

    # ## 2) Create an Agent and provide the tool to it
    # it_agent = Agent(
    #     "it",
    #     "You are an IT specialist of the Hilton Hotel supporting front desk.",
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
    # response = supervisor_agent.ask("I'm at room 451, What is the Wi-Fi password?")
    # print(response)

    # Option B: Same Multi Agent Pattern using pre-configured agents
    default_agents = AgentFactory.get_default_agents()
    agent_it = default_agents.it
    agent_it.bind_tools(tool_wifi_pass)

    supervisor_agent = default_agents.reception_supervisor
    supervisor_agent.bind_tools(agent_it.get_agent_as_tool())
    response = supervisor_agent.ask("I'm at room 451, What is the Wi-Fi password?")
    print(response)


if __name__ == "__main__":
    main()
