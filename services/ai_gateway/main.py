import random

from ai_gateway.domain.agent_factory import create_lc_agent
from ai_gateway.domain.tool_factory import create_lc_tool


def generate_hilton_guest_password():
    num = random.randint(0, 999)
    return f"Hilton!Guest{num:03d}"


def main():
    tool = create_lc_tool(
        target=generate_hilton_guest_password, description="Wifi Password Generator"
    )
    agent = create_lc_agent("helpdesk", overrides={"tools": [tool]})
    question = "What is the Wi‑Fi password for guests?"
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": question}],
        }
    )

    agent_response = result["messages"][-1].content
    print("Agent response:")
    print(agent_response)


if __name__ == "__main__":
    main()
