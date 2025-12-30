"""Agent scaffolding graph with dynamic agent generation, QA review, and retry logic."""

import asyncio
import json
from typing import Any

from data_portal.helpers.colored_logger import ColoredLogger
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ai_gateway.config.settings import TOOLS_CONFIG
from ai_gateway.domain import (
    Agent,
    AgentCreationTask,
    AgentCreationTaskResult,
    GraphState,
    PlanTask,
)
from ai_gateway.utils import extract_json_from_response

# Initialize logger
logger = ColoredLogger(level="DEBUG")

# Configuration
MAX_RETRIES = 3  # Maximum retry attempts for both plan and agent tasks


# ============= HELPER FUNCTIONS =============


def get_available_tools_info() -> str:
    available_tools = TOOLS_CONFIG.list_names()
    return "\n".join(
        [f"  - {name}: {TOOLS_CONFIG.raw(name).description}" for name in available_tools]
    )


def find_task_by_id(plan: list[AgentCreationTask], task_id: str) -> AgentCreationTask | None:
    for task in plan:
        if task.id == task_id:
            return task
    return None


def find_previous_result(
    task_results: list[AgentCreationTaskResult], task_id: str
) -> tuple[AgentCreationTaskResult | None, str, int]:
    """
    Find previous result for a task.

    Returns:
        Tuple of (previous_result, feedback, retry_count)
    """
    for existing in task_results:
        if existing.task_id == task_id:
            return existing, existing.feedback, existing.retry_count + 1
    return None, "", 0


def get_latest_results(
    task_results: list[AgentCreationTaskResult],
    pending_qa_only: bool = False,
) -> dict[str, AgentCreationTaskResult]:
    """
    Get the latest result for each task_id.

    Priority:
    1. If a result has passed QA, always prefer it (regardless of retry_count)
    2. Otherwise, prefer the result with the highest retry_count

    Args:
        task_results: List of all task results
        pending_qa_only: If True, only return results that haven't passed QA yet

    Returns:
        Dictionary mapping task_id to its latest TaskResult
    """
    latest: dict[str, AgentCreationTaskResult] = {}
    for result in task_results:
        task_id = result.task_id

        if task_id not in latest:
            latest[task_id] = result
        else:
            current = latest[task_id]

            # If current has passed QA, don't replace it
            if current.passed_qa:
                continue

            # If this result passed QA, replace current (even if retry_count is lower)
            if result.passed_qa:
                latest[task_id] = result
            # Both haven't passed, prefer higher retry_count
            elif result.retry_count > current.retry_count:
                latest[task_id] = result

    # Filter to only pending QA if requested
    if pending_qa_only:
        return {k: v for k, v in latest.items() if not v.passed_qa}

    return latest


# ============= GRAPH NODES =============


async def planner_node(state: GraphState, agents: dict[str, Agent]) -> dict[str, Any]:
    """Planner generates dynamic task configs based on user request."""
    # Check if this is a retry - increment retry count if retrying
    if state.plan_task and not state.plan_task.passed_qa:
        retry_count = state.plan_task.retry_count + 1
        feedback = state.plan_task.feedback
    else:
        retry_count = 0
        feedback = ""

    if retry_count > 0:
        logger.info(f"RE-PLANNING (Attempt #{retry_count})")
        logger.warn(f"Previous Plan QA feedback: {feedback}")
    else:
        logger.info("Planner agent running...")

    planner_agent = agents["planner"]
    tools_info = get_available_tools_info()

    # Create planner prompt with user request and available tools
    planner_query = f"""User Request: {state.user_request}
        Analyze the user request and create a plan for a multi-agent architecture.

        Available Tools:
        {tools_info}

        {f"Previous Plan QA Feedback: {feedback}" if feedback else ""}
    """

    # Invoke planner
    response = await planner_agent.ask(planner_query)
    # logger.debug(f"Planner response: {response}")

    # Parse the response
    try:
        plan_data = extract_json_from_response(response)

        # Create AgentTask objects
        tasks = [
            AgentCreationTask(
                id=agent["id"],
                name=agent["name"],
                purpose=agent["purpose"],
                tools=[],  # Will be decided by agent_generator
                system_prompt="",  # Will be created by agent_generator
            )
            for agent in plan_data.get("agents", [])
        ]

        logger.info(f"Generated {len(tasks)} agent task(s).")
        for task in tasks:
            current_task = task.model_dump()
            logger.debug(f"\t- Task ID: {current_task['id']}:")
            logger.debug(f"\t\t Name: {current_task['name']}, Purpose: {current_task['purpose']}")

        # Create PlanTask
        plan_task = PlanTask(
            plan=tasks,
            plan_json=json.dumps(plan_data),
            passed_qa=False,  # Will be set by plan_qa_node
            feedback="",
            retry_count=retry_count,
        )

        return {"plan_task": plan_task}

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse planner response: {str(e)}")
        logger.error(f"Raw response: {response}")

        # Auto-retry once on first parsing error without counting against MAX_RETRIES
        if retry_count == 0:
            logger.warn("First attempt parsing failed. Auto-retrying once...")
            # Recursive call with incremented retry to prevent infinite loop
            error_msg = (
                f"Previous attempt failed to parse JSON: {str(e)}. "
                "Please ensure output is valid JSON."
            )
            state.plan_task = PlanTask(
                plan=[],
                plan_json="",
                passed_qa=False,
                feedback=error_msg,
                retry_count=0,  # Keep at 0 for auto-retry
            )
            return await planner_node(state, agents)

        # Create empty PlanTask for error case after auto-retry
        plan_task = PlanTask(
            plan=[],
            plan_json="",
            passed_qa=False,
            feedback=f"Planner failed to produce valid JSON: {str(e)}",
            retry_count=retry_count,
        )
        return {"plan_task": plan_task}


async def plan_qa_node(state: GraphState, agents: dict[str, Agent]) -> dict[str, Any]:
    """QA reviews the overall plan generated by the planner."""
    logger.info("QA reviewing the overall plan...")

    if not state.plan_task or not state.plan_task.plan:
        logger.warn("No plan to review...")
        return {}

    plan_qa_agent = agents["plan_qa"]

    # Create QA review query with user request and generated plan
    qa_prompt = f"""
        Given Original User Request:
            {state.user_request}
        And Generated Plan:
            {state.plan_task.plan_json}

        Perform the review.
        """

    # Invoke Plan QA agent
    qa_response = await plan_qa_agent.ask(qa_prompt)
    logger.debug(f"Plan QA response:\n {qa_response}")

    # Parse QA response
    try:
        qa_data = extract_json_from_response(qa_response)
        passed = qa_data.get("overall_decision") == "approved"
        feedback = qa_data.get("feedback", "")

        if passed:
            logger.info("Plan QA PASSED")
        else:
            logger.warn(f"Plan QA FAILED: {feedback}")

        # Update PlanTask with QA results
        updated_plan_task = PlanTask(
            plan=state.plan_task.plan,
            plan_json=state.plan_task.plan_json,
            passed_qa=passed,
            feedback=feedback,
            retry_count=state.plan_task.retry_count,
        )

        return {"plan_task": updated_plan_task}

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse plan QA response: {str(e)}")

        # Mark as failed with error feedback
        updated_plan_task = PlanTask(
            plan=state.plan_task.plan,
            plan_json=state.plan_task.plan_json,
            passed_qa=False,
            feedback=f"Plan QA failed to parse response: {str(e)}",
            retry_count=state.plan_task.retry_count,
        )

        return {"plan_task": updated_plan_task}


async def agent_generator_node(state: dict[str, Any], agents: dict[str, Agent]) -> dict[str, Any]:
    """
    Agent generator creates system prompts and selects tools for each agent.
    Receives a single task via Send (always as dict).
    """
    # Extract data from dict state (sent via Send)
    plan = [AgentCreationTask(**t) if isinstance(t, dict) else t for t in state.get("plan", [])]
    task_results = [
        AgentCreationTaskResult(**r) if isinstance(r, dict) else r
        for r in state.get("task_results", [])
    ]
    user_request = state.get("user_request", "")

    task = plan[0]  # Single task sent via Send
    task_id = task.id
    task_name = task.name
    task_purpose = task.purpose

    # Check if this is a retry
    _, feedback, retry_count = find_previous_result(task_results, task_id)

    if feedback:
        logger.info(f"RE-GENERATING agent: {task_name} (ID: {task_id}, Attempt #{retry_count + 1})")
        logger.warn(f"Previous QA feedback: {feedback}")
    else:
        logger.info(f"Generating agent: {task_name} (ID: {task_id})")

    agent_generator = agents["agent_generator"]
    tools_info = get_available_tools_info()

    # Create prompt for agent_generator
    base_query = f"""Agent Name: {task_name}
Agent Purpose: {task_purpose}
User's Original Request: {user_request}

Available Tools:
{tools_info}"""

    if feedback:
        # Include feedback for retry attempts
        generator_query = f"""{base_query}

Previous QA Feedback: {feedback}

Please regenerate the agent configuration addressing the feedback above.
Return your response as a JSON object with the required structure."""
    else:
        # Initial generation
        generator_query = f"""{base_query}

Please create a complete agent configuration.
Return your response as a JSON object with the required structure."""

    # Invoke agent_generator
    response = await agent_generator.ask(generator_query)
    logger.debug(f"Agent generator response for {task_name}: {response[:200]}")

    # Parse the response
    try:
        config_data = extract_json_from_response(response)

        # Validate required fields in agent configuration
        required_fields = ["agent_id", "agent_name", "tools", "system_prompt"]
        missing_fields = [f for f in required_fields if f not in config_data]

        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate tools is a list
        if not isinstance(config_data.get("tools"), list):
            raise ValueError("Field 'tools' must be a list")

        # Create TaskResult with the generated configuration
        task_result = AgentCreationTaskResult(
            task_id=task_id,
            name=task_name,
            output=json.dumps(config_data, indent=2),
            passed_qa=False,  # Will be checked by QA
            feedback="",
            retry_count=retry_count,
        )

        logger.info(f"Agent configuration generated for {task_name}")
        logger.debug(f"Selected tools: {', '.join(config_data.get('tools', []))}")

        return {"task_results": [task_result]}

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Failed to parse agent_generator response for {task_name}: {str(e)}")

        # Create a failed result
        task_result = AgentCreationTaskResult(
            task_id=task_id,
            name=task_name,
            output=f"ERROR: Failed to generate configuration - {str(e)}",
            passed_qa=False,
            feedback=f"Agent generator failed validation: {str(e)}",
            retry_count=retry_count,
        )

        return {"task_results": [task_result]}


async def qa_node(state: dict[str, Any], agents: dict[str, Agent]) -> dict[str, Any]:
    """
    QA reviews ONE task result and validates the generated agent configuration.
    Receives a single result via Send (always as dict).
    Returns TaskResult with passed_qa=True/False and feedback.
    """
    # Extract data from dict state (sent via Send)
    task_results = [
        AgentCreationTaskResult(**r) if isinstance(r, dict) else r
        for r in state.get("task_results", [])
    ]
    plan = [AgentCreationTask(**t) if isinstance(t, dict) else t for t in state.get("plan", [])]
    user_request = state.get("user_request", "")

    result = task_results[0]  # Single result sent via Send
    task_id = result.task_id

    logger.info(
        f"QA reviewing agent: {result.name} (ID: {task_id}, Attempt #{result.retry_count + 1})"
    )

    qa_agent = agents["qa"]

    # Parse the agent configuration from output
    try:
        if result.output.startswith("ERROR:"):
            # Configuration generation failed, mark as failed
            logger.error(
                f"QA found generation error for {result.name}",
            )
            updated_result = AgentCreationTaskResult(
                task_id=task_id,
                name=result.name,
                output=result.output,
                passed_qa=False,
                feedback=result.feedback,
                retry_count=result.retry_count,
            )
            return {"task_results": [updated_result]}

        config_data = json.loads(result.output)

        # Find the original task from plan
        original_task = find_task_by_id(plan, task_id)

        # Get available tools info for QA context
        tools_info = get_available_tools_info()

        # Build context about other agents in the team (for detecting redundancy)
        other_agents_context = []
        latest_results = get_latest_results(task_results)

        for other_result in latest_results.values():
            if other_result.task_id != task_id and not other_result.output.startswith("ERROR:"):
                try:
                    other_config = json.loads(other_result.output)
                    other_tools = other_config.get("tools", [])
                    other_task = find_task_by_id(plan, other_result.task_id)
                    other_purpose = other_task.purpose if other_task else "N/A"

                    tools_str = ", ".join(other_tools) if other_tools else "None"
                    other_agents_context.append(  # pyright: ignore
                        f"- {other_result.name}: {other_purpose}\n  Tools: {tools_str}"
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

        team_context = (
            "\n".join(other_agents_context)
            if other_agents_context
            else "No other agents in the team yet."
        )

        # Create QA review prompt
        # Note: The QA agent already has comprehensive guidelines in its system_prompt
        qa_query = f"""Original User Request: {user_request}

Agent Under Review:
- Name: {result.name}
- Purpose: {original_task.purpose if original_task else "N/A"}

Other Agents in the Team:
{team_context}

Available Tools in System:
{tools_info}

Generated Configuration for {result.name}:
{json.dumps(config_data, indent=2)}

Please review this agent configuration and provide an audit report."""

        # Invoke QA agent
        qa_response = await qa_agent.ask(qa_query)
        logger.debug(f"QA response for {result.name}: {qa_response[:200]}")

        # Parse QA response
        qa_data = extract_json_from_response(qa_response)

        passed = qa_data.get("overall_decision") == "approved"
        feedback = qa_data.get("feedback", "")

        if passed:
            logger.info(f"QA PASSED for {result.name}")
        else:
            logger.warn(f"QA FAILED for {result.name}: {feedback}")

        # Create updated TaskResult
        updated_result = AgentCreationTaskResult(
            task_id=task_id,
            name=result.name,
            output=result.output,
            passed_qa=passed,
            feedback=feedback,
            retry_count=result.retry_count,
        )

        return {"task_results": [updated_result]}

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"QA failed to process result for {result.name}: {str(e)}")

        # Mark as failed with feedback
        updated_result = AgentCreationTaskResult(
            task_id=task_id,
            name=result.name,
            output=result.output,
            passed_qa=False,
            feedback=f"QA processing error: {str(e)}. Please ensure output is valid JSON.",
            retry_count=result.retry_count,
        )

        return {"task_results": [updated_result]}


def route_plan_qa(state: GraphState) -> str:
    """
    Routes after plan QA:
    - Plan failed + retries available → back to planner
    - Plan failed + no retries left → proceed to agent_generator anyway
    - Plan passed → proceed to agent_generator
    """
    if not state.plan_task:
        logger.error("No plan_task found in state")
        return "aggregator"

    if state.plan_task.passed_qa:
        logger.info("Plan QA passed! Proceeding to agent generation...")
        return "agent_generator"

    # Plan failed QA - check if we can retry
    if state.plan_task.retry_count < MAX_RETRIES:
        logger.warn(
            f"Plan QA failed (attempt #{state.plan_task.retry_count + 1}). Retrying planner..."
        )
        # Retry count will be incremented in planner_node
        return "planner"
    else:
        logger.error(
            f"Plan QA failed after {MAX_RETRIES} retries. Proceeding with current plan anyway..."
        )
        return "agent_generator"


async def qa_review_barrier(state: GraphState) -> dict[str, Any]:  # noqa: ARG001
    """
    Synchronization barrier that waits for all parallel QA nodes to complete.

    This is necessary because:
    - Multiple QA nodes run in parallel (one per agent being reviewed)
    - Without this barrier, route_qa_results would be called multiple times
    - The barrier ensures route_qa_results is called only once with all QA results
    """
    latest_results = get_latest_results(state.task_results)
    logger.debug(
        f"QA review barrier - all {len(latest_results)} QA nodes completed, proceeding to routing"
    )
    return {}


def route_qa_results(state: GraphState) -> list[Send] | str:
    """
    Routes results after ALL QA reviews complete:
    - Failed + retries available → back to agent_generator
    - Failed + no retries left → go to aggregator anyway
    - Passed → to aggregator
    """
    # Get only the latest result for each task
    latest_results = get_latest_results(state.task_results)
    results = list(latest_results.values())

    # Separate passed and failed tasks
    passed = [r for r in results if r.passed_qa]
    failed = [r for r in results if not r.passed_qa]

    logger.info(f"[ROUTING] Total: {len(results)}, Passed: {len(passed)}, Failed: {len(failed)}")
    for r in results:
        logger.debug(f"  - {r.name}: passed={r.passed_qa}, retry_count={r.retry_count}")

    if failed:
        # Check if we can retry (only failed tasks that haven't exceeded max retries)
        retriable = [r for r in failed if r.retry_count < MAX_RETRIES]

        if retriable:
            logger.info(
                f"Routing {len(retriable)} failed agents back to generator (retrying). "
                f"{len(passed)} agents already passed and won't be retried."
            )

            # Find original tasks from plan to preserve their purposes
            if not state.plan_task or not state.plan_task.plan:
                logger.error("No plan available for retry")
                return "aggregator"

            retry_sends = []
            for r in retriable:
                original_task = find_task_by_id(state.plan_task.plan, r.task_id)
                if not original_task:
                    logger.error(f"Could not find original task for {r.task_id}, skipping retry")
                    continue

                retry_sends.append(  # pyright: ignore
                    Send(
                        "agent_generator",
                        {
                            "plan": [original_task],
                            "user_request": state.user_request,
                            "task_results": state.task_results,
                        },
                    )
                )

            return retry_sends if retry_sends else "aggregator"
        else:
            # Out of retries, go to aggregator anyway
            logger.warn(
                f"{len(failed)} agents failed and exceeded max retries. Moving to aggregator.",
            )
            return "aggregator"

    # All passed, go to aggregator
    logger.info("All agents passed QA! Moving to aggregator")
    return "aggregator"


async def aggregator_node(state: GraphState) -> dict[str, Any]:
    """Aggregates all task results (passed and failed) into final output."""
    # Get only the latest result for each task
    latest_results = get_latest_results(state.task_results)
    all_results = list(latest_results.values())

    passed = [r for r in all_results if r.passed_qa]
    failed = [r for r in all_results if not r.passed_qa]

    logger.info("Aggregating results:")
    logger.info(f"  Passed: {len(passed)}")
    if failed:
        logger.warn(
            f"  Failed: {len(failed)}",
        )

    # Format final output
    final_lines: list[str] = []

    if passed:
        final_lines.append("=== APPROVED AGENT CONFIGURATIONS ===\n")
        for r in passed:
            final_lines.append(f"Agent: {r.name}")
            final_lines.append(r.output)
            final_lines.append("")

    if failed:
        final_lines.append("\n=== FAILED AGENT CONFIGURATIONS (after max retries) ===\n")
        for r in failed:
            final_lines.append(f"Agent: {r.name}")
            final_lines.append(f"Last feedback: {r.feedback}")
            final_lines.append(f"Last output: {r.output}")
            final_lines.append("")

    final = "\n".join(final_lines)

    return {"final_output": final}


# ============= GRAPH BUILDER =============


def build_graph(agents: dict[str, Agent]) -> StateGraph:
    """
    Build and compile the agent scaffolding graph.

    Args:
        agents: Dictionary of agent instances (planner, agent_generator, qa)
    """
    builder = StateGraph(GraphState)

    # Create wrapper functions that bind agents
    async def planner_wrapper(state: GraphState) -> dict[str, Any]:
        return await planner_node(state, agents)

    async def plan_qa_wrapper(state: GraphState) -> dict[str, Any]:
        return await plan_qa_node(state, agents)

    async def agent_generator_wrapper(state: GraphState | dict[str, Any]) -> dict[str, Any]:
        # When called from conditional edges with Send, state is dict
        # When called normally, state is GraphState
        return await agent_generator_node(state, agents)  # type: ignore

    async def qa_wrapper(state: GraphState | dict[str, Any]) -> dict[str, Any]:
        # When called from conditional edges with Send, state is dict
        # When called normally, state is GraphState
        return await qa_node(state, agents)  # type: ignore

    # Add nodes with agent dependencies
    builder.add_node("planner", planner_wrapper)
    builder.add_node("plan_qa", plan_qa_wrapper)
    builder.add_node("agent_generator", agent_generator_wrapper)
    builder.add_node("qa", qa_wrapper)
    builder.add_node("qa_barrier", qa_review_barrier)
    builder.add_node("aggregator", aggregator_node)

    # Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "plan_qa")

    # Plan QA → Route (conditional: retry planner or proceed to agent_generator)
    def route_after_plan_qa(state: GraphState) -> str | list[Send]:
        """Route based on plan QA results."""
        route = route_plan_qa(state)

        # If route is "agent_generator", dispatch tasks in parallel
        if route == "agent_generator":
            if not state.plan_task or not state.plan_task.plan:
                logger.error("No plan tasks to dispatch!")
                return "aggregator"

            # Validate plan structure before dispatching
            for task in state.plan_task.plan:
                if not task.id or not task.name or not task.purpose:
                    logger.error(
                        f"Invalid task in plan: id={task.id}, name={task.name}, "
                        f"purpose={task.purpose}"
                    )
                    return "aggregator"

            return [
                Send(
                    "agent_generator",
                    {
                        "plan": [task],
                        "user_request": state.user_request,
                        "task_results": [],
                    },
                )
                for task in state.plan_task.plan
            ]
        # Otherwise return the route (planner or aggregator)
        return route

    builder.add_conditional_edges("plan_qa", route_after_plan_qa)

    # Agent Generator → QA (fan out all results in parallel)
    def dispatch_qa(state: GraphState) -> list[Send]:
        """Send each task result to QA independently (only latest version that needs review)."""
        # Get only the latest results that haven't passed QA yet (optimized filtering)
        pending_results = get_latest_results(state.task_results, pending_qa_only=True)
        pending_review = list(pending_results.values())

        # Calculate how many already passed for logging
        all_latest = get_latest_results(state.task_results)
        already_passed = len(all_latest) - len(pending_review)

        logger.info(
            f"[DISPATCH_QA] Dispatching {len(pending_review)} results to QA "
            f"({already_passed} already passed)"
        )
        for r in pending_review:
            logger.debug(f"  - {r.name} (retry_count={r.retry_count})")

        # Get plan from plan_task
        plan = state.plan_task.plan if state.plan_task else []

        return [
            Send(
                "qa",
                {
                    "task_results": [r],
                    "plan": plan,
                    "user_request": state.user_request,
                },
            )
            for r in pending_review
        ]

    builder.add_conditional_edges("agent_generator", dispatch_qa)

    # QA → Barrier (wait for all QA nodes to complete)
    builder.add_edge("qa", "qa_barrier")

    # Barrier → Route (conditional: retry or move to aggregator)
    builder.add_conditional_edges("qa_barrier", route_qa_results)

    # Aggregator is final
    builder.add_edge("aggregator", END)

    return builder.compile()  # pyright: ignore


# ============= MAIN EXECUTION =============


async def run_agent_scaffolding(user_request: str, agents: dict[str, Agent]) -> str:
    """
    Run the agent scaffolding graph with the given user request.

    Args:
        user_request: The user's request for agent generation
        agents: Dictionary of agent instances

    Returns:
        Final aggregated output containing approved and failed agent configurations
    """
    logger.info("=" * 70)
    logger.info("DYNAMIC AGENT SCAFFOLDING WITH QA FEEDBACK LOOP")
    logger.info("=" * 70)
    logger.info(f"User Request: {user_request}")
    logger.info("")

    graph = build_graph(agents)

    # Create initial state
    initial_state = GraphState(
        user_request=user_request,
        plan_task=None,
        task_results=[],
        final_output="",
    )

    # Run the graph with increased recursion limit for retry loops
    # Each retry cycle: agent_generator → qa → qa_barrier → route_qa_results
    # Max iterations = (initial_generation + MAX_RETRIES) * num_agents
    # Setting limit to 100 to handle complex scenarios with multiple retries
    result = await graph.ainvoke(initial_state.model_dump(), {"recursion_limit": 100})  # pyright: ignore

    logger.info("=" * 70)
    logger.info("FINAL OUTPUT:")
    logger.info("=" * 70)
    logger.info(result["final_output"])

    # Summary
    final_state = GraphState(**result)
    latest_results = get_latest_results(final_state.task_results)
    all_results = list(latest_results.values())

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXECUTION SUMMARY:")
    logger.info("=" * 70)
    logger.info(f"Total agents: {len(all_results)}")
    logger.info(f"Passed QA: {sum(1 for r in all_results if r.passed_qa)}")
    logger.info(f"Failed QA: {sum(1 for r in all_results if not r.passed_qa)}")

    return result["final_output"]


async def main() -> None:
    """Main entry point."""
    from ai_gateway.domain import AgentFactory

    # Generate agents once at the start
    logger.info("Initializing agents...")
    default_agents = AgentFactory.generate_default_agents()

    # Create agents dictionary
    agents = {
        "planner": default_agents.planner,
        "plan_qa": default_agents.plan_qa,
        "agent_generator": default_agents.agent_generator,
        "qa": default_agents.qa,
    }

    # Example usage
    # user_request = (
    #     "I need a team to perform research on a given subject from internet and create a report."
    # )
    user_request = (
        "I need a team to perform statistical analysis on historical data and create a report"
    )

    await run_agent_scaffolding(user_request, agents)


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 70)
    logger.info("DYNAMIC AGENT GENERATOR")
    logger.info("=" * 70)
    logger.info("")

    asyncio.run(main())
