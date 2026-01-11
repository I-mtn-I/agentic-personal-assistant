"""Agent scaffolding graph with dynamic agent generation, QA review, and retry logic."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from data_portal.helpers.colored_logger import ColoredLogger
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ai_gateway.config.settings import TOOLS_CONFIG
from ai_gateway.domain.agent import Agent
from ai_gateway.domain.agent_persistence import AgentConfigRepository, ToolSpec
from ai_gateway.domain.scaffolding_models import (
    AgentCreationTask,
    AgentCreationTaskResult,
    GraphState,
    PlanTask,
)
from ai_gateway.utils import extract_json_from_response

MAX_RETRIES = 3  # Maximum retry attempts for both plan and agent tasks


class AgentScaffoldingGraph:
    """Builds and runs the dynamic agent scaffolding graph."""

    def __init__(
        self,
        agents: dict[str, Agent],
        persistence: AgentConfigRepository | None = None,
        logger: ColoredLogger | None = None,
    ) -> None:
        self.agents = agents
        self.persistence = persistence
        self.logger = logger or ColoredLogger(level="DEBUG")
        self._team_id: uuid.UUID | None = None
        self._team_user_request = ""
        self._persisted_agent_ids: list[uuid.UUID] = []
        self._manager_agent_config_id: uuid.UUID | None = None
        self._persist_lock = asyncio.Lock()

    @staticmethod
    def _find_task_by_id(plan: list[AgentCreationTask], task_id: str) -> AgentCreationTask | None:
        for task in plan:
            if task.id == task_id:
                return task
        return None

    @staticmethod
    def _find_previous_result(
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

    @staticmethod
    def _get_latest_results(
        task_results: list[AgentCreationTaskResult],
        pending_qa_only: bool = False,
    ) -> dict[str, AgentCreationTaskResult]:
        """
        Get the latest result for each task_id.

        Priority:
        1. If a result has passed QA, always prefer it (regardless of retry_count)
        2. Otherwise, prefer the result with the highest retry_count
        """
        latest: dict[str, AgentCreationTaskResult] = {}
        for result in task_results:
            task_id = result.task_id

            if task_id not in latest:
                latest[task_id] = result
            else:
                current = latest[task_id]

                if current.passed_qa:
                    continue

                if result.passed_qa:
                    latest[task_id] = result
                elif result.retry_count > current.retry_count:
                    latest[task_id] = result

        if pending_qa_only:
            return {k: v for k, v in latest.items() if not v.passed_qa}

        return latest

    def _get_available_tools_info(self) -> str:
        available_tools = TOOLS_CONFIG.list_names()
        return "\n".join(
            [f"  - {name}: {TOOLS_CONFIG.raw(name).description}" for name in available_tools]
        )

    async def _persist_agent_config(
        self,
        *,
        config_data: dict[str, Any],
        task: AgentCreationTask | None,
    ) -> None:
        if not self.persistence or not task or not self._team_id:
            return

        tools: list[ToolSpec] = []
        for tool_name in config_data.get("tools", []):
            tool_cfg = TOOLS_CONFIG.raw(tool_name)
            tools.append(
                ToolSpec(
                    name=tool_name,
                    target=tool_cfg.target,
                    description=tool_cfg.description,
                )
            )

        agent_id = config_data.get("agent_id", task.id)
        agent_name = config_data.get("agent_name", task.name)
        system_prompt = config_data.get("system_prompt", "")

        try:
            agent_config_id = await asyncio.to_thread(
                self.persistence.save_agent_config,
                agent_id=agent_id,
                agent_name=agent_name,
                purpose=task.purpose,
                system_prompt=system_prompt,
                tools=tools,
                team_id=self._team_id,
                is_manager=task.is_manager,
            )
            async with self._persist_lock:
                self._persisted_agent_ids.append(agent_config_id)
                if task.is_manager:
                    self._manager_agent_config_id = agent_config_id
        except Exception as exc:  # pragma: no cover - fail-safe persistence
            self.logger.error(f"Failed to persist agent config for {agent_name}: {exc}")

    async def planner_node(self, state: GraphState) -> dict[str, Any]:
        """Planner generates dynamic task configs based on user request."""
        if state.plan_task and not state.plan_task.passed_qa:
            retry_count = state.plan_task.retry_count + 1
            feedback = state.plan_task.feedback
        else:
            retry_count = 0
            feedback = ""

        if retry_count > 0:
            self.logger.info(f"RE-PLANNING (Attempt #{retry_count})")
            self.logger.warn(f"Previous Plan QA feedback: {feedback}")
        else:
            self.logger.info("Planner agent running...")

        planner_agent = self.agents["planner"]
        tools_info = self._get_available_tools_info()

        planner_query = f"""User Request: {state.user_request}
        Analyze the user request and create a plan for a multi-agent architecture.

        Available Tools:
        {tools_info}

        {f"Previous Plan QA Feedback: {feedback}" if feedback else ""}
    """

        response = await planner_agent.ask(planner_query)

        try:
            plan_data = extract_json_from_response(response)

            tasks = [
                AgentCreationTask(
                    id=agent["id"],
                    name=agent["name"],
                    purpose=agent["purpose"],
                    is_manager=agent.get("is_manager", False),
                    tools=[],
                    system_prompt="",
                )
                for agent in plan_data.get("agents", [])
            ]

            self.logger.info(f"Generated {len(tasks)} agent task(s).")
            for task in tasks:
                current_task = task.model_dump()
                self.logger.debug(f"\t- Task ID: {current_task['id']}:")
                self.logger.debug(
                    f"\t\t Name: {current_task['name']}, Purpose: {current_task['purpose']}"
                )

            plan_task = PlanTask(
                plan=tasks,
                plan_json=json.dumps(plan_data),
                passed_qa=False,
                feedback="",
                retry_count=retry_count,
            )

            return {"plan_task": plan_task}

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse planner response: {str(e)}")
            self.logger.error(f"Raw response: {response}")

            if retry_count == 0:
                self.logger.warn("First attempt parsing failed. Auto-retrying once...")
                error_msg = (
                    f"Previous attempt failed to parse JSON: {str(e)}. "
                    "Please ensure output is valid JSON."
                )
                state.plan_task = PlanTask(
                    plan=[],
                    plan_json="",
                    passed_qa=False,
                    feedback=error_msg,
                    retry_count=0,
                )
                return await self.planner_node(state)

            plan_task = PlanTask(
                plan=[],
                plan_json="",
                passed_qa=False,
                feedback=f"Planner failed to produce valid JSON: {str(e)}",
                retry_count=retry_count,
            )
            return {"plan_task": plan_task}

    async def plan_qa_node(self, state: GraphState) -> dict[str, Any]:
        """QA reviews the overall plan generated by the planner."""
        self.logger.info("QA reviewing the overall plan...")

        if not state.plan_task or not state.plan_task.plan:
            self.logger.warn("No plan to review...")
            return {}

        manager_count = sum(1 for task in state.plan_task.plan if task.is_manager)
        if manager_count != 1:
            feedback = (
                "Plan must include exactly one manager agent with is_manager=true. "
                f"Found {manager_count} manager agents."
            )
            self.logger.warn(f"Plan QA pre-check failed: {feedback}")
            updated_plan_task = PlanTask(
                plan=state.plan_task.plan,
                plan_json=state.plan_task.plan_json,
                passed_qa=False,
                feedback=feedback,
                retry_count=state.plan_task.retry_count,
            )
            return {"plan_task": updated_plan_task}

        plan_qa_agent = self.agents["plan_qa"]

        qa_prompt = f"""
        Given Original User Request:
            {state.user_request}
        And Generated Plan:
            {state.plan_task.plan_json}

        Perform the review.
        """

        qa_response = await plan_qa_agent.ask(qa_prompt)
        self.logger.debug(f"Plan QA response:\n {qa_response}")

        try:
            qa_data = extract_json_from_response(qa_response)
            passed = qa_data.get("overall_decision") == "approved"
            feedback = qa_data.get("feedback", "")

            if passed:
                self.logger.info("Plan QA PASSED")
            else:
                self.logger.warn(f"Plan QA FAILED: {feedback}")

            updated_plan_task = PlanTask(
                plan=state.plan_task.plan,
                plan_json=state.plan_task.plan_json,
                passed_qa=passed,
                feedback=feedback,
                retry_count=state.plan_task.retry_count,
            )

            return {"plan_task": updated_plan_task}

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse plan QA response: {str(e)}")

            updated_plan_task = PlanTask(
                plan=state.plan_task.plan,
                plan_json=state.plan_task.plan_json,
                passed_qa=False,
                feedback=f"Plan QA failed to parse response: {str(e)}",
                retry_count=state.plan_task.retry_count,
            )

            return {"plan_task": updated_plan_task}

    async def agent_generator_node(self, state: GraphState | dict[str, Any]) -> dict[str, Any]:
        """
        Agent generator creates system prompts and selects tools for each agent.
        Receives a single task via Send (always as dict).
        """
        state_dict = state.model_dump() if isinstance(state, GraphState) else state
        plan = [
            AgentCreationTask(**t) if isinstance(t, dict) else t for t in state_dict.get("plan", [])
        ]
        full_plan = [
            AgentCreationTask(**t) if isinstance(t, dict) else t
            for t in state_dict.get("full_plan", plan)
        ]
        task_results = [
            AgentCreationTaskResult(**r) if isinstance(r, dict) else r
            for r in state_dict.get("task_results", [])
        ]
        user_request = state_dict.get("user_request", "")

        task = plan[0]
        task_id = task.id
        task_name = task.name
        task_purpose = task.purpose

        _, feedback, retry_count = self._find_previous_result(task_results, task_id)

        if feedback:
            self.logger.info(
                f"RE-GENERATING agent: {task_name} (ID: {task_id}, Attempt #{retry_count + 1})"
            )
            self.logger.warn(f"Previous QA feedback: {feedback}")
        else:
            self.logger.info(f"Generating agent: {task_name} (ID: {task_id})")

        agent_generator = self.agents["agent_generator"]
        tools_info = self._get_available_tools_info()

        base_query = f"""Agent Name: {task_name}
Agent Purpose: {task_purpose}
User's Original Request: {user_request}

Available Tools:
{tools_info}"""

        if task.is_manager:
            sub_agents = [
                {"id": t.id, "name": t.name, "purpose": t.purpose}
                for t in full_plan
                if not t.is_manager
            ]
            base_query = f"""{base_query}

Team Context:
- You are the manager agent (is_manager=true).
- Sub-agents available to you:
{json.dumps(sub_agents, indent=2)}"""

        if feedback:
            generator_query = f"""{base_query}

Previous QA Feedback: {feedback}

Please regenerate the agent configuration addressing the feedback above.
Return your response as a JSON object with the required structure."""
        else:
            generator_query = f"""{base_query}

Please create a complete agent configuration.
Return your response as a JSON object with the required structure."""

        response = await agent_generator.ask(generator_query)
        self.logger.debug(f"Agent generator response for {task_name}: {response[:200]}")

        try:
            config_data = extract_json_from_response(response)

            required_fields = ["agent_id", "agent_name", "tools", "system_prompt", "sub_agents"]
            missing_fields = [f for f in required_fields if f not in config_data]

            if missing_fields:
                raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

            if not isinstance(config_data.get("tools"), list):
                raise ValueError("Field 'tools' must be a list")
            if "sub_agents" not in config_data:
                raise KeyError("Missing required fields: sub_agents")
            if not isinstance(config_data.get("sub_agents"), list):
                raise ValueError("Field 'sub_agents' must be a list")

            task_result = AgentCreationTaskResult(
                task_id=task_id,
                name=task_name,
                output=json.dumps(config_data, indent=2),
                passed_qa=False,
                feedback="",
                retry_count=retry_count,
            )

            self.logger.info(f"Agent configuration generated for {task_name}")
            self.logger.debug(f"Selected tools: {', '.join(config_data.get('tools', []))}")

            return {"task_results": [task_result]}

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse agent_generator response for {task_name}: {str(e)}")

            task_result = AgentCreationTaskResult(
                task_id=task_id,
                name=task_name,
                output=f"ERROR: Failed to generate configuration - {str(e)}",
                passed_qa=False,
                feedback=f"Agent generator failed validation: {str(e)}",
                retry_count=retry_count,
            )

            return {"task_results": [task_result]}

    async def qa_node(self, state: GraphState | dict[str, Any]) -> dict[str, Any]:
        """
        QA reviews ONE task result and validates the generated agent configuration.
        Receives a single result via Send (always as dict).
        Returns TaskResult with passed_qa=True/False and feedback.
        """
        state_dict = state.model_dump() if isinstance(state, GraphState) else state
        task_results = [
            AgentCreationTaskResult(**r) if isinstance(r, dict) else r
            for r in state_dict.get("task_results", [])
        ]
        plan = [
            AgentCreationTask(**t) if isinstance(t, dict) else t for t in state_dict.get("plan", [])
        ]
        user_request = state_dict.get("user_request", "")

        result = task_results[0]
        task_id = result.task_id

        self.logger.info(
            f"QA reviewing agent: {result.name} (ID: {task_id}, Attempt #{result.retry_count + 1})"
        )

        qa_agent = self.agents["qa"]

        try:
            if result.output.startswith("ERROR:"):
                self.logger.error(f"QA found generation error for {result.name}")
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

            original_task = self._find_task_by_id(plan, task_id)

            tools_info = self._get_available_tools_info()

            other_agents_context = []
            latest_results = self._get_latest_results(task_results)

            for other_result in latest_results.values():
                if other_result.task_id != task_id and not other_result.output.startswith("ERROR:"):
                    try:
                        other_config = json.loads(other_result.output)
                        other_tools = other_config.get("tools", [])
                        other_task = self._find_task_by_id(plan, other_result.task_id)
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

            manager_expectations = ""
            if original_task and original_task.is_manager:
                expected_subagents = [
                    {"id": t.id, "name": t.name, "purpose": t.purpose}
                    for t in plan
                    if not t.is_manager
                ]
                manager_expectations = f"""
Manager Requirements:
- This agent is the manager and must include sub_agents for all non-manager agents.
- Expected sub_agents list:
{json.dumps(expected_subagents, indent=2)}
"""

            qa_query = f"""Original User Request: {user_request}

Agent Under Review:
- Name: {result.name}
- Purpose: {original_task.purpose if original_task else "N/A"}
- Is Manager: {original_task.is_manager if original_task else False}

Other Agents in the Team:
{team_context}

Available Tools in System:
{tools_info}

{manager_expectations}

Generated Configuration for {result.name}:
{json.dumps(config_data, indent=2)}

Please review this agent configuration and provide an audit report."""

            qa_response = await qa_agent.ask(qa_query)
            self.logger.debug(f"QA response for {result.name}: {qa_response[:200]}")

            qa_data = extract_json_from_response(qa_response)

            passed = qa_data.get("overall_decision") == "approved"
            feedback = qa_data.get("feedback", "")

            if passed:
                self.logger.info(f"QA PASSED for {result.name}")
            else:
                self.logger.warn(f"QA FAILED for {result.name}: {feedback}")

            updated_result = AgentCreationTaskResult(
                task_id=task_id,
                name=result.name,
                output=result.output,
                passed_qa=passed,
                feedback=feedback,
                retry_count=result.retry_count,
            )

            if passed:
                await self._persist_agent_config(config_data=config_data, task=original_task)

            return {"task_results": [updated_result]}

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"QA failed to process result for {result.name}: {str(e)}")

            updated_result = AgentCreationTaskResult(
                task_id=task_id,
                name=result.name,
                output=result.output,
                passed_qa=False,
                feedback=f"QA processing error: {str(e)}. Please ensure output is valid JSON.",
                retry_count=result.retry_count,
            )

            return {"task_results": [updated_result]}

    def _route_plan_qa(self, state: GraphState) -> str:
        """
        Routes after plan QA:
        - Plan failed + retries available → back to planner
        - Plan failed + no retries left → proceed to agent_generator anyway
        - Plan passed → proceed to agent_generator
        """
        if not state.plan_task:
            self.logger.error("No plan_task found in state")
            return "aggregator"

        if state.plan_task.passed_qa:
            self.logger.info("Plan QA passed! Proceeding to agent generation...")
            return "agent_generator"

        if state.plan_task.retry_count < MAX_RETRIES:
            self.logger.warn(
                f"Plan QA failed (attempt #{state.plan_task.retry_count + 1}). Retrying planner..."
            )
            return "planner"
        else:
            self.logger.error(
                f"Plan QA failed after {MAX_RETRIES} retries. Proceeding with current plan anyway.."
            )
            return "agent_generator"

    async def _qa_review_barrier(self, state: GraphState) -> dict[str, Any]:  # noqa: ARG001
        latest_results = self._get_latest_results(state.task_results)
        self.logger.debug(
            f"QA review - all {len(latest_results)} QA nodes completed, proceeding to routing"
        )
        return {}

    def _route_qa_results(self, state: GraphState) -> list[Send] | str:
        """
        Routes results after ALL QA reviews complete:
        - Failed + retries available → back to agent_generator
        - Failed + no retries left → go to aggregator anyway
        - Passed → to aggregator
        """
        latest_results = self._get_latest_results(state.task_results)
        results = list(latest_results.values())

        passed = [r for r in results if r.passed_qa]
        failed = [r for r in results if not r.passed_qa]

        self.logger.info(
            f"[ROUTING] Total: {len(results)}, Passed: {len(passed)}, Failed: {len(failed)}"
        )
        for r in results:
            self.logger.debug(f"  - {r.name}: passed={r.passed_qa}, retry_count={r.retry_count}")

        if failed:
            retriable = [r for r in failed if r.retry_count < MAX_RETRIES]

            if retriable:
                self.logger.info(
                    f"Routing {len(retriable)} failed agents back to generator (retrying). "
                    f"{len(passed)} agents already passed and won't be retried."
                )

                if not state.plan_task or not state.plan_task.plan:
                    self.logger.error("No plan available for retry")
                    return "aggregator"

                retry_sends = []
                for r in retriable:
                    original_task = self._find_task_by_id(state.plan_task.plan, r.task_id)
                    if not original_task:
                        self.logger.error(
                            f"Could not find original task for {r.task_id}, skipping retry"
                        )
                        continue

                    retry_sends.append(  # pyright: ignore
                        Send(
                            "agent_generator",
                            {
                                "plan": [original_task],
                                "full_plan": state.plan_task.plan,
                                "user_request": state.user_request,
                                "task_results": state.task_results,
                            },
                        )
                    )

                return retry_sends if retry_sends else "aggregator"
            else:
                self.logger.warn(
                    f"{len(failed)} agents failed and exceeded max retries. Moving to aggregator.",
                )
                return "aggregator"

        self.logger.info("All agents passed QA! Moving to aggregator")
        return "aggregator"

    async def aggregator_node(self, state: GraphState) -> dict[str, Any]:
        """Aggregates all task results (passed and failed) into final output."""
        latest_results = self._get_latest_results(state.task_results)
        all_results = list(latest_results.values())

        passed = [r for r in all_results if r.passed_qa]
        failed = [r for r in all_results if not r.passed_qa]

        self.logger.info("Aggregating results:")
        self.logger.info(f"  Passed: {len(passed)}")
        if failed:
            self.logger.warn(f"  Failed: {len(failed)}")

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

        if self.persistence and self._team_id and self._persisted_agent_ids:
            if self._manager_agent_config_id:
                try:
                    await asyncio.to_thread(
                        self.persistence.save_team_config,
                        team_id=self._team_id,
                        agent_config_ids=self._persisted_agent_ids,
                        manager_agent_id=self._manager_agent_config_id,
                        user_request=self._team_user_request,
                    )
                except Exception as exc:  # pragma: no cover - fail-safe persistence
                    self.logger.error(f"Failed to persist team config: {exc}")
            else:
                self.logger.warn("Skipping team persistence: manager agent not persisted.")

        return {"final_output": final}

    def build_graph(self) -> StateGraph:
        """Build and compile the agent scaffolding graph."""
        builder = StateGraph(GraphState)

        async def planner_wrapper(state: GraphState) -> dict[str, Any]:
            return await self.planner_node(state)

        async def plan_qa_wrapper(state: GraphState) -> dict[str, Any]:
            return await self.plan_qa_node(state)

        async def agent_generator_wrapper(state: GraphState | dict[str, Any]) -> dict[str, Any]:
            return await self.agent_generator_node(state)

        async def qa_wrapper(state: GraphState | dict[str, Any]) -> dict[str, Any]:
            return await self.qa_node(state)

        builder.add_node("planner", planner_wrapper)  # pyright: ignore
        builder.add_node("plan_qa", plan_qa_wrapper)  # pyright: ignore
        builder.add_node("agent_generator", agent_generator_wrapper)  # pyright: ignore
        builder.add_node("qa", qa_wrapper)  # pyright: ignore
        builder.add_node("qa_barrier", self._qa_review_barrier)  # pyright: ignore
        builder.add_node("aggregator", self.aggregator_node)  # pyright: ignore

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "plan_qa")

        def route_after_plan_qa(state: GraphState) -> str | list[Send]:
            route = self._route_plan_qa(state)

            if route == "agent_generator":
                if not state.plan_task or not state.plan_task.plan:
                    self.logger.error("No plan tasks to dispatch!")
                    return "aggregator"

                for task in state.plan_task.plan:
                    if not task.id or not task.name or not task.purpose:
                        self.logger.error(
                            f"Invalid task in plan: id={task.id}, name={task.name}, "
                            f"purpose={task.purpose}"
                        )
                        return "aggregator"

                return [
                    Send(
                        "agent_generator",
                        {
                            "plan": [task],
                            "full_plan": state.plan_task.plan,
                            "user_request": state.user_request,
                            "task_results": [],
                        },
                    )
                    for task in state.plan_task.plan
                ]
            return route

        builder.add_conditional_edges("plan_qa", route_after_plan_qa)

        def dispatch_qa(state: GraphState) -> list[Send]:
            pending_results = self._get_latest_results(state.task_results, pending_qa_only=True)
            pending_review = list(pending_results.values())

            all_latest = self._get_latest_results(state.task_results)
            already_passed = len(all_latest) - len(pending_review)

            self.logger.info(
                f"[DISPATCH_QA] Dispatching {len(pending_review)} results to QA "
                f"({already_passed} already passed)"
            )
            for r in pending_review:
                self.logger.debug(f"  - {r.name} (retry_count={r.retry_count})")

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
        builder.add_edge("qa", "qa_barrier")
        builder.add_conditional_edges("qa_barrier", self._route_qa_results)
        builder.add_edge("aggregator", END)

        return builder.compile()  # pyright: ignore

    async def run(self, user_request: str) -> str:
        self._team_id = uuid.uuid4()
        self._team_user_request = user_request
        self._persisted_agent_ids = []
        self._manager_agent_config_id = None

        self.logger.info("=" * 70)
        self.logger.info("DYNAMIC AGENT SCAFFOLDING WITH QA FEEDBACK LOOP")
        self.logger.info("=" * 70)
        self.logger.info(f"User Request: {user_request}")
        self.logger.info("")

        graph = self.build_graph()  # pyright: ignore

        initial_state = GraphState(
            user_request=user_request,
            plan_task=None,
            task_results=[],
            final_output="",
        )

        result = await graph.ainvoke(initial_state.model_dump(), {"recursion_limit": 100})  # pyright: ignore

        self.logger.info("=" * 70)
        self.logger.info("FINAL OUTPUT:")
        self.logger.info("=" * 70)
        self.logger.info(result["final_output"])

        final_state = GraphState(**result)
        latest_results = self._get_latest_results(final_state.task_results)
        all_results = list(latest_results.values())

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("EXECUTION SUMMARY:")
        self.logger.info("=" * 70)
        self.logger.info(f"Total agents: {len(all_results)}")
        self.logger.info(f"Passed QA: {sum(1 for r in all_results if r.passed_qa)}")
        self.logger.info(f"Failed QA: {sum(1 for r in all_results if not r.passed_qa)}")

        return result["final_output"]
