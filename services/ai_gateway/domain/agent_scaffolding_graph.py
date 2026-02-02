"""Agent scaffolding graph with dynamic agent generation, QA review, and retry logic."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from data_portal.helpers.colored_logger import ColoredLogger
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from ai_gateway.config.settings import AVAILABLE_TAGS, TOOLS_CONFIG
from ai_gateway.domain.agent import Agent
from ai_gateway.domain.agent_persistence import AgentConfigRepository, ToolSpec
from ai_gateway.domain.scaffolding_models import (
    AgentCreationTask,
    AgentCreationTaskResult,
    AgentGeneratorOutput,
    AgentQAOutput,
    GraphState,
    PlannerOutput,
    PlanQAOutput,
    PlanTask,
    TeamQAOutput,
    TeamQAPlanIssue,
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
        self._persisted_task_ids: set[str] = set()
        self._persist_lock = asyncio.Lock()

    @staticmethod
    def _find_task_by_id(plan: list[AgentCreationTask], task_id: str) -> AgentCreationTask | None:
        for task in plan:
            if task.id == task_id:
                return task
        return None

    @staticmethod
    def _find_previous_result(
        task_results: list[AgentCreationTaskResult],
        task_id: str,
        *,
        plan_run_id: str | None = None,
    ) -> tuple[AgentCreationTaskResult | None, str, int]:
        """
        Find previous result for a task.

        Returns:
            Tuple of (previous_result, feedback, retry_count)
        """
        latest: AgentCreationTaskResult | None = None
        for existing in task_results:
            if existing.task_id != task_id:
                continue
            if plan_run_id and existing.plan_run_id != plan_run_id:
                continue
            if latest is None or existing.retry_count >= latest.retry_count:
                latest = existing
        if latest is not None:
            return latest, latest.feedback, latest.retry_count + 1
        return None, "", 0

    @staticmethod
    def _extract_structured_response(response: dict[str, Any]) -> dict[str, Any] | None:
        structured = response.get("structured_response")
        if structured is None:
            return None
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        if isinstance(structured, dict):
            return structured
        return None

    @staticmethod
    def _get_latest_results(
        task_results: list[AgentCreationTaskResult],
        *,
        plan_run_id: str | None = None,
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
            if plan_run_id and result.plan_run_id != plan_run_id:
                continue
            task_id = result.task_id

            if task_id not in latest:
                latest[task_id] = result
            else:
                current = latest[task_id]

                if current.passed_qa:
                    continue

                if result.passed_qa:
                    latest[task_id] = result
                elif result.retry_count >= current.retry_count:
                    latest[task_id] = result

        if pending_qa_only:
            return {k: v for k, v in latest.items() if not v.passed_qa}

        return latest

    @staticmethod
    def _safe_json_loads(value: str) -> dict[str, Any] | None:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    def _build_team_review_results(
        self,
        *,
        plan: list[AgentCreationTask],
        latest_results: dict[str, AgentCreationTaskResult],
        team_qa: dict[str, Any],
        plan_run_id: str,
    ) -> list[AgentCreationTaskResult]:
        reviews = team_qa.get("agent_reviews") or []
        review_by_task = {r.get("task_id"): r for r in reviews if isinstance(r, dict)}
        new_results: list[AgentCreationTaskResult] = []

        for task in plan:
            review = review_by_task.get(task.id)
            if not review:
                continue
            previous = latest_results.get(task.id)
            output = previous.output if previous else ""
            retry_count = previous.retry_count if previous else 0
            issues = review.get("issues") or []
            blocking = any(isinstance(issue, dict) and issue.get("severity") in {"high", "critical"} for issue in issues)
            decision = review.get("decision") == "approved" or not blocking
            feedback = ""
            if not decision:
                if isinstance(issues, list) and issues:
                    feedback = json.dumps(issues, ensure_ascii=True)
                else:
                    feedback = str(review.get("feedback") or "Needs revision")

            new_results.append(
                AgentCreationTaskResult(
                    task_id=task.id,
                    name=task.name,
                    output=output,
                    passed_qa=decision,
                    feedback=feedback,
                    retry_count=retry_count,
                    plan_run_id=plan_run_id,
                )
            )

        return new_results

    def _render_markdown_report(
        self,
        *,
        user_request: str,
        plan: list[AgentCreationTask],
        results: list[AgentCreationTaskResult],
    ) -> str:
        plan_by_id = {task.id: task for task in plan}

        passed_count = sum(1 for r in results if r.passed_qa)
        failed_count = sum(1 for r in results if not r.passed_qa)

        lines: list[str] = []
        lines.append("# Agent Scaffolding Result")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Total agents: {len(results)}")
        lines.append(f"- Passed QA: {passed_count}")
        lines.append(f"- Failed QA: {failed_count}")
        if user_request:
            lines.append(f"- User request: {user_request}")
        lines.append("")
        lines.append("## Agents")

        if not results:
            lines.append("")
            lines.append("_No agent results available._")
            return "\n".join(lines)

        ordered_results = results
        if plan_by_id:
            order_map = {task_id: index for index, task_id in enumerate(plan_by_id)}
            ordered_results = sorted(
                results,
                key=lambda result: order_map.get(result.task_id, len(order_map)),
            )

        for result in ordered_results:
            plan_task = plan_by_id.get(result.task_id)
            parsed = self._safe_json_loads(result.output)
            agent_id = parsed.get("agent_id") if parsed else None
            agent_name = parsed.get("agent_name") if parsed else result.name
            tools = parsed.get("tools") if parsed else None
            system_prompt = parsed.get("system_prompt") if parsed else None
            sub_agents = parsed.get("sub_agents") if parsed else None

            lines.append("")
            lines.append(f"### {agent_name}")
            lines.append(f"- Status: {'Approved' if result.passed_qa else 'Failed'}")
            if agent_id:
                lines.append(f"- Agent ID: {agent_id}")
            if plan_task:
                lines.append(f"- Purpose: {plan_task.purpose}")
                lines.append(f"- Manager: {'Yes' if plan_task.is_manager else 'No'}")
            if tools is not None:
                tools_display = ", ".join(tools) if tools else "None"
                lines.append(f"- Tools: {tools_display}")
            if sub_agents is not None:
                sub_agents_display = ", ".join(sub_agents) if sub_agents else "None"
                lines.append(f"- Sub-agents: {sub_agents_display}")
            if not result.passed_qa and result.feedback:
                lines.append(f"- QA feedback: {result.feedback}")

            if system_prompt:
                lines.append("")
                lines.append("#### System Prompt")
                lines.append("```")
                lines.append(system_prompt)
                lines.append("```")
            elif parsed is None:
                lines.append("")
                lines.append("#### Raw Output")
                lines.append("```")
                lines.append(result.output)
                lines.append("```")

        return "\n".join(lines)

    @staticmethod
    def _resolve_project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    def _write_markdown_report(self, markdown: str) -> None:
        try:
            report_path = self._resolve_project_root() / "result.md"
            report_path.write_text(markdown, encoding="utf-8")
            self.logger.info(f"Markdown report written to {report_path}")
        except Exception as exc:  # pragma: no cover - fail-safe IO
            self.logger.error(f"Failed to write markdown report: {exc}")

    async def _persist_agent_config(
        self,
        *,
        config_data: dict[str, Any],
        task: AgentCreationTask | None,
    ) -> None:
        if not self.persistence or not task or not self._team_id:
            return
        if task.id in self._persisted_task_ids:
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
                self._persisted_task_ids.add(task.id)
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
        allowed_tags = ", ".join(AVAILABLE_TAGS)
        planner_query = f"""User Request: {state.user_request}
        Analyze the user request and create a plan for a multi-agent architecture.
        You must call the list_all_tools tool to view all tools and their tags. It returns JSON:
        {{"tools": [{{"name": "<tool_name>", "description": "<tool_description>", "tags": ["<tag>"], "disallowed_tags": ["<tag>"]}}]}}

        Allowed Tags:
        {allowed_tags}

        {f"Previous Plan QA Feedback: {feedback}" if feedback else ""}

        Return a structured response matching the configured schema.
    """

        response = await planner_agent.ask_raw(planner_query)

        try:
            plan_data = self._extract_structured_response(response)
            if plan_data is None:
                plan_data = extract_json_from_response(response["messages"][-1].content)
            else:
                plan_data = PlannerOutput.model_validate(plan_data).model_dump()

            tasks = [
                AgentCreationTask(
                    id=agent["id"],
                    name=agent["name"],
                    purpose=agent["purpose"],
                    tags=agent.get("tags", []),
                    is_manager=agent.get("is_manager", False),
                    tools=[],
                    system_prompt="",
                )
                for agent in plan_data.get("agents", [])
            ]

            invalid_tags = set()
            allowed_tags = set(AVAILABLE_TAGS)
            for task in tasks:
                invalid_tags.update(tag for tag in task.tags if tag not in allowed_tags)
            if invalid_tags:
                raise ValueError(f"Invalid tags found in plan: {sorted(invalid_tags)}")

            self.logger.info(f"Generated {len(tasks)} agent task(s).")
            for task in tasks:
                current_task = task.model_dump()
                self.logger.debug(f"\t- Task ID: {current_task['id']}:")
                self.logger.debug(f"\t\t Name: {current_task['name']}, Purpose: {current_task['purpose']}")

            plan_task = PlanTask(
                plan=tasks,
                plan_json=json.dumps(plan_data),
                passed_qa=False,
                feedback="",
                retry_count=retry_count,
                run_id=str(uuid.uuid4()),
            )

            return {"plan_task": plan_task}

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse planner response: {str(e)}")
            self.logger.error(f"Raw response: {response.get('messages', [])}")

            if retry_count == 0:
                self.logger.warn("First attempt parsing failed. Auto-retrying once...")
                error_msg = f"Previous attempt failed to parse JSON: {str(e)}. Please ensure output is valid JSON."
                state.plan_task = PlanTask(
                    plan=[],
                    plan_json="",
                    passed_qa=False,
                    feedback=error_msg,
                    retry_count=0,
                    run_id=str(uuid.uuid4()),
                )
                return await self.planner_node(state)

            plan_task = PlanTask(
                plan=[],
                plan_json="",
                passed_qa=False,
                feedback=f"Planner failed to produce valid JSON: {str(e)}",
                retry_count=retry_count,
                run_id=str(uuid.uuid4()),
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
            feedback = f"Plan must include exactly one manager agent with is_manager=true. Found {manager_count} manager agents."
            self.logger.warn(f"Plan QA pre-check failed: {feedback}")
            updated_plan_task = PlanTask(
                plan=state.plan_task.plan,
                plan_json=state.plan_task.plan_json,
                passed_qa=False,
                feedback=feedback,
                retry_count=state.plan_task.retry_count,
                run_id=state.plan_task.run_id,
            )
            return {"plan_task": updated_plan_task}

        plan_qa_agent = self.agents["plan_qa"]

        qa_prompt = f"""Given Original User Request:
            {state.user_request}
        And Generated Plan:
            {state.plan_task.plan_json}

        Perform the review.
        Use overall_decision with one of: approved, needs_revision.
        Return a structured response matching the configured schema.
        """

        qa_response = await plan_qa_agent.ask_raw(qa_prompt)
        self.logger.debug(f"Plan QA response:\n {qa_response.get('messages', [])}")

        try:
            qa_data = self._extract_structured_response(qa_response)
            if qa_data is None:
                qa_data = extract_json_from_response(qa_response["messages"][-1].content)
            else:
                qa_data = PlanQAOutput.model_validate(qa_data).model_dump()
            passed = qa_data.get("overall_decision") == "approved"
            feedback = qa_data.get("feedback", "")
            if not isinstance(feedback, str):
                feedback = json.dumps(feedback, ensure_ascii=True)

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
                run_id=state.plan_task.run_id,
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
                run_id=state.plan_task.run_id,
            )

            return {"plan_task": updated_plan_task}

    async def agent_generator_node(self, state: GraphState | dict[str, Any]) -> dict[str, Any]:
        """
        Agent generator creates system prompts and selects tools for each agent.
        Receives a single task via Send (always as dict).
        """
        state_dict = state.model_dump() if isinstance(state, GraphState) else state
        plan = [AgentCreationTask(**t) if isinstance(t, dict) else t for t in state_dict.get("plan", [])]
        full_plan = [AgentCreationTask(**t) if isinstance(t, dict) else t for t in state_dict.get("full_plan", plan)]
        task_results = [AgentCreationTaskResult(**r) if isinstance(r, dict) else r for r in state_dict.get("task_results", [])]
        user_request = state_dict.get("user_request", "")
        plan_run_id = state_dict.get("plan_run_id", "")

        task = plan[0]
        task_id = task.id
        task_name = task.name
        task_purpose = task.purpose

        _, feedback, retry_count = self._find_previous_result(task_results, task_id, plan_run_id=plan_run_id or None)

        if feedback:
            self.logger.info(f"RE-GENERATING agent: {task_name} (ID: {task_id}, Attempt #{retry_count + 1})")
            self.logger.warn(f"Previous QA feedback: {feedback}")
        else:
            self.logger.info(f"Generating agent: {task_name} (ID: {task_id})")

        agent_generator = self.agents["agent_generator"]
        base_query = f"""Agent Name: {task_name}
Agent Purpose: {task_purpose}
User's Original Request: {user_request}
Is Manager: {task.is_manager}
Agent Tags: {", ".join(task.tags)}

You must call the list_available_tools tool with the agent tags to fetch filtered tools. It returns JSON:
{{"tools": [{{"name": "<tool_name>", "description": "<tool_description>", "tags": ["<tag>"], "disallowed_tags": ["<tag>"]}}], "filtered_by_tags": ["<tag>"]}}"""

        if task.is_manager:
            sub_agents = [{"id": t.id, "name": t.name, "purpose": t.purpose} for t in full_plan if not t.is_manager]
            base_query = f"""{base_query}

Team Context:
- You are the manager agent (is_manager=true).
- Sub-agents available to you:
{json.dumps(sub_agents, indent=2)}"""

        if feedback:
            generator_query = f"""{base_query}

Previous QA Feedback: {feedback}

Please regenerate the agent configuration addressing the feedback above.
Return a structured response matching the configured schema."""
        else:
            generator_query = f"""{base_query}

Please create a complete agent configuration.
Return a structured response matching the configured schema."""

        response = await agent_generator.ask_raw(generator_query)
        raw_messages = response.get("messages", [])
        if raw_messages:
            self.logger.debug(f"Agent generator response for {task_name}: {raw_messages[-1].content[:200]}")

        try:
            config_data = self._extract_structured_response(response)
            if config_data is None:
                config_data = extract_json_from_response(response["messages"][-1].content)
            else:
                config_data = AgentGeneratorOutput.model_validate(config_data).model_dump()

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

            if config_data.get("agent_id") != task_id:
                self.logger.warn(f"Normalizing agent_id for {task_name}: {config_data.get('agent_id')} -> {task_id}")
                config_data["agent_id"] = task_id

            task_result = AgentCreationTaskResult(
                task_id=task_id,
                name=task_name,
                output=json.dumps(config_data, indent=2),
                passed_qa=False,
                feedback="",
                retry_count=retry_count,
                plan_run_id=plan_run_id,
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
                plan_run_id=plan_run_id,
            )

            return {"task_results": [task_result]}

    async def qa_node(self, state: GraphState | dict[str, Any]) -> dict[str, Any]:
        """
        QA reviews ONE task result and validates the generated agent configuration.
        Receives a single result via Send (always as dict).
        Returns TaskResult with passed_qa=True/False and feedback.
        """
        state_dict = state.model_dump() if isinstance(state, GraphState) else state
        task_results = [AgentCreationTaskResult(**r) if isinstance(r, dict) else r for r in state_dict.get("task_results", [])]
        plan = [AgentCreationTask(**t) if isinstance(t, dict) else t for t in state_dict.get("plan", [])]
        user_request = state_dict.get("user_request", "")

        result = task_results[0]
        task_id = result.task_id

        self.logger.info(f"QA reviewing agent: {result.name} (ID: {task_id}, Attempt #{result.retry_count + 1})")

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
            config_tools = config_data.get("tools", [])

            if original_task and original_task.is_manager and config_tools:
                feedback = "Manager agent must not use tools directly. Set tools to an empty list for orchestration-only managers."
                updated_result = AgentCreationTaskResult(
                    task_id=task_id,
                    name=result.name,
                    output=result.output,
                    passed_qa=False,
                    feedback=feedback,
                    retry_count=result.retry_count,
                )
                return {"task_results": [updated_result]}

            if original_task and not original_task.tags:
                feedback = "Agent tags are required and must not be empty."
                updated_result = AgentCreationTaskResult(
                    task_id=task_id,
                    name=result.name,
                    output=result.output,
                    passed_qa=False,
                    feedback=feedback,
                    retry_count=result.retry_count,
                )
                return {"task_results": [updated_result]}

            other_agents_context = []
            latest_results = self._get_latest_results(task_results)
            other_tool_names: set[str] = set()

            for other_result in latest_results.values():
                if other_result.task_id != task_id and not other_result.output.startswith("ERROR:"):
                    try:
                        other_config = json.loads(other_result.output)
                        other_tools = other_config.get("tools", [])
                        other_task = self._find_task_by_id(plan, other_result.task_id)
                        other_purpose = other_task.purpose if other_task else "N/A"

                        tools_str = ", ".join(other_tools) if other_tools else "None"
                        other_agents_context.append(f"- {other_result.name}: {other_purpose}\n  Tools: {tools_str}")
                        other_tool_names.update(other_tools)
                    except (json.JSONDecodeError, KeyError):
                        continue

            if original_task and config_tools:
                agent_tags = set(original_task.tags)
                for tool_name in config_tools:
                    tool_cfg = TOOLS_CONFIG.raw(tool_name)
                    tool_tags = set(tool_cfg.tags)
                    disallowed_tags = set(tool_cfg.disallowed_tags)

                    if disallowed_tags.intersection(agent_tags):
                        feedback = f"Tool tag conflict: '{tool_name}' is disallowed for tags {sorted(disallowed_tags.intersection(agent_tags))}."
                        updated_result = AgentCreationTaskResult(
                            task_id=task_id,
                            name=result.name,
                            output=result.output,
                            passed_qa=False,
                            feedback=feedback,
                            retry_count=result.retry_count,
                        )
                        return {"task_results": [updated_result]}

                    if tool_tags and not tool_tags.intersection(agent_tags):
                        feedback = f"Tool tag mismatch: '{tool_name}' tags {sorted(tool_tags)} do not overlap agent tags {sorted(agent_tags)}."
                        updated_result = AgentCreationTaskResult(
                            task_id=task_id,
                            name=result.name,
                            output=result.output,
                            passed_qa=False,
                            feedback=feedback,
                            retry_count=result.retry_count,
                        )
                        return {"task_results": [updated_result]}

            if original_task and config_tools and other_tool_names:
                purpose_lower = original_task.purpose.lower()
                for tool_name in config_tools:
                    if tool_name in other_tool_names:
                        tokens = [token for token in tool_name.lower().split("_") if token]
                        if tokens and not any(token in purpose_lower for token in tokens):
                            feedback = (
                                f"Tool redundancy detected: '{tool_name}' is already used by "
                                "another agent, and this agent's purpose does not indicate "
                                "a need for it. Remove the tool or justify with a distinct purpose."
                            )
                            updated_result = AgentCreationTaskResult(
                                task_id=task_id,
                                name=result.name,
                                output=result.output,
                                passed_qa=False,
                                feedback=feedback,
                                retry_count=result.retry_count,
                            )
                            return {"task_results": [updated_result]}

            team_context = "\n".join(other_agents_context) if other_agents_context else "No other agents in the team yet."

            manager_expectations = ""
            if original_task and original_task.is_manager:
                expected_subagents = [t.id for t in plan if not t.is_manager]
                manager_expectations = f"""
Manager Requirements:
- This agent is the manager and must include sub_agents for all non-manager agents.
- Expected sub_agents list (ids only):
{json.dumps(expected_subagents, indent=2)}
"""

            qa_query = f"""Original User Request: {user_request}

Agent Under Review:
- Name: {result.name}
- Purpose: {original_task.purpose if original_task else "N/A"}
- Is Manager: {original_task.is_manager if original_task else False}
- Tags: {", ".join(original_task.tags) if original_task else "N/A"}

Other Agents in the Team:
{team_context}

You must call the list_all_tools tool to fetch available tools and their tags. It returns JSON:
{{"tools": [{{"name": "<tool_name>", "description": "<tool_description>", "tags": ["<tag>"], "disallowed_tags": ["<tag>"]}}]}}

{manager_expectations}

Generated Configuration for {result.name}:
{json.dumps(config_data, indent=2)}

Please review this agent configuration and provide an audit report.
Use overall_decision with one of: approved, needs_revision.
Return a structured response matching the configured schema."""

            qa_response = await qa_agent.ask_raw(qa_query)
            raw_messages = qa_response.get("messages", [])
            if raw_messages:
                self.logger.debug(f"QA response for {result.name}: {raw_messages[-1]}")

            qa_data = self._extract_structured_response(qa_response)
            if qa_data is None:
                qa_data = extract_json_from_response(qa_response["messages"][-1].content)
            else:
                qa_data = AgentQAOutput.model_validate(qa_data).model_dump()

            passed = qa_data.get("overall_decision") == "approved"
            feedback = qa_data.get("feedback", "")
            if not isinstance(feedback, str):
                feedback = json.dumps(feedback, ensure_ascii=True)

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

    async def team_qa_node(self, state: GraphState) -> dict[str, Any]:
        """Team QA reviews all generated agents at once and provides routing signals."""
        self.logger.info("Team QA reviewing all generated agents...")

        if not state.plan_task or not state.plan_task.plan:
            self.logger.warn("No plan available for team QA review.")
            team_qa = TeamQAOutput(
                overall_decision="needs_revision",
                plan_level_issues=[
                    TeamQAPlanIssue(
                        issue_type="workflow_gap",
                        description="No plan or agents available for review.",
                        severity="critical",
                    )
                ],
                agent_reviews=[],
                summary="Team QA could not run due to missing plan.",
            ).model_dump()
            return {"team_qa": team_qa}

        plan_run_id = state.plan_task.run_id
        latest_results = self._get_latest_results(state.task_results, plan_run_id=plan_run_id)

        agent_payloads: list[dict[str, Any]] = []
        generation_errors: list[tuple[AgentCreationTask, str]] = []

        for task in state.plan_task.plan:
            result = latest_results.get(task.id)
            if not result:
                generation_errors.append((task, "Missing generated configuration."))
                continue
            if result.output.startswith("ERROR:"):
                generation_errors.append((task, result.output))
                continue

            config_data = self._safe_json_loads(result.output)
            if not isinstance(config_data, dict):
                generation_errors.append((task, "Invalid JSON in generated configuration."))
                continue

            agent_payloads.append(
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "agent_purpose": task.purpose,
                    "is_manager": task.is_manager,
                    "tags": task.tags,
                    "config": config_data,
                }
            )

        if generation_errors:
            reviews = []
            for task, error in generation_errors:
                reviews.append(
                    {
                        "task_id": task.id,
                        "agent_id": task.id,
                        "agent_name": task.name,
                        "decision": "needs_revision",
                        "issues": [
                            {
                                "issue": "generation_error",
                                "severity": "critical",
                                "suggestion": str(error),
                            }
                        ],
                        "feedback": f"Generation error: {error}",
                    }
                )

            team_qa = TeamQAOutput(
                overall_decision="needs_revision",
                plan_level_issues=[],
                agent_reviews=reviews,
                summary="One or more agent configurations failed generation.",
            ).model_dump()
            updated_results = self._build_team_review_results(
                plan=state.plan_task.plan,
                latest_results=latest_results,
                team_qa=team_qa,
                plan_run_id=plan_run_id,
            )
            return {"team_qa": team_qa, "task_results": updated_results}

        qa_agent = self.agents["qa"]
        qa_prompt = f"""Original User Request:
{state.user_request}

Planned Agents (task_id, name, purpose, is_manager, tags):
{json.dumps([{"task_id": t.id, "name": t.name, "purpose": t.purpose, "is_manager": t.is_manager, "tags": t.tags} for t in state.plan_task.plan], indent=2)}

Generated Agent Configurations:
{json.dumps(agent_payloads, indent=2)}

You must call the list_all_tools tool to fetch available tools and their tags. It returns JSON:
{{"tools": [{{"name": "<tool_name>", "description": "<tool_description>", "tags": ["<tag>"], "disallowed_tags": ["<tag>"]}}]}}

Review the entire team and provide a structured response matching the configured schema."""

        qa_response = await qa_agent.ask_raw(qa_prompt)
        raw_messages = qa_response.get("messages", [])
        if raw_messages:
            self.logger.debug(f"Team QA response: {raw_messages[-1]}")

        qa_data = self._extract_structured_response(qa_response)
        if qa_data is None:
            qa_data = extract_json_from_response(qa_response["messages"][-1].content)
        else:
            qa_data = TeamQAOutput.model_validate(qa_data).model_dump()

        if isinstance(qa_data, dict) and "feedback" in qa_data and isinstance(qa_data.get("feedback"), dict):
            feedback = qa_data["feedback"]
            if "plan_level_issues" in feedback or "agent_reviews" in feedback:
                qa_data = {
                    "overall_decision": qa_data.get("overall_decision", "needs_revision"),
                    **feedback,
                }

        plan_level_issues = qa_data.get("plan_level_issues") or []
        agent_reviews = qa_data.get("agent_reviews") or []
        reviewed_task_ids = {r.get("task_id") for r in agent_reviews if isinstance(r, dict)}
        for task in state.plan_task.plan:
            if task.id in reviewed_task_ids:
                continue
            agent_reviews.append(
                {
                    "task_id": task.id,
                    "agent_id": task.id,
                    "agent_name": task.name,
                    "decision": "needs_revision",
                    "issues": [
                        {
                            "issue": "missing_review",
                            "severity": "high",
                            "suggestion": "Provide an explicit review for this agent.",
                        }
                    ],
                    "feedback": "Missing QA review for this agent.",
                }
            )
        qa_data["agent_reviews"] = agent_reviews
        updated_results = self._build_team_review_results(
            plan=state.plan_task.plan,
            latest_results=latest_results,
            team_qa=qa_data,
            plan_run_id=plan_run_id,
        )

        updated_plan_task = None
        if plan_level_issues:
            updated_plan_task = PlanTask(
                plan=state.plan_task.plan,
                plan_json=state.plan_task.plan_json,
                passed_qa=False,
                feedback=json.dumps(plan_level_issues, ensure_ascii=True),
                retry_count=state.plan_task.retry_count,
                run_id=str(uuid.uuid4()),
            )

        # Persist approved agents only if there are no plan-level issues.
        if not plan_level_issues:
            for task in state.plan_task.plan:
                review = next(
                    (r for r in qa_data.get("agent_reviews", []) if r.get("task_id") == task.id),
                    None,
                )
                if review and review.get("decision") == "approved":
                    result = latest_results.get(task.id)
                    config_data = self._safe_json_loads(result.output) if result else None
                    if isinstance(config_data, dict):
                        await self._persist_agent_config(config_data=config_data, task=task)

        payload: dict[str, Any] = {"team_qa": qa_data, "task_results": updated_results}
        if updated_plan_task:
            payload["plan_task"] = updated_plan_task
        return payload

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
            self.logger.warn(f"Plan QA failed (attempt #{state.plan_task.retry_count + 1}). Retrying planner...")
            return "planner"
        else:
            self.logger.error(f"Plan QA failed after {MAX_RETRIES} retries. Proceeding with current plan anyway..")
            return "agent_generator"

    async def _qa_review_barrier(self, state: GraphState) -> dict[str, Any]:  # noqa: ARG001
        latest_results = self._get_latest_results(state.task_results)
        self.logger.debug(f"QA review - all {len(latest_results)} QA nodes completed, proceeding to routing")
        return {}

    async def _generation_barrier(self, state: GraphState) -> dict[str, Any]:  # noqa: ARG001
        if not state.plan_task:
            return {}
        plan_run_id = state.plan_task.run_id
        latest_results = self._get_latest_results(state.task_results, plan_run_id=plan_run_id)
        self.logger.debug(
            f"Generation barrier - received {len(latest_results)}/{len(state.plan_task.plan)} agent configs",
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

        self.logger.info(f"[ROUTING] Total: {len(results)}, Passed: {len(passed)}, Failed: {len(failed)}")
        for r in results:
            self.logger.debug(f"  - {r.name}: passed={r.passed_qa}, retry_count={r.retry_count}")

        if failed:
            retriable = [r for r in failed if r.retry_count < MAX_RETRIES]

            if retriable:
                self.logger.info(f"Routing {len(retriable)} failed agents back to generator (retrying). {len(passed)} agents already passed and won't be retried.")

                if not state.plan_task or not state.plan_task.plan:
                    self.logger.error("No plan available for retry")
                    return "aggregator"

                retry_sends = []
                for r in retriable:
                    original_task = self._find_task_by_id(state.plan_task.plan, r.task_id)
                    if not original_task:
                        self.logger.error(f"Could not find original task for {r.task_id}, skipping retry")
                        continue

                    retry_sends.append(
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

    def _route_after_generation(self, state: GraphState) -> str:
        """Routes after each agent generation; proceeds to team QA once all agents are generated."""
        if not state.plan_task or not state.plan_task.plan:
            self.logger.error("No plan available after generation")
            return "aggregator"

        plan_run_id = state.plan_task.run_id
        latest_results = self._get_latest_results(state.task_results, plan_run_id=plan_run_id)
        if len(latest_results) < len(state.plan_task.plan):
            return END
        return "team_qa"

    def _route_after_team_qa(self, state: GraphState) -> list[Send] | str:
        """Routes after team QA based on plan-level vs agent-level issues."""
        if not state.plan_task or not state.plan_task.plan:
            self.logger.error("No plan available for team QA routing")
            return "aggregator"

        team_qa = state.team_qa or {}
        plan_level_issues = team_qa.get("plan_level_issues") or []
        agent_reviews = team_qa.get("agent_reviews") or []

        if plan_level_issues:
            blocking_issue_types = {
                "missing_agent",
                "coverage_gap",
                "role_overlap_requires_split",
                "dependency_mismatch",
                "workflow_gap",
            }
            blocking_severities = {"high", "critical"}
            has_blocking = any(isinstance(issue, dict) and issue.get("issue_type") in blocking_issue_types and issue.get("severity", "medium") in blocking_severities for issue in plan_level_issues)
            if has_blocking and state.plan_task.retry_count < MAX_RETRIES:
                self.logger.warn("Team QA found plan-level issues. Re-routing to planner for a new plan.")
                return "planner"
            if has_blocking:
                self.logger.error("Plan-level issues exceeded retry limit. Moving to aggregator.")
                return "aggregator"

        if agent_reviews:
            latest_results = self._get_latest_results(state.task_results, plan_run_id=state.plan_task.run_id)
            failed = []
            for review in agent_reviews:
                issues = review.get("issues") or []
                blocking = any(isinstance(issue, dict) and issue.get("severity") in {"high", "critical"} for issue in issues)
                if review.get("decision") != "approved" and blocking:
                    failed.append(review)
            if failed:
                retriable: list[Send] = []
                for review in failed:
                    task_id = review.get("task_id")
                    task = self._find_task_by_id(state.plan_task.plan, task_id) if task_id else None
                    if not task:
                        continue
                    prev = latest_results.get(task.id)
                    retry_count = prev.retry_count if prev else 0
                    if retry_count >= MAX_RETRIES:
                        continue
                    retriable.append(
                        Send(
                            "agent_generator",
                            {
                                "plan": [task],
                                "full_plan": state.plan_task.plan,
                                "user_request": state.user_request,
                                "task_results": state.task_results,
                                "plan_run_id": state.plan_task.run_id,
                            },
                        )
                    )
                if retriable:
                    self.logger.info(f"Routing {len(retriable)} failed agents back to generator (retrying).")
                    return retriable
                self.logger.warn("Failed agents exceeded retry limit. Moving to aggregator.")
                return "aggregator"

        self.logger.info("Team QA approved all agents. Moving to aggregator.")
        return "aggregator"

    async def aggregator_node(self, state: GraphState) -> dict[str, Any]:
        """Aggregates all task results (passed and failed) into final output."""
        plan_run_id = state.plan_task.run_id if state.plan_task else None
        latest_results = self._get_latest_results(state.task_results, plan_run_id=plan_run_id)
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
                except Exception as exc:
                    self.logger.error(f"Failed to persist team config: {exc}")
            else:
                self.logger.warn("Skipping team persistence: manager agent not persisted.")

        return {"final_output": final}

    def build_graph(self) -> CompiledStateGraph:
        """Build and compile the agent scaffolding graph."""
        builder = StateGraph(GraphState)

        builder.add_node("planner", self.planner_node)
        builder.add_node("plan_qa", self.plan_qa_node)
        builder.add_node("agent_generator", self.agent_generator_node)
        builder.add_node("generation_barrier", self._generation_barrier)
        builder.add_node("team_qa", self.team_qa_node)
        builder.add_node("aggregator", self.aggregator_node)

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
                        self.logger.error(f"Invalid task in plan: id={task.id}, name={task.name}, purpose={task.purpose}")
                        return "aggregator"

                return [
                    Send(
                        "agent_generator",
                        {
                            "plan": [task],
                            "full_plan": state.plan_task.plan,
                            "user_request": state.user_request,
                            "task_results": [],
                            "plan_run_id": state.plan_task.run_id,
                        },
                    )
                    for task in state.plan_task.plan
                ]
            return route

        builder.add_conditional_edges("plan_qa", route_after_plan_qa)

        builder.add_edge("agent_generator", "generation_barrier")
        builder.add_conditional_edges("generation_barrier", self._route_after_generation)
        builder.add_conditional_edges("team_qa", self._route_after_team_qa)
        builder.add_edge("aggregator", END)

        return builder.compile()

    async def run(self, user_request: str) -> str:
        self._team_id = uuid.uuid4()
        self._team_user_request = user_request
        self._persisted_agent_ids = []
        self._manager_agent_config_id = None
        self._persisted_task_ids = set()

        self.logger.info("=" * 70)
        self.logger.info("DYNAMIC AGENT SCAFFOLDING WITH QA FEEDBACK LOOP")
        self.logger.info("=" * 70)
        self.logger.info(f"User Request: {user_request}")
        self.logger.info("")

        graph = self.build_graph()

        initial_state = GraphState(
            user_request=user_request,
            plan_task=None,
            task_results=[],
            team_qa=None,
            final_output="",
        )

        result = await graph.ainvoke(initial_state.model_dump(), {"recursion_limit": 100})

        self.logger.info("=" * 70)
        self.logger.info("FINAL OUTPUT:")
        self.logger.info("=" * 70)
        self.logger.info(result["final_output"])

        final_state = GraphState(**result)
        final_plan_run_id = final_state.plan_task.run_id if final_state.plan_task else None
        latest_results = self._get_latest_results(final_state.task_results, plan_run_id=final_plan_run_id)
        all_results = list(latest_results.values())

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("EXECUTION SUMMARY:")
        self.logger.info("=" * 70)
        self.logger.info(f"Total agents: {len(all_results)}")
        self.logger.info(f"Passed QA: {sum(1 for r in all_results if r.passed_qa)}")
        self.logger.info(f"Failed QA: {sum(1 for r in all_results if not r.passed_qa)}")

        markdown_report = self._render_markdown_report(
            user_request=self._team_user_request,
            plan=final_state.plan_task.plan if final_state.plan_task else [],
            results=all_results,
        )
        self._write_markdown_report(markdown_report)

        return result["final_output"]
