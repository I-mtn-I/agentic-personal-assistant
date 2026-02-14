from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from colorama import Fore, Style, init
from langchain_core.callbacks import BaseCallbackHandler


class StreamPrinter(BaseCallbackHandler):
    def __init__(self, agent_name: str, *, stream_color: str = "", tool_color: Optional[str] = None) -> None:
        init(autoreset=False)
        self.agent_name = agent_name
        self.stream_color = stream_color
        self.tool_color = tool_color or Fore.CYAN
        self.seen_token = False
        self._started = False

    def _agent_label(self) -> str:
        if self.stream_color:
            return f"{self.stream_color}[{self.agent_name}]{Style.RESET_ALL} "
        return f"[{self.agent_name}] "

    def _tool_label(self, tool_name: str) -> str:
        return f"{self.tool_color}[tool:{tool_name}]{Style.RESET_ALL} "

    def _ensure_prefix(self) -> None:
        if not self._started:
            heading = self._heading()
            print(heading, end="")
            self._started = True

    def _heading(self) -> str:
        name = self.agent_name
        if self.stream_color:
            name = f"{self.stream_color}{name}{Style.RESET_ALL}"
        line = "-" * max(7, len(self.agent_name))
        return f"\n{line}\n{name}\n{line}\n"

    def on_llm_new_token(self, token: str, **_kwargs: Any) -> None:
        self.seen_token = True
        self._ensure_prefix()
        if self.stream_color:
            print(f"{self.stream_color}{token}{Style.RESET_ALL}", end="")
        else:
            print(token, end="")

    def on_llm_end(self, *args: Any, **_kwargs: Any) -> None:
        if self.seen_token:
            print()

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **_kwargs: Any) -> None:
        tool_name = serialized.get("name", "tool")
        print(f"\n{self._agent_label()}{self._tool_label(tool_name)}{input_str}")

    def on_tool_end(self, output: str, **_kwargs: Any) -> None:
        print(f"{self._agent_label()}{self._tool_label('result')}{output}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: Any | None = None,
        metadata: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        print(f"{self._agent_label()}{self._tool_label('error')}{error}")


@dataclass(frozen=True)
class StreamSession:
    callbacks: list[BaseCallbackHandler]
    handler: StreamPrinter

    def should_print_final(self) -> bool:
        return not self.handler.seen_token


def build_streaming_session(agent_name: str, *, is_subagent: bool) -> StreamSession:
    color = str(Fore.YELLOW) if is_subagent else ""
    handler = StreamPrinter(agent_name, stream_color=color)
    return StreamSession(callbacks=[handler], handler=handler)
