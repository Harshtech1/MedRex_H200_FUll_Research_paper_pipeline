"""CPU-safe registry validation for already-constructed MedRAX tools.

The registry neither imports concrete medical tools nor constructs models. It is an
additive injection seam for validating the six-tool contract with supplied tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel
from langchain_core.tools import BaseTool


PAPER_TOOL_NAMES = (
    "chest_xray_report_generator",
    "chest_xray_classifier",
    "chest_xray_segmentation",
    "xray_phrase_grounding",
    "chest_xray_expert",
    "llava_med_qa",
)


class ToolRegistryError(ValueError):
    """Raised when supplied tools do not satisfy the registry contract."""


@dataclass(frozen=True)
class ToolDispatchResult:
    """The selected tool's result or an explicit safe dispatch failure."""

    tool_name: str
    result: Any | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ToolRegistry:
    """Validate, look up, and dispatch injected LangChain tools by canonical name."""

    def __init__(self, tools: Iterable[BaseTool], *, required_names: Iterable[str] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self._register(tool)

        required = tuple(required_names)
        if required:
            missing = sorted(set(required) - set(self._tools))
            extra = sorted(set(self._tools) - set(required))
            if missing or extra:
                raise ToolRegistryError(
                    "Registry must contain exactly the required canonical tool names."
                )
        elif not self._tools:
            raise ToolRegistryError("Registry must contain at least one tool.")

    @classmethod
    def for_six_paper_tools(cls, tools: Iterable[BaseTool]) -> "ToolRegistry":
        """Build a complete registry for the paper-facing six-tool contract."""
        return cls(tools, required_names=PAPER_TOOL_NAMES)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError("No tool is registered under the requested name.") from exc

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> ToolDispatchResult:
        """Invoke one selected tool without altering its successful result format."""
        tool = self.get(name)
        try:
            return ToolDispatchResult(tool_name=name, result=tool.invoke(dict(arguments)))
        except Exception:
            return ToolDispatchResult(
                tool_name=name,
                result=None,
                error="Selected tool execution failed.",
            )

    def _register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError("Registered tools must implement LangChain BaseTool.")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ToolRegistryError("Every registered tool must have a non-empty name.")
        if not isinstance(tool.description, str) or not tool.description.strip():
            raise ToolRegistryError("Every registered tool must have a non-empty description.")
        schema = tool.args_schema
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise ToolRegistryError("Every registered tool must expose a Pydantic input schema.")
        if tool.name in self._tools:
            raise ToolRegistryError("Duplicate registered tool name.")
        self._tools[tool.name] = tool
