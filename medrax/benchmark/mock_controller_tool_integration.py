"""Mock-only controller-to-tool integration for contract testing.

This module contains no concrete medical-tool imports.  It deliberately keeps
controller decision, trusted preparation, fake execution, and attempt-result
creation as separate steps.  The returned attempt is not persisted here: a
caller must use ``LockedRunOrchestrator.record_outcome`` under its existing lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from medrax.tool_registry import PAPER_TOOL_NAMES

from .controller_invocation_adapter import ControllerInvocationAdapter, ControllerInvocationInput
from .locked_orchestrator import AttemptPreparation
from .runner_contract import AttemptStatus, ControllerAttempt
from .trusted_tool_dispatch import PreparedToolCall, TrustedToolDispatchError, TrustedToolDispatcher


class MockIntegrationError(RuntimeError):
    """Invalid mock integration setup or a consumed mock invocation."""


@dataclass(frozen=True)
class MockToolDecision:
    """A fake-controller decision containing only opaque image identifiers."""

    tool_name: str
    image_ids: tuple[str, ...]
    arguments: Mapping[str, object] = field(repr=False, compare=False)


@dataclass(frozen=True)
class MockToolOutput:
    """Validated fake-tool output; this is not a medical inference result."""

    response: str


class MockController(Protocol):
    def __call__(self, input: ControllerInvocationInput) -> MockToolDecision: ...


class MockTool(Protocol):
    def __call__(self, arguments: Mapping[str, object]) -> MockToolOutput: ...


class MockControllerToolIntegration:
    """Execute exactly one injected fake decision and fake tool for a preparation.

    Invalid decisions become ``INVALID_RESPONSE`` attempts.  Fake controller or
    fake tool exceptions become ``EXECUTION_ERROR`` attempts.  Neither outcome
    retries, chooses a fallback, writes a snapshot, or calls a real tool.
    """

    def __init__(
        self,
        *,
        preparation: AttemptPreparation,
        controller: MockController,
        dispatcher: TrustedToolDispatcher,
        mock_tools: Mapping[str, MockTool],
    ) -> None:
        if not callable(controller) or not isinstance(dispatcher, TrustedToolDispatcher):
            raise MockIntegrationError("An injected fake controller and trusted dispatcher are required.")
        if not isinstance(mock_tools, Mapping) or any(name not in PAPER_TOOL_NAMES or not callable(tool) for name, tool in mock_tools.items()):
            raise MockIntegrationError("Mock tools must be callable members of the six-tool allowlist.")
        # Reuse the public Task 14 constructor solely for preparation validation
        # and the canonical path-free controller input; its placeholder is never run.
        self._controller_input = ControllerInvocationAdapter(preparation, lambda _: None).input
        self._preparation = preparation
        self._controller = controller
        self._dispatcher = dispatcher
        self._mock_tools = MappingProxyType(dict(mock_tools))
        self._consumed = False

    @property
    def controller_input(self) -> ControllerInvocationInput:
        """The safe fake-controller view; accessing it has no side effects."""
        return self._controller_input

    def invoke_once(self) -> ControllerAttempt:
        """Run one mocked integration attempt and return, but never persist, it."""
        if self._consumed:
            raise MockIntegrationError("This mock integration has already been invoked; duplicate invocation is rejected.")
        self._consumed = True
        try:
            decision = self._controller(self._controller_input)
        except Exception:
            return self._execution_error("mock_controller_error")
        try:
            prepared = self._prepare(decision)
        except (MockIntegrationError, TrustedToolDispatchError):
            return self._invalid_response("mock_controller_invalid_decision")
        tool = self._mock_tools.get(prepared.tool_name)
        if tool is None:
            return self._execution_error("mock_tool_unavailable")
        try:
            output = tool(prepared.execution_arguments())
        except Exception:
            return self._execution_error("mock_tool_error")
        if not isinstance(output, MockToolOutput) or not isinstance(output.response, str) or not output.response:
            return self._invalid_response("mock_tool_invalid_output")
        if any(path in output.response for path in _path_values(prepared)):
            return self._invalid_response("mock_tool_path_leak")
        return ControllerAttempt.from_request(
            request=self._preparation.request,
            attempt_id=self._preparation.attempt_id,
            status=AttemptStatus.COMPLETED,
            raw_response=output.response,
        )

    def _prepare(self, decision: object) -> PreparedToolCall:
        if not isinstance(decision, MockToolDecision):
            raise MockIntegrationError("Mock controller returned no valid decision.")
        if not isinstance(decision.tool_name, str) or not isinstance(decision.image_ids, tuple) or not isinstance(decision.arguments, Mapping):
            raise MockIntegrationError("Mock controller decision fields are invalid.")
        if not set(decision.image_ids).issubset(self._preparation.request.question.image_ids):
            raise MockIntegrationError("Decision image identifiers are not authorized for this benchmark question.")
        return self._dispatcher.prepare(
            context=self._preparation.request.case_context,
            tool_name=decision.tool_name,
            image_ids=decision.image_ids,
            arguments=decision.arguments,
        )

    def _invalid_response(self, marker: str) -> ControllerAttempt:
        return ControllerAttempt.from_request(
            request=self._preparation.request,
            attempt_id=self._preparation.attempt_id,
            status=AttemptStatus.INVALID_RESPONSE,
            raw_response=marker,
            error_code="invalid_response",
        )

    def _execution_error(self, code: str) -> ControllerAttempt:
        return ControllerAttempt.from_request(
            request=self._preparation.request,
            attempt_id=self._preparation.attempt_id,
            status=AttemptStatus.EXECUTION_ERROR,
            error_code=code,
        )


def _path_values(prepared: PreparedToolCall) -> tuple[str, ...]:
    """Internal path check for fake output validation; never exposed to controller."""
    values = prepared.execution_arguments()
    paths: list[str] = []
    image_path = values.get("image_path")
    if isinstance(image_path, str):
        paths.append(image_path)
    image_paths = values.get("image_paths")
    if isinstance(image_paths, list):
        paths.extend(value for value in image_paths if isinstance(value, str))
    return tuple(paths)
