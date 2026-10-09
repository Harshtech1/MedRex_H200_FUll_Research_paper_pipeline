"""Strictly opt-in adapter for a future controller invocation boundary.

Nothing is invoked on import or construction.  ``invoke_once`` is the sole
execution method, consumes its preparation before calling the injected callable,
and has no default/live controller implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .locked_orchestrator import AttemptPreparation
from .runner_contract import AttemptStatus, BenchmarkRequest, ControllerAttempt


class ControllerInvocationError(RuntimeError):
    """Invalid preparation, duplicate invocation, or invalid callable output."""


class ControllerCallableError(ControllerInvocationError):
    """The injected callable raised; no retry or result substitution occurs."""


@dataclass(frozen=True)
class ControllerInvocationInput:
    """Exact fields allowed across the future controller boundary."""

    run_id: str
    dataset_revision: str
    schema_version: str
    provider: str
    model: str
    temperature: float
    top_p: float
    question_id: str
    full_question_id: str
    case_id: str
    question: str
    categories: str
    sections: str
    question_type: str
    image_ids: tuple[str, ...]


class ControllerInvocationCallable(Protocol):
    def __call__(self, input: ControllerInvocationInput) -> ControllerAttempt: ...


class ControllerInvocationAdapter:
    """Validate and explicitly invoke one injected, caller-owned callable.

    The adapter neither creates clients nor persists output.  A future caller
    must pass the returned attempt back to ``LockedRunOrchestrator.record_outcome``.
    """

    def __init__(self, preparation: AttemptPreparation, invocation: ControllerInvocationCallable) -> None:
        self.preparation = _validate_preparation(preparation)
        if not callable(invocation):
            raise ControllerInvocationError("An explicit invocation callable is required.")
        self._invocation = invocation
        self._consumed = False

    @property
    def input(self) -> ControllerInvocationInput:
        """Validated, path-free controller input; retrieving it does not invoke."""
        request = self.preparation.request
        question = request.question
        config = request.controller_config
        return ControllerInvocationInput(
            request.run_id, request.dataset_revision, request.schema_version,
            config.provider.value, config.model, config.temperature, config.top_p,
            question.question_id, question.full_question_id, question.case_id,
            question.question, question.categories, question.sections,
            question.question_type, question.image_ids,
        )

    def invoke_once(self) -> ControllerAttempt:
        """Explicitly invoke exactly once and validate the returned terminal attempt."""
        if self._consumed:
            raise ControllerInvocationError("This preparation has already been invoked; duplicate invocation is rejected.")
        self._consumed = True
        try:
            result = self._invocation(self.input)
        except Exception as exc:
            raise ControllerCallableError("Injected controller callable failed; no retry was attempted.") from exc
        return _validate_result(self.preparation, result)


def _validate_preparation(value: object) -> AttemptPreparation:
    if not isinstance(value, AttemptPreparation) or not isinstance(value.request, BenchmarkRequest):
        raise ControllerInvocationError("A valid orchestration AttemptPreparation is required.")
    try:
        # Reuse the Task 10 factory to validate the opaque attempt identifier
        # without accepting any output or marking the attempt complete.
        ControllerAttempt.from_request(request=value.request, attempt_id=value.attempt_id, status=AttemptStatus.NOT_EXECUTED)
    except Exception as exc:
        raise ControllerInvocationError("Attempt preparation identity is invalid.") from exc
    return value


def _validate_result(preparation: AttemptPreparation, result: object) -> ControllerAttempt:
    if not isinstance(result, ControllerAttempt):
        raise ControllerInvocationError("Invocation result must be a ControllerAttempt.")
    if result.request_identity != preparation.request.identity or result.attempt_id != preparation.attempt_id:
        raise ControllerInvocationError("Invocation result identity does not match its preparation.")
    if result.status is AttemptStatus.NOT_EXECUTED:
        raise ControllerInvocationError("Invocation result must have a terminal attempt status.")
    try:
        canonical = ControllerAttempt.from_request(
            request=preparation.request, attempt_id=preparation.attempt_id,
            status=result.status, raw_response=result.raw_response, error_code=result.error_code,
        )
    except Exception as exc:
        raise ControllerInvocationError("Invocation result does not satisfy the controller-attempt contract.") from exc
    if canonical != result:
        raise ControllerInvocationError("Invocation result contains inconsistent fields.")
    return result
