"""Non-executing state transitions between checkpoint and controller contracts.

Transition table (all omitted transitions are rejected):

* run: ``new -> initialized -> running -> completed|interrupted``;
* question: ``pending -> attempt_started -> completed|failed|interrupted``;
* recovery: ``attempt_started -> interrupted``;
* retry: ``failed -> retry_eligible -> attempt_started`` only after explicit
  ``authorize_retry``; this module never performs a retry;
* an attempt can finish as completed, execution_error, invalid_response, or
  interrupted.  A failed/interrupted attempt never becomes a completion.

``RunCheckpoint`` is intentionally only a completed/failed *question*
projection.  It cannot represent revision/config identity, attempts, or an
interrupted state.  ``TransitionSnapshot`` retains those fields and is written
atomically by ``AtomicTransitionSnapshotStore``; legacy projection remains
separate so its narrower semantics cannot discard richer state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping

from .run_state import CHECKPOINT_SCHEMA_VERSION, CheckpointValidationError, RunCheckpoint
from .runner_contract import AttemptStatus, BenchmarkContractError, BenchmarkRequest, ControllerAttempt


TRANSITION_SNAPSHOT_SCHEMA_VERSION = 1


class TransitionPlanError(ValueError):
    """Illegal lifecycle, identity, or serialized transition-plan state."""


class RunLifecycle(StrEnum):
    NEW = "new"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class QuestionLifecycle(StrEnum):
    PENDING = "pending"
    ATTEMPT_STARTED = "attempt_started"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    RETRY_ELIGIBLE = "retry_eligible"


@dataclass(frozen=True)
class RunIdentity:
    """Non-secret identity that must agree for all requests and snapshots."""

    run_id: str
    dataset_revision: str
    controller_config_fingerprint: str
    request_schema_version: str
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION

    @classmethod
    def from_request(cls, request: BenchmarkRequest) -> "RunIdentity":
        if not isinstance(request, BenchmarkRequest):
            raise TransitionPlanError("A validated benchmark request is required.")
        config = request.controller_config
        config_bytes = json.dumps(
            {"model": config.model, "provider": config.provider.value, "temperature": config.temperature, "top_p": config.top_p},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return cls(request.run_id, request.dataset_revision, hashlib.sha256(config_bytes).hexdigest(), request.schema_version)

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.run_id, self.dataset_revision, self.controller_config_fingerprint, self.request_schema_version)):
            raise TransitionPlanError("Run identity fields must be non-empty strings.")
        if self.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise TransitionPlanError("Unsupported checkpoint schema version.")


@dataclass(frozen=True)
class QuestionState:
    question_id: str
    case_id: str
    lifecycle: QuestionLifecycle
    attempt_count: int = 0
    active_attempt_id: str | None = None


@dataclass(frozen=True)
class AttemptState:
    """Attempt metadata absent from Task 10's terminal ControllerAttempt."""

    attempt_id: str
    question_id: str
    case_id: str
    attempt_number: int
    outcome: AttemptStatus | None = None
    raw_response: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class RunTransitionPlan:
    """Immutable transition state; methods return a new plan and execute nothing."""

    identity: RunIdentity
    lifecycle: RunLifecycle = RunLifecycle.NEW
    questions: Mapping[str, QuestionState] = MappingProxyType({})
    attempts: Mapping[str, AttemptState] = MappingProxyType({})

    @classmethod
    def new(cls, identity: RunIdentity) -> "RunTransitionPlan":
        return cls(identity)

    def initialize(self, requests: Iterable[BenchmarkRequest]) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.NEW:
            raise TransitionPlanError("Only a new run can be initialized.")
        states: dict[str, QuestionState] = {}
        for request in requests:
            self._validate_request(request)
            question = request.question
            if question.question_id in states:
                raise TransitionPlanError("Duplicate question identity in run initialization.")
            states[question.question_id] = QuestionState(question.question_id, question.case_id, QuestionLifecycle.PENDING)
        if not states:
            raise TransitionPlanError("Run initialization requires at least one request.")
        return self._replace(lifecycle=RunLifecycle.INITIALIZED, questions=states)

    def start(self) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.INITIALIZED:
            raise TransitionPlanError("Only an initialized run can start.")
        return self._replace(lifecycle=RunLifecycle.RUNNING)

    def start_attempt(self, request: BenchmarkRequest, attempt_id: str) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.RUNNING:
            raise TransitionPlanError("Attempts require a running run.")
        self._validate_request(request)
        question = self.questions.get(request.question.question_id)
        if question is None or question.case_id != request.question.case_id:
            raise TransitionPlanError("Request question identity is not in this run.")
        if question.lifecycle not in {QuestionLifecycle.PENDING, QuestionLifecycle.RETRY_ELIGIBLE}:
            raise TransitionPlanError("Question is not eligible to start an attempt.")
        if not _safe_id(attempt_id) or attempt_id in self.attempts:
            raise TransitionPlanError("Attempt identifier is invalid or already exists.")
        attempts = dict(self.attempts)
        attempts[attempt_id] = AttemptState(attempt_id, question.question_id, question.case_id, question.attempt_count + 1)
        questions = dict(self.questions)
        questions[question.question_id] = QuestionState(question.question_id, question.case_id, QuestionLifecycle.ATTEMPT_STARTED, question.attempt_count + 1, attempt_id)
        return self._replace(questions=questions, attempts=attempts)

    def record_attempt(self, attempt: ControllerAttempt) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.RUNNING or not isinstance(attempt, ControllerAttempt):
            raise TransitionPlanError("A running plan and validated controller attempt are required.")
        run_id, revision, question_id, case_id = attempt.request_identity
        if (run_id, revision) != (self.identity.run_id, self.identity.dataset_revision):
            raise TransitionPlanError("Controller attempt run identity does not match this run.")
        question = self.questions.get(question_id)
        active = self.attempts.get(attempt.attempt_id)
        if question is None or question.case_id != case_id or question.active_attempt_id != attempt.attempt_id or active is None:
            raise TransitionPlanError("Controller attempt does not match an active question attempt.")
        if attempt.status is AttemptStatus.NOT_EXECUTED:
            raise TransitionPlanError("A non-executed attempt cannot finalize an active attempt.")
        lifecycle = {
            AttemptStatus.COMPLETED: QuestionLifecycle.COMPLETED,
            AttemptStatus.EXECUTION_ERROR: QuestionLifecycle.FAILED,
            AttemptStatus.INVALID_RESPONSE: QuestionLifecycle.FAILED,
            AttemptStatus.INTERRUPTED: QuestionLifecycle.INTERRUPTED,
        }[attempt.status]
        attempts = dict(self.attempts)
        attempts[attempt.attempt_id] = AttemptState(active.attempt_id, active.question_id, active.case_id, active.attempt_number, attempt.status, attempt.raw_response, attempt.error_code)
        questions = dict(self.questions)
        questions[question_id] = QuestionState(question_id, case_id, lifecycle, question.attempt_count)
        return self._replace(questions=questions, attempts=attempts)

    def recover_interrupted(self) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.RUNNING:
            raise TransitionPlanError("Recovery requires a running run.")
        questions, attempts = dict(self.questions), dict(self.attempts)
        for question_id, question in self.questions.items():
            if question.lifecycle is QuestionLifecycle.ATTEMPT_STARTED:
                active = attempts[question.active_attempt_id or ""]
                attempts[active.attempt_id] = AttemptState(active.attempt_id, active.question_id, active.case_id, active.attempt_number, AttemptStatus.INTERRUPTED)
                questions[question_id] = QuestionState(question.question_id, question.case_id, QuestionLifecycle.INTERRUPTED, question.attempt_count)
        return self._replace(lifecycle=RunLifecycle.INTERRUPTED, questions=questions, attempts=attempts)

    def authorize_retry(self, question_id: str) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.RUNNING:
            raise TransitionPlanError("Retry authorization requires a running run.")
        question = self.questions.get(question_id)
        if question is None or question.lifecycle is not QuestionLifecycle.FAILED:
            raise TransitionPlanError("Only a failed question can be explicitly authorized for retry.")
        questions = dict(self.questions)
        questions[question_id] = QuestionState(question.question_id, question.case_id, QuestionLifecycle.RETRY_ELIGIBLE, question.attempt_count)
        return self._replace(questions=questions)

    def complete(self) -> "RunTransitionPlan":
        if self.lifecycle is not RunLifecycle.RUNNING or any(q.lifecycle is not QuestionLifecycle.COMPLETED for q in self.questions.values()):
            raise TransitionPlanError("Only a running run with every question completed can complete.")
        return self._replace(lifecycle=RunLifecycle.COMPLETED)

    def checkpoint_projection(self) -> RunCheckpoint:
        """Project only representable completed/failed state into Task 08 format."""
        if any(q.lifecycle in {QuestionLifecycle.ATTEMPT_STARTED, QuestionLifecycle.INTERRUPTED, QuestionLifecycle.RETRY_ELIGIBLE} for q in self.questions.values()):
            raise TransitionPlanError("Task 08 checkpoints cannot faithfully represent active, interrupted, or retry-eligible state.")
        completed = frozenset(q.question_id for q in self.questions.values() if q.lifecycle is QuestionLifecycle.COMPLETED)
        failed = frozenset(q.question_id for q in self.questions.values() if q.lifecycle is QuestionLifecycle.FAILED)
        return RunCheckpoint(self.identity.run_id, completed, failed)

    @classmethod
    def from_checkpoint(cls, identity: RunIdentity, questions: Iterable[tuple[str, str]], checkpoint: RunCheckpoint) -> "RunTransitionPlan":
        if not isinstance(checkpoint, RunCheckpoint) or checkpoint.namespace != identity.run_id:
            raise TransitionPlanError("Checkpoint namespace does not match run identity.")
        states: dict[str, QuestionState] = {}
        for question_id, case_id in questions:
            if question_id in states:
                raise TransitionPlanError("Duplicate question identity in checkpoint recovery.")
            lifecycle = QuestionLifecycle.COMPLETED if question_id in checkpoint.completed_case_ids else QuestionLifecycle.FAILED if question_id in checkpoint.failed_case_ids else QuestionLifecycle.PENDING
            states[question_id] = QuestionState(question_id, case_id, lifecycle)
        known = set(states)
        if not checkpoint.completed_case_ids <= known or not checkpoint.failed_case_ids <= known:
            raise TransitionPlanError("Checkpoint references a question outside this run.")
        return cls(identity, RunLifecycle.INITIALIZED, MappingProxyType(states), MappingProxyType({}))

    def snapshot_json(self) -> str:
        """Deterministic richer snapshot for AtomicTransitionSnapshotStore."""
        return json.dumps(
            {
                "attempts": [
                    {"attempt_id": a.attempt_id, "attempt_number": a.attempt_number, "case_id": a.case_id, "error_code": a.error_code, "outcome": a.outcome.value if a.outcome else None, "question_id": a.question_id, "raw_response": a.raw_response}
                    for a in sorted(self.attempts.values(), key=lambda item: item.attempt_id)
                ],
                "identity": self.identity.__dict__,
                "lifecycle": self.lifecycle.value,
                "questions": [
                    {"active_attempt_id": q.active_attempt_id, "attempt_count": q.attempt_count, "case_id": q.case_id, "lifecycle": q.lifecycle.value, "question_id": q.question_id}
                    for q in sorted(self.questions.values(), key=lambda item: item.question_id)
                ],
                "schema_version": TRANSITION_SNAPSHOT_SCHEMA_VERSION,
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )

    @classmethod
    def from_snapshot_json(cls, raw: str, expected_identity: RunIdentity) -> "RunTransitionPlan":
        try:
            payload = json.loads(raw)
            if set(payload) != {"attempts", "identity", "lifecycle", "questions", "schema_version"} or payload["schema_version"] != TRANSITION_SNAPSHOT_SCHEMA_VERSION:
                raise ValueError
            identity = RunIdentity(**payload["identity"])
            if identity != expected_identity:
                raise ValueError
            if not isinstance(payload["questions"], list) or not isinstance(payload["attempts"], list):
                raise ValueError
            questions = {item["question_id"]: QuestionState(item["question_id"], item["case_id"], QuestionLifecycle(item["lifecycle"]), item["attempt_count"], item["active_attempt_id"]) for item in payload["questions"]}
            attempts = {item["attempt_id"]: AttemptState(item["attempt_id"], item["question_id"], item["case_id"], item["attempt_number"], AttemptStatus(item["outcome"]) if item["outcome"] else None, item["raw_response"], item["error_code"]) for item in payload["attempts"]}
            if len(questions) != len(payload["questions"]) or len(attempts) != len(payload["attempts"]):
                raise ValueError
            plan = cls(identity, RunLifecycle(payload["lifecycle"]), MappingProxyType(questions), MappingProxyType(attempts))
            plan._validate_snapshot()
            return plan
        except (KeyError, TypeError, ValueError, BenchmarkContractError, CheckpointValidationError) as exc:
            raise TransitionPlanError("Transition snapshot is malformed, unsupported, or identity-mismatched.") from exc

    def _validate_request(self, request: BenchmarkRequest) -> None:
        if not isinstance(request, BenchmarkRequest) or RunIdentity.from_request(request) != self.identity:
            raise TransitionPlanError("Request run identity or controller configuration does not match this run.")

    def _validate_snapshot(self) -> None:
        for question in self.questions.values():
            if question.lifecycle is QuestionLifecycle.ATTEMPT_STARTED:
                active = self.attempts.get(question.active_attempt_id or "")
                if active is None or active.outcome is not None or active.question_id != question.question_id:
                    raise TransitionPlanError("Transition snapshot has an invalid active attempt.")
            elif question.active_attempt_id is not None:
                raise TransitionPlanError("Transition snapshot retains an inactive attempt reference.")
        for attempt in self.attempts.values():
            if attempt.question_id not in self.questions or self.questions[attempt.question_id].case_id != attempt.case_id:
                raise TransitionPlanError("Transition snapshot attempt identity is inconsistent.")

    def _replace(self, *, lifecycle: RunLifecycle | None = None, questions: Mapping[str, QuestionState] | None = None, attempts: Mapping[str, AttemptState] | None = None) -> "RunTransitionPlan":
        return RunTransitionPlan(self.identity, lifecycle if lifecycle is not None else self.lifecycle, MappingProxyType(dict(questions if questions is not None else self.questions)), MappingProxyType(dict(attempts if attempts is not None else self.attempts)))


def _safe_id(value: object) -> bool:
    return isinstance(value, str) and value.replace("_", "").replace("-", "").isalnum() and bool(value)
