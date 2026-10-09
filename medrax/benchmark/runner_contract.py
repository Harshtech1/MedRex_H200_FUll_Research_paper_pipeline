"""Typed boundaries for a future benchmark runner, without execution logic.

This module has no controller client, model, tool, checkpoint, retry, or scoring
implementation.  It only makes the data hand-offs explicit and serializable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
import re

from medrax.controller_config import ControllerConfig, ControllerProvider

from .chestagentbench_loader import (
    PINNED_REVISION,
    ChestAgentBenchDataset,
    ControllerQuestion,
)
from .image_identifier_adapter import CaseContext


RUNNER_CONTRACT_SCHEMA_VERSION = "medrax-benchmark-runner/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class BenchmarkContractError(ValueError):
    """Raised before a future runner could invoke a controller."""


class AttemptStatus(StrEnum):
    """Terminal and non-terminal states; no scoring implication is attached."""

    NOT_EXECUTED = "not_executed"
    COMPLETED = "completed"
    EXECUTION_ERROR = "execution_error"
    INVALID_RESPONSE = "invalid_response"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, init=False)
class BenchmarkRequest:
    """A loader-issued question plus trusted context and explicit controller config.

    Instances can only be produced by :meth:`from_loader`.  The trusted local
    context is intentionally excluded from controller serialization; only opaque
    image IDs appear in the payload.
    """

    run_id: str
    dataset_revision: str
    schema_version: str
    controller_config: ControllerConfig
    question: ControllerQuestion
    _case_context: CaseContext = field(repr=False, compare=False)

    @classmethod
    def from_loader(
        cls,
        *,
        dataset: ChestAgentBenchDataset,
        question: ControllerQuestion,
        run_id: str,
        dataset_revision: str,
        controller_config: ControllerConfig,
        schema_version: str = RUNNER_CONTRACT_SCHEMA_VERSION,
    ) -> "BenchmarkRequest":
        if not isinstance(dataset, ChestAgentBenchDataset):
            raise BenchmarkContractError("A loaded ChestAgentBenchDataset is required.")
        if not isinstance(question, ControllerQuestion):
            raise BenchmarkContractError("A loader-issued ControllerQuestion is required.")
        _safe_identifier("run ID", run_id)
        if dataset_revision != PINNED_REVISION:
            raise BenchmarkContractError("Dataset revision must equal the pinned benchmark revision.")
        if schema_version != RUNNER_CONTRACT_SCHEMA_VERSION:
            raise BenchmarkContractError("Unsupported benchmark request schema version.")
        config = _validated_config(controller_config)
        try:
            context = dataset.context_for(question)
        except Exception as exc:
            raise BenchmarkContractError("Question was not issued by the supplied dataset load.") from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "run_id", run_id)
        object.__setattr__(instance, "dataset_revision", dataset_revision)
        object.__setattr__(instance, "schema_version", schema_version)
        object.__setattr__(instance, "controller_config", config)
        object.__setattr__(instance, "question", question)
        object.__setattr__(instance, "_case_context", context)
        return instance

    @property
    def case_context(self) -> CaseContext:
        """Trusted dispatch-only context, never derived from controller arguments."""
        return self._case_context

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Stable association: run, revision, question, case."""
        return (self.run_id, self.dataset_revision, self.question.question_id, self.question.case_id)

    def controller_json(self) -> str:
        """Deterministic controller-facing JSON with no local paths or ground truth."""
        payload = {
            "controller": {
                "model": self.controller_config.model,
                "provider": self.controller_config.provider.value,
                "temperature": self.controller_config.temperature,
                "top_p": self.controller_config.top_p,
            },
            "dataset_revision": self.dataset_revision,
            "question": {
                "case_id": self.question.case_id,
                "categories": self.question.categories,
                "full_question_id": self.question.full_question_id,
                "image_ids": list(self.question.image_ids),
                "question": self.question.question,
                "question_id": self.question.question_id,
                "question_type": self.question.question_type,
                "sections": self.question.sections,
            },
            "run_id": self.run_id,
            "schema_version": self.schema_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, init=False)
class ControllerAttempt:
    """Persistable controller-attempt outcome with no evaluation outcome fields."""

    request_identity: tuple[str, str, str, str]
    attempt_id: str
    status: AttemptStatus
    raw_response: str | None
    error_code: str | None

    @classmethod
    def from_request(
        cls,
        *,
        request: BenchmarkRequest,
        attempt_id: str,
        status: AttemptStatus | str,
        raw_response: str | None = None,
        error_code: str | None = None,
    ) -> "ControllerAttempt":
        if not isinstance(request, BenchmarkRequest):
            raise BenchmarkContractError("A validated BenchmarkRequest is required.")
        _safe_identifier("attempt ID", attempt_id)
        try:
            parsed_status = AttemptStatus(status)
        except ValueError as exc:
            raise BenchmarkContractError("Unsupported controller attempt status.") from exc
        _validate_attempt_payload(parsed_status, raw_response, error_code)
        instance = object.__new__(cls)
        object.__setattr__(instance, "request_identity", request.identity)
        object.__setattr__(instance, "attempt_id", attempt_id)
        object.__setattr__(instance, "status", parsed_status)
        object.__setattr__(instance, "raw_response", raw_response)
        object.__setattr__(instance, "error_code", error_code)
        return instance

    def persisted_json(self) -> str:
        """Deterministic record referencing a question, never copying ground truth."""
        run_id, revision, question_id, case_id = self.request_identity
        return json.dumps(
            {
                "attempt_id": self.attempt_id,
                "case_id": case_id,
                "dataset_revision": revision,
                "error_code": self.error_code,
                "question_id": question_id,
                "raw_response": self.raw_response,
                "run_id": run_id,
                "status": self.status.value,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def _safe_identifier(label: str, value: object) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise BenchmarkContractError(f"{label.capitalize()} must be a non-empty path-safe identifier.")
    return value


def _validated_config(config: object) -> ControllerConfig:
    if not isinstance(config, ControllerConfig) or not isinstance(config.provider, ControllerProvider):
        raise BenchmarkContractError("A validated ControllerConfig is required.")
    try:
        normalized = ControllerConfig.from_values(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    except Exception as exc:
        raise BenchmarkContractError("Controller configuration is invalid.") from exc
    if normalized != config:
        raise BenchmarkContractError("Controller configuration must use canonical validated values.")
    return config


def _validate_attempt_payload(status: AttemptStatus, raw_response: str | None, error_code: str | None) -> None:
    if raw_response is not None and (not isinstance(raw_response, str) or not raw_response):
        raise BenchmarkContractError("Raw response must be a non-empty string when present.")
    if error_code is not None and (not isinstance(error_code, str) or not _SAFE_ID.fullmatch(error_code)):
        raise BenchmarkContractError("Error code must be a non-empty path-safe identifier when present.")
    if status is AttemptStatus.COMPLETED:
        valid = raw_response is not None and error_code is None
    elif status is AttemptStatus.INVALID_RESPONSE:
        valid = raw_response is not None and error_code == "invalid_response"
    elif status is AttemptStatus.EXECUTION_ERROR:
        valid = raw_response is None and error_code is not None
    else:
        valid = raw_response is None and error_code is None
    if not valid:
        raise BenchmarkContractError("Attempt status does not match response/error fields.")
