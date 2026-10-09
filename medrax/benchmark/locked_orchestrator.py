"""Locked, non-executing orchestration for future benchmark attempts.

No method in this module invokes the injected invocation boundary.  It only
persists lifecycle transitions before and after a future caller-owned boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .run_state import RunNamespaceLock
from .run_transition_plan import RunIdentity, RunLifecycle, RunTransitionPlan, TransitionPlanError
from .runner_contract import AttemptStatus, BenchmarkRequest, ControllerAttempt
from .transition_snapshot_store import AtomicTransitionSnapshotStore


class OrchestrationError(RuntimeError):
    """A lock, identity, or durable-transition precondition failed."""


class FutureInvocationBoundary(Protocol):
    """Reserved injected boundary; Task 13 deliberately never calls it."""

    def __call__(self, request: BenchmarkRequest) -> ControllerAttempt: ...


@dataclass(frozen=True)
class AttemptPreparation:
    """Proof that an attempt-started transition was durably persisted."""

    request: BenchmarkRequest
    attempt_id: str


class LockedRunOrchestrator:
    """Coordinate locks and durable transitions without controller execution.

    Callers must provide an initialized transition plan; this boundary neither
    selects questions nor initializes external clients.  A snapshot write occurs
    before each in-memory plan update becomes visible through ``plan``.
    """

    def __init__(
        self,
        *,
        lock: RunNamespaceLock,
        plan: RunTransitionPlan,
        snapshot_store: AtomicTransitionSnapshotStore,
        invocation_boundary: FutureInvocationBoundary | None = None,
    ) -> None:
        if not isinstance(lock, RunNamespaceLock) or not isinstance(plan, RunTransitionPlan) or not isinstance(snapshot_store, AtomicTransitionSnapshotStore):
            raise OrchestrationError("Existing lock, transition plan, and snapshot store are required.")
        if lock.namespace != plan.identity.run_id or snapshot_store.expected_identity != plan.identity:
            raise OrchestrationError("Lock namespace or snapshot identity does not match the run plan.")
        self.lock = lock
        self.plan = plan
        self.snapshot_store = snapshot_store
        self.invocation_boundary = invocation_boundary

    def preflight(self) -> RunTransitionPlan:
        """Durably persist initialized then running state; never invoke a controller."""
        with self.lock:
            self._validate_identity()
            if self.plan.lifecycle is RunLifecycle.INITIALIZED:
                self._commit(self.plan)
                self._commit(self.plan.start())
            elif self.plan.lifecycle is not RunLifecycle.RUNNING:
                raise OrchestrationError("Preflight requires an initialized or running plan.")
            return self.plan

    def prepare_attempt(self, request: BenchmarkRequest, attempt_id: str) -> AttemptPreparation:
        """Persist an attempt-started state before any future invocation boundary."""
        with self.lock:
            self._validate_request(request)
            if self.plan.lifecycle is RunLifecycle.INITIALIZED:
                self._commit(self.plan)
                self._commit(self.plan.start())
            if self.plan.lifecycle is not RunLifecycle.RUNNING:
                raise OrchestrationError("Attempt preparation requires a running plan.")
            try:
                candidate = self.plan.start_attempt(request, attempt_id)
            except TransitionPlanError as exc:
                raise OrchestrationError("Attempt-started transition is invalid.") from exc
            self._commit(candidate)
            return AttemptPreparation(request, attempt_id)

    def record_outcome(self, attempt: ControllerAttempt) -> RunTransitionPlan:
        """Persist a caller-supplied terminal outcome; no response is deemed valid here."""
        with self.lock:
            self._validate_identity()
            try:
                candidate = self.plan.record_attempt(attempt)
            except TransitionPlanError as exc:
                raise OrchestrationError("Attempt outcome transition is invalid.") from exc
            self._commit(candidate)
            return self.plan

    def record_execution_failure(self, request: BenchmarkRequest, attempt_id: str, error_code: str) -> RunTransitionPlan:
        """Persist an explicit failure supplied by a future invocation owner."""
        attempt = ControllerAttempt.from_request(request=request, attempt_id=attempt_id, status=AttemptStatus.EXECUTION_ERROR, error_code=error_code)
        return self.record_outcome(attempt)

    def recover_interruption(self) -> RunTransitionPlan:
        """Recover in memory only; caller must explicitly persist the returned plan."""
        with self.lock:
            self._validate_identity()
            recovered = self.snapshot_store.recover_after_interruption()
            if recovered.identity != self.plan.identity:
                raise OrchestrationError("Recovered snapshot identity does not match this run.")
            self.plan = recovered
            return recovered

    def persist_recovered(self) -> RunTransitionPlan:
        """Make a previously recovered in-memory interruption state durable."""
        with self.lock:
            self._validate_identity()
            self._commit(self.plan)
            return self.plan

    def _commit(self, candidate: RunTransitionPlan) -> None:
        self.snapshot_store.write(candidate)
        self.plan = candidate

    def _validate_identity(self) -> None:
        if self.lock.namespace != self.plan.identity.run_id or self.snapshot_store.expected_identity != self.plan.identity:
            raise OrchestrationError("Run identity no longer matches injected dependencies.")

    def _validate_request(self, request: BenchmarkRequest) -> None:
        self._validate_identity()
        if not isinstance(request, BenchmarkRequest) or RunIdentity.from_request(request) != self.plan.identity:
            raise OrchestrationError("Request identity does not match this run.")
