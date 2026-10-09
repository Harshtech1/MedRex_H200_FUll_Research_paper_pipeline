"""Atomic local-filesystem persistence for versioned transition snapshots.

The richer snapshot is authoritative on restart because it retains run identity,
question lifecycle, and attempt history.  Legacy ``RunCheckpoint`` remains a
separate completed/failed projection with deliberately narrower semantics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .run_state import _fsync_directory, _write_temp
from .run_transition_plan import RunIdentity, RunLifecycle, RunTransitionPlan, TransitionPlanError


_CREDENTIAL_MARKER = re.compile(r"(?:api[_-]?key|authorization|bearer\s+|secret|password|token\s*[:=]|sk-[A-Za-z0-9])", re.IGNORECASE)


class TransitionSnapshotStoreError(RuntimeError):
    """Base error for persisted transition snapshots."""


class TransitionSnapshotCorruptError(TransitionSnapshotStoreError):
    """The on-disk JSON is malformed or truncated."""


class TransitionSnapshotValidationError(TransitionSnapshotStoreError):
    """The snapshot is well-formed JSON but invalid for this expected run."""


class AtomicTransitionSnapshotStore:
    """Same-directory temp-write/fsync/replace storage for one run identity.

    This makes no power-loss or distributed-filesystem durability claim beyond
    the local filesystem's rename and fsync behavior.
    """

    def __init__(self, path: str | Path, expected_identity: RunIdentity) -> None:
        if not isinstance(expected_identity, RunIdentity):
            raise TransitionSnapshotValidationError("A validated expected run identity is required.")
        self.path = Path(path)
        self.expected_identity = expected_identity

    def write(self, plan: RunTransitionPlan) -> None:
        if not isinstance(plan, RunTransitionPlan) or plan.identity != self.expected_identity:
            raise TransitionSnapshotValidationError("Snapshot run identity does not match this store.")
        raw = plan.snapshot_json()
        self._validate_raw(raw)  # Validate before any temp-file write.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            temporary = _write_temp(self.path.parent, self.path.name, raw.encode("utf-8"))
            os.replace(temporary, self.path)
            temporary = None
            _fsync_directory(self.path.parent)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def read(self) -> RunTransitionPlan:
        return self._validate_raw(self.path.read_text(encoding="utf-8"))

    def recover_after_interruption(self) -> RunTransitionPlan:
        """Return an explicit interrupted plan only when restart finds it running."""
        plan = self.read()
        return plan.recover_interrupted() if plan.lifecycle is RunLifecycle.RUNNING else plan

    def _validate_raw(self, raw: str) -> RunTransitionPlan:
        try:
            json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TransitionSnapshotCorruptError("Transition snapshot is malformed or truncated.") from exc
        try:
            plan = RunTransitionPlan.from_snapshot_json(raw, self.expected_identity)
        except TransitionPlanError as exc:
            raise TransitionSnapshotValidationError("Transition snapshot is unsupported, inconsistent, or identity-mismatched.") from exc
        if any(attempt.raw_response and _CREDENTIAL_MARKER.search(attempt.raw_response) for attempt in plan.attempts.values()):
            raise TransitionSnapshotValidationError("Transition snapshot contains credential-like response content.")
        return plan
