"""Local-filesystem safety primitives for resumable benchmark runs.

This module is deliberately independent of model and dataset code.  Locks are
for one local filesystem only; they do not claim distributed-filesystem safety.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any
from uuid import uuid4


CHECKPOINT_SCHEMA_VERSION = 1


class RunStateError(RuntimeError):
    """Base class for local benchmark state failures."""


class RunLockHeldError(RunStateError):
    """Raised when a namespace already has an active or unrecovered owner."""


class LockOwnershipError(RunStateError):
    """Raised when a non-owner attempts to release a lock."""


class StaleLockRecoveryError(RunStateError):
    """Raised when explicit stale-lock recovery cannot prove the owner is dead."""


class CheckpointValidationError(RunStateError):
    """Raised when a checkpoint is well-formed JSON but invalid for this run."""


class CheckpointCorruptError(RunStateError):
    """Raised when a checkpoint cannot be parsed as a complete JSON document."""


def _namespace(value: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("Namespace must be a non-empty path-safe identifier.")
    return value


def _fsync_directory(directory: Path) -> None:
    """Persist a prior rename on local POSIX filesystems where supported."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RunNamespaceLock:
    """Exclusive lock file with explicit, fail-closed stale recovery.

    A dead process leaves its lock in place.  A caller must explicitly invoke
    ``recover_stale``; this method only removes a same-host lock after checking
    that its PID is no longer alive.  It intentionally makes no distributed FS
    correctness claim.
    """

    def __init__(self, directory: str | Path, namespace: str) -> None:
        self.directory = Path(directory)
        self.namespace = _namespace(namespace)
        self.path = self.directory / f".{self.namespace}.lock"
        self._owner_id: str | None = None

    def acquire(self, *, wait: bool = False, timeout: float | None = None, poll_interval: float = 0.05) -> "RunNamespaceLock":
        if self._owner_id is not None:
            raise LockOwnershipError("This lock object already owns the namespace.")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative.")
        self.directory.mkdir(parents=True, exist_ok=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            owner_id = uuid4().hex
            payload = {
                "namespace": self.namespace,
                "owner_id": owner_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_unix": time.time(),
            }
            try:
                fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if not wait or (deadline is not None and time.monotonic() >= deadline):
                    raise RunLockHeldError("Benchmark namespace is already locked; use explicit stale recovery if appropriate.")
                time.sleep(poll_interval)
                continue
            try:
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                os.write(fd, raw)
                os.fsync(fd)
            except BaseException:
                os.close(fd)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                raise
            else:
                os.close(fd)
                _fsync_directory(self.directory)
                self._owner_id = owner_id
                return self

    def release(self) -> None:
        if self._owner_id is None:
            raise LockOwnershipError("This lock object does not own the namespace.")
        try:
            data = _read_lock_payload(self.path)
            if data.get("owner_id") != self._owner_id:
                raise LockOwnershipError("Lock ownership does not match; refusing to remove it.")
            self.path.unlink()
            _fsync_directory(self.directory)
        finally:
            # Retain ownership on mismatch so the caller cannot silently reuse
            # the object; clear it only after its own file has been removed.
            if not self.path.exists():
                self._owner_id = None

    @classmethod
    def recover_stale(cls, directory: str | Path, namespace: str) -> None:
        """Explicitly remove a local lock only after proving its PID is dead."""
        lock = cls(directory, namespace)
        try:
            data = _read_lock_payload(lock.path)
        except FileNotFoundError as exc:
            raise StaleLockRecoveryError("No lock exists to recover.") from exc
        if data.get("namespace") != lock.namespace:
            raise StaleLockRecoveryError("Lock namespace metadata is invalid; refusing recovery.")
        if data.get("hostname") != socket.gethostname() or not isinstance(data.get("pid"), int) or data["pid"] <= 0:
            raise StaleLockRecoveryError("Cannot prove a non-local lock owner is dead; refusing recovery.")
        if _pid_alive(data["pid"]):
            raise StaleLockRecoveryError("Lock owner is still alive; refusing recovery.")
        # A lock cannot be atomically compared-and-deleted portably with O_EXCL.
        # This is only for a proven-dead local owner and must be coordinated by
        # the invoking operator; a new acquirer can proceed after unlink.
        lock.path.unlink()
        _fsync_directory(lock.directory)

    def __enter__(self) -> "RunNamespaceLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.release()
        return False


def _read_lock_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StaleLockRecoveryError("Lock metadata is malformed; refusing recovery.") from exc
    if not isinstance(data, dict):
        raise StaleLockRecoveryError("Lock metadata is invalid; refusing recovery.")
    return data


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


@dataclass(frozen=True)
class RunCheckpoint:
    """Immutable completed/failed case state for one benchmark namespace."""

    namespace: str
    completed_case_ids: frozenset[str] = frozenset()
    failed_case_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _namespace(self.namespace)
        for values, label in ((self.completed_case_ids, "completed"), (self.failed_case_ids, "failed")):
            if not isinstance(values, frozenset) or any(not isinstance(value, str) or not value for value in values):
                raise CheckpointValidationError(f"{label} case IDs must be non-empty strings.")
        if self.completed_case_ids & self.failed_case_ids:
            raise CheckpointValidationError("A case cannot be both completed and failed.")

    @classmethod
    def fresh(cls, namespace: str) -> "RunCheckpoint":
        return cls(namespace)

    def mark_completed(self, case_id: str) -> "RunCheckpoint":
        if not isinstance(case_id, str) or not case_id:
            raise CheckpointValidationError("Case ID must be a non-empty string.")
        return RunCheckpoint(self.namespace, self.completed_case_ids | {case_id}, self.failed_case_ids - {case_id})

    def mark_failed(self, case_id: str) -> "RunCheckpoint":
        if not isinstance(case_id, str) or not case_id:
            raise CheckpointValidationError("Case ID must be a non-empty string.")
        if case_id in self.completed_case_ids:
            return self
        return RunCheckpoint(self.namespace, self.completed_case_ids, self.failed_case_ids | {case_id})

    def payload(self) -> dict[str, Any]:
        return {
            "completed_case_ids": sorted(self.completed_case_ids),
            "failed_case_ids": sorted(self.failed_case_ids),
            "namespace": self.namespace,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
        }


class AtomicCheckpointStore:
    """Deterministic JSON checkpoint persistence using same-directory rename."""

    def __init__(self, path: str | Path, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = _namespace(namespace)

    def write(self, checkpoint: RunCheckpoint) -> None:
        if not isinstance(checkpoint, RunCheckpoint) or checkpoint.namespace != self.namespace:
            raise CheckpointValidationError("Checkpoint namespace does not match this store.")
        # Serialize before creating a temp file, preserving a previous final
        # checkpoint when serialization fails.
        encoded = json.dumps(checkpoint.payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            temp_path = _write_temp(self.path.parent, self.path.name, encoded)
            os.replace(temp_path, self.path)
            temp_path = None
            _fsync_directory(self.path.parent)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def read(self) -> RunCheckpoint:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptError("Checkpoint is malformed or truncated.") from exc
        return _checkpoint_from_payload(payload, self.namespace)

    def load_or_fresh(self) -> RunCheckpoint:
        try:
            return self.read()
        except FileNotFoundError:
            return RunCheckpoint.fresh(self.namespace)


def _write_temp(directory: Path, basename: str, encoded: bytes) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix=f".{basename}.", suffix=".tmp", dir=directory)
    temp_path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return temp_path


def _checkpoint_from_payload(payload: object, namespace: str) -> RunCheckpoint:
    if not isinstance(payload, dict):
        raise CheckpointValidationError("Checkpoint root must be an object.")
    required = {"schema_version", "namespace", "completed_case_ids", "failed_case_ids"}
    if set(payload) != required:
        raise CheckpointValidationError("Checkpoint fields do not match the required schema.")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointValidationError("Checkpoint schema version is unsupported.")
    if payload["namespace"] != namespace:
        raise CheckpointValidationError("Checkpoint namespace does not match this run.")
    completed, failed = payload["completed_case_ids"], payload["failed_case_ids"]
    if not isinstance(completed, list) or not isinstance(failed, list):
        raise CheckpointValidationError("Checkpoint case lists are invalid.")
    if len(completed) != len(set(completed)) or len(failed) != len(set(failed)):
        raise CheckpointValidationError("Checkpoint contains duplicate case IDs.")
    try:
        return RunCheckpoint(namespace, frozenset(completed), frozenset(failed))
    except (TypeError, CheckpointValidationError) as exc:
        raise CheckpointValidationError("Checkpoint case IDs are invalid.") from exc
