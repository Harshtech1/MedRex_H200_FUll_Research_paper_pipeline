"""Synthetic CPU-only tests for B2 lock and checkpoint primitives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from medrax.benchmark import run_state
from medrax.benchmark.run_state import (
    AtomicCheckpointStore,
    CheckpointCorruptError,
    CheckpointValidationError,
    LockOwnershipError,
    RunCheckpoint,
    RunLockHeldError,
    RunNamespaceLock,
    StaleLockRecoveryError,
)


def test_namespace_lock_is_exclusive_and_namespaces_are_independent(tmp_path: Path) -> None:
    first = RunNamespaceLock(tmp_path, "run-a").acquire()
    with pytest.raises(RunLockHeldError):
        RunNamespaceLock(tmp_path, "run-a").acquire()
    other = RunNamespaceLock(tmp_path, "run-b").acquire()
    other.release()
    first.release()


def test_lock_only_owner_can_release(tmp_path: Path) -> None:
    owner = RunNamespaceLock(tmp_path, "run-a").acquire()
    impostor = RunNamespaceLock(tmp_path, "run-a")
    with pytest.raises(LockOwnershipError):
        impostor.release()
    owner.release()


def test_lock_releases_when_context_exits_with_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with RunNamespaceLock(tmp_path, "run-a"):
            raise RuntimeError("synthetic worker failure")
    RunNamespaceLock(tmp_path, "run-a").acquire().release()


def _worker_code() -> str:
    return """
import sys
from medrax.benchmark.run_state import RunNamespaceLock
with RunNamespaceLock(sys.argv[1], sys.argv[2]):
    print('LOCKED', flush=True)
    sys.stdin.readline()
"""


def _start_worker(directory: Path, namespace: str) -> subprocess.Popen[str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.Popen(
        [sys.executable, "-c", _worker_code(), str(directory), namespace],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "LOCKED"
    return process


def test_subprocess_exclusion_and_normal_release(tmp_path: Path) -> None:
    worker = _start_worker(tmp_path, "run-a")
    try:
        with pytest.raises(RunLockHeldError):
            RunNamespaceLock(tmp_path, "run-a").acquire()
        assert worker.stdin is not None
        worker.stdin.write("release\n")
        worker.stdin.flush()
        assert worker.wait(timeout=5) == 0
        RunNamespaceLock(tmp_path, "run-a").acquire().release()
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)


def test_terminated_worker_requires_explicit_conservative_recovery(tmp_path: Path) -> None:
    worker = _start_worker(tmp_path, "run-a")
    worker.terminate()
    assert worker.wait(timeout=5) != 0
    with pytest.raises(RunLockHeldError):
        RunNamespaceLock(tmp_path, "run-a").acquire()
    RunNamespaceLock.recover_stale(tmp_path, "run-a")
    RunNamespaceLock(tmp_path, "run-a").acquire().release()


def test_stale_recovery_refuses_living_owner(tmp_path: Path) -> None:
    lock = RunNamespaceLock(tmp_path, "run-a").acquire()
    with pytest.raises(StaleLockRecoveryError):
        RunNamespaceLock.recover_stale(tmp_path, "run-a")
    lock.release()


def test_fresh_checkpoint_and_deterministic_atomic_round_trip(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run-a")
    assert store.load_or_fresh() == RunCheckpoint.fresh("run-a")
    checkpoint = RunCheckpoint.fresh("run-a").mark_completed("case-b").mark_completed("case-a")
    store.write(checkpoint)
    assert store.read() == checkpoint
    assert store.path.read_text(encoding="utf-8") == '{"completed_case_ids":["case-a","case-b"],"failed_case_ids":[],"namespace":"run-a","schema_version":1}'


def test_resume_preserves_completion_deduplicates_and_distinguishes_failure(tmp_path: Path) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run-a")
    saved = RunCheckpoint.fresh("run-a").mark_completed("case-a").mark_failed("case-b")
    store.write(saved)
    resumed = store.read().mark_completed("case-a").mark_completed("case-b")
    assert resumed.completed_case_ids == frozenset({"case-a", "case-b"})
    assert resumed.failed_case_ids == frozenset()
    store.write(resumed)
    assert store.read() == resumed


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "namespace": "run-a", "completed_case_ids": [], "failed_case_ids": []},
        {"schema_version": 1, "namespace": "wrong", "completed_case_ids": [], "failed_case_ids": []},
        {"schema_version": 1, "namespace": "run-a", "completed_case_ids": ["case-a"], "failed_case_ids": ["case-a"]},
        {"schema_version": 1, "namespace": "run-a", "completed_case_ids": [], "failed_case_ids": [], "extra": True},
    ],
)
def test_checkpoint_validation_rejects_wrong_schema_namespace_and_state(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointValidationError):
        AtomicCheckpointStore(path, "run-a").read()


@pytest.mark.parametrize("content", ["{", "not json", "[1, 2]"])
def test_corrupt_or_truncated_checkpoint_never_becomes_fresh_success(tmp_path: Path, content: str) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(content, encoding="utf-8")
    store = AtomicCheckpointStore(path, "run-a")
    with pytest.raises((CheckpointCorruptError, CheckpointValidationError)):
        store.read()
    with pytest.raises((CheckpointCorruptError, CheckpointValidationError)):
        store.load_or_fresh()


def _old_checkpoint(store: AtomicCheckpointStore) -> str:
    store.write(RunCheckpoint.fresh("run-a").mark_completed("previous"))
    return store.path.read_text(encoding="utf-8")


def test_serialization_failure_preserves_previous_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run-a")
    previous = _old_checkpoint(store)
    monkeypatch.setattr(run_state.json, "dumps", lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("synthetic serialization failure")))
    with pytest.raises(TypeError):
        store.write(RunCheckpoint.fresh("run-a").mark_completed("new"))
    assert store.path.read_text(encoding="utf-8") == previous


def test_temp_write_failure_preserves_previous_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run-a")
    previous = _old_checkpoint(store)
    monkeypatch.setattr(run_state, "_write_temp", lambda *args: (_ for _ in ()).throw(OSError("synthetic temp failure")))
    with pytest.raises(OSError):
        store.write(RunCheckpoint.fresh("run-a").mark_completed("new"))
    assert store.path.read_text(encoding="utf-8") == previous


def test_replace_failure_preserves_previous_checkpoint_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run-a")
    previous = _old_checkpoint(store)
    monkeypatch.setattr(run_state.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("synthetic replace failure")))
    with pytest.raises(OSError):
        store.write(RunCheckpoint.fresh("run-a").mark_completed("new"))
    assert store.path.read_text(encoding="utf-8") == previous
    assert not list(tmp_path.glob(".checkpoint.json.*.tmp"))
