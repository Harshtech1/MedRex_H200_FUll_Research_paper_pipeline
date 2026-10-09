"""Synthetic non-executing lock/snapshot orchestration tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medrax.benchmark.chestagentbench_loader import PINNED_REVISION, ChestAgentBenchLoader
from medrax.benchmark.locked_orchestrator import LockedRunOrchestrator, OrchestrationError
from medrax.benchmark.run_state import RunLockHeldError, RunNamespaceLock
from medrax.benchmark.run_transition_plan import QuestionLifecycle, RunIdentity, RunLifecycle, RunTransitionPlan
from medrax.benchmark.runner_contract import BenchmarkRequest, ControllerAttempt
from medrax.benchmark.transition_snapshot_store import AtomicTransitionSnapshotStore
from medrax.controller_config import ControllerConfig


def _requests(root: Path) -> list[BenchmarkRequest]:
    (root / "figures").mkdir(parents=True); (root / "figures" / "a.jpg").write_bytes(b"a")
    row = {"answer": "synthetic", "case_id": "case-a", "categories": "synthetic", "explanation": "synthetic", "full_question_id": "full-a", "image_source_urls": ["https://example.invalid"], "images": ["figures/a.jpg"], "question": "synthetic", "question_id": "q-a", "sections": "synthetic", "type": "synthetic"}
    (root / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = ChestAgentBenchLoader(root).load(); config = ControllerConfig.from_values(provider="gemini", model="synthetic", temperature=0.2, top_p=0.95)
    return [BenchmarkRequest.from_loader(dataset=dataset, question=dataset.questions[0], run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=config)]


@pytest.fixture
def setup(tmp_path: Path):
    requests = _requests(tmp_path / "dataset"); plan = RunTransitionPlan.new(RunIdentity.from_request(requests[0])).initialize(requests)
    return requests, plan, AtomicTransitionSnapshotStore(tmp_path / "state" / "transition.json", plan.identity), tmp_path / "locks"


def test_preflight_and_attempt_start_are_persisted_in_order_without_invocation(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    requests, plan, store, lock_dir = setup; events = []; original = store.write
    monkeypatch.setattr(store, "write", lambda candidate: (events.append(candidate.lifecycle.value + ":" + next(iter(candidate.questions.values())).lifecycle.value), original(candidate))[1])
    calls = []
    orchestrator = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store, invocation_boundary=lambda request: calls.append(request))
    orchestrator.prepare_attempt(requests[0], "attempt_1")
    assert events == ["initialized:pending", "running:pending", "running:attempt_started"]
    assert calls == [] and orchestrator.plan.questions["q-a"].lifecycle is QuestionLifecycle.ATTEMPT_STARTED


def test_completion_and_execution_failure_are_distinct_and_durable(setup) -> None:
    requests, plan, store, lock_dir = setup; orchestrator = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store)
    orchestrator.prepare_attempt(requests[0], "attempt_1")
    completed = ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_1", status="completed", raw_response="synthetic")
    assert orchestrator.record_outcome(completed).questions["q-a"].lifecycle is QuestionLifecycle.COMPLETED
    assert store.read().questions["q-a"].lifecycle is QuestionLifecycle.COMPLETED
    with pytest.raises(OrchestrationError): orchestrator.record_outcome(completed)


def test_execution_failure_is_persisted_and_never_retried_automatically(setup) -> None:
    requests, plan, store, lock_dir = setup; orchestrator = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store)
    orchestrator.prepare_attempt(requests[0], "attempt_1")
    failed = orchestrator.record_execution_failure(requests[0], "attempt_1", "synthetic_failure")
    assert failed.questions["q-a"].lifecycle is QuestionLifecycle.FAILED and len(failed.attempts) == 1
    assert store.read().questions["q-a"].lifecycle is QuestionLifecycle.FAILED


def test_lock_contention_and_exception_release(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    requests, plan, store, lock_dir = setup; first = RunNamespaceLock(lock_dir, "run_01").acquire()
    blocked = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store)
    with pytest.raises(RunLockHeldError): blocked.preflight()
    first.release()
    orchestrator = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store)
    monkeypatch.setattr(store, "write", lambda plan: (_ for _ in ()).throw(OSError("synthetic write failure")))
    with pytest.raises(OSError): orchestrator.preflight()
    assert orchestrator.plan.lifecycle is RunLifecycle.INITIALIZED
    RunNamespaceLock(lock_dir, "run_01").acquire().release()


def test_identity_mismatch_fails_before_attempt_and_write(setup) -> None:
    requests, plan, store, lock_dir = setup; wrong = AtomicTransitionSnapshotStore(store.path, RunIdentity("wrong", plan.identity.dataset_revision, plan.identity.controller_config_fingerprint, plan.identity.request_schema_version))
    with pytest.raises(OrchestrationError): LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=wrong)
    assert not store.path.exists()


def test_interruption_recovery_is_not_durable_until_explicit_write(setup) -> None:
    requests, plan, store, lock_dir = setup; original = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=plan, snapshot_store=store)
    original.prepare_attempt(requests[0], "attempt_1"); before = store.read().snapshot_json()
    restarted = LockedRunOrchestrator(lock=RunNamespaceLock(lock_dir, "run_01"), plan=original.plan, snapshot_store=store)
    recovered = restarted.recover_interruption()
    assert recovered.lifecycle is RunLifecycle.INTERRUPTED and recovered.questions["q-a"].lifecycle is QuestionLifecycle.INTERRUPTED
    assert store.read().snapshot_json() == before
    restarted.persist_recovered(); assert store.read().lifecycle is RunLifecycle.INTERRUPTED
