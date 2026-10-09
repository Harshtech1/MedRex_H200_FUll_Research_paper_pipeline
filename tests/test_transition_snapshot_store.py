"""Synthetic atomic persistence tests for versioned transition snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medrax.benchmark import transition_snapshot_store as store_module
from medrax.benchmark.chestagentbench_loader import PINNED_REVISION, ChestAgentBenchLoader
from medrax.benchmark.run_state import _write_temp
from medrax.benchmark.run_transition_plan import QuestionLifecycle, RunIdentity, RunTransitionPlan, TransitionPlanError
from medrax.benchmark.runner_contract import BenchmarkRequest, ControllerAttempt
from medrax.benchmark.transition_snapshot_store import AtomicTransitionSnapshotStore, TransitionSnapshotCorruptError, TransitionSnapshotValidationError
from medrax.controller_config import ControllerConfig


def _requests(root: Path) -> list[BenchmarkRequest]:
    (root / "figures").mkdir(parents=True); (root / "figures" / "a.jpg").write_bytes(b"a"); (root / "figures" / "b.jpg").write_bytes(b"b")
    rows = []
    for suffix in ("a", "b"):
        rows.append({"answer": "synthetic", "case_id": f"case-{suffix}", "categories": "synthetic", "explanation": "synthetic", "full_question_id": f"full-{suffix}", "image_source_urls": ["https://example.invalid"], "images": [f"figures/{suffix}.jpg"], "question": "synthetic", "question_id": f"q-{suffix}", "sections": "synthetic", "type": "synthetic"})
    (root / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = ChestAgentBenchLoader(root).load(); config = ControllerConfig.from_values(provider="gemini", model="synthetic", temperature=0.2, top_p=0.95)
    return [BenchmarkRequest.from_loader(dataset=dataset, question=q, run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=config) for q in dataset.questions]


@pytest.fixture
def requests(tmp_path: Path) -> list[BenchmarkRequest]: return _requests(tmp_path / "dataset")


def _running(requests: list[BenchmarkRequest]) -> RunTransitionPlan:
    return RunTransitionPlan.new(RunIdentity.from_request(requests[0])).initialize(requests).start()


def _complete_failed(requests: list[BenchmarkRequest]) -> RunTransitionPlan:
    plan = _running(requests).start_attempt(requests[0], "attempt_a")
    plan = plan.record_attempt(ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_a", status="completed", raw_response="synthetic"))
    plan = plan.start_attempt(requests[1], "attempt_b")
    return plan.record_attempt(ControllerAttempt.from_request(request=requests[1], attempt_id="attempt_b", status="execution_error", error_code="synthetic_failure"))


def test_roundtrip_preserves_identity_question_and_attempt_state(tmp_path: Path, requests) -> None:
    plan = _complete_failed(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity)
    store.write(plan); restored = store.read()
    assert restored.snapshot_json() == plan.snapshot_json()
    assert restored.questions["q-a"].lifecycle is QuestionLifecycle.COMPLETED
    assert restored.questions["q-b"].lifecycle is QuestionLifecycle.FAILED
    assert restored.attempts["attempt_b"].error_code == "synthetic_failure"


def test_equivalent_writes_are_deterministic(tmp_path: Path, requests) -> None:
    plan = _complete_failed(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity)
    store.write(plan); first = store.path.read_text(encoding="utf-8"); store.write(plan)
    assert store.path.read_text(encoding="utf-8") == first == plan.snapshot_json()


def test_recovery_marks_active_attempt_interrupted_not_completed(tmp_path: Path, requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_a")
    store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity); store.write(plan)
    recovered = store.recover_after_interruption()
    assert recovered.questions["q-a"].lifecycle is QuestionLifecycle.INTERRUPTED
    assert recovered.attempts["attempt_a"].outcome.value == "interrupted"
    assert recovered.questions["q-b"].lifecycle is QuestionLifecycle.PENDING


def _previous(store: AtomicTransitionSnapshotStore, plan: RunTransitionPlan) -> str:
    store.write(plan); return store.path.read_text(encoding="utf-8")


def test_serialization_failure_preserves_previous_snapshot(tmp_path: Path, requests, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _complete_failed(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity); previous = _previous(store, plan)
    monkeypatch.setattr(RunTransitionPlan, "snapshot_json", lambda self: (_ for _ in ()).throw(TypeError("synthetic serialization failure")))
    with pytest.raises(TypeError): store.write(plan)
    assert store.path.read_text(encoding="utf-8") == previous


def test_temp_write_and_replace_failure_preserve_previous_snapshot(tmp_path: Path, requests, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _complete_failed(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity); previous = _previous(store, plan)
    monkeypatch.setattr(store_module, "_write_temp", lambda *args: (_ for _ in ()).throw(OSError("synthetic write failure")))
    with pytest.raises(OSError): store.write(plan)
    assert store.path.read_text(encoding="utf-8") == previous
    monkeypatch.setattr(store_module, "_write_temp", _write_temp)
    monkeypatch.setattr(store_module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("synthetic replace failure")))
    with pytest.raises(OSError): store.write(plan)
    assert store.path.read_text(encoding="utf-8") == previous


@pytest.mark.parametrize("raw,error", [("{", TransitionSnapshotCorruptError), ("not-json", TransitionSnapshotCorruptError), (json.dumps({"schema_version": 999}), TransitionSnapshotValidationError)])
def test_corrupt_truncated_and_unsupported_snapshots_fail_closed(tmp_path: Path, requests, raw: str, error: type[Exception]) -> None:
    plan = _running(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity)
    store.path.write_text(raw, encoding="utf-8")
    with pytest.raises(error): store.read()


def test_run_revision_config_and_inconsistent_identity_mismatches_rejected(tmp_path: Path, requests) -> None:
    plan = _running(requests); store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity); store.write(plan)
    wrong = RunIdentity("other", plan.identity.dataset_revision, plan.identity.controller_config_fingerprint, plan.identity.request_schema_version)
    with pytest.raises(TransitionSnapshotValidationError): AtomicTransitionSnapshotStore(store.path, wrong).read()
    wrong_config = RunIdentity(plan.identity.run_id, plan.identity.dataset_revision, "0" * 64, plan.identity.request_schema_version)
    with pytest.raises(TransitionSnapshotValidationError): AtomicTransitionSnapshotStore(store.path, wrong_config).read()
    changed = json.loads(store.path.read_text(encoding="utf-8")); changed["identity"]["dataset_revision"] = "other"
    store.path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(TransitionSnapshotValidationError): store.read()


def test_inconsistent_persisted_attempt_identity_is_rejected(tmp_path: Path, requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_a")
    store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity); store.write(plan)
    changed = json.loads(store.path.read_text(encoding="utf-8")); changed["attempts"][0]["question_id"] = "unknown"
    store.path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(TransitionSnapshotValidationError): store.read()


def test_credential_like_raw_response_is_rejected_before_persistence(tmp_path: Path, requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_a")
    plan = plan.record_attempt(ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_a", status="completed", raw_response="Bearer synthetic-secret"))
    store = AtomicTransitionSnapshotStore(tmp_path / "transition.json", plan.identity)
    with pytest.raises(TransitionSnapshotValidationError): store.write(plan)
    assert not store.path.exists()


def test_legacy_projection_is_explicit_and_rejects_interrupted_state(requests) -> None:
    projection = _complete_failed(requests).checkpoint_projection()
    assert projection.completed_case_ids == frozenset({"q-a"}) and projection.failed_case_ids == frozenset({"q-b"})
    interrupted = _running(requests).start_attempt(requests[0], "attempt_a").recover_interrupted()
    with pytest.raises(TransitionPlanError): interrupted.checkpoint_projection()
