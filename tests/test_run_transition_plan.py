"""Synthetic, non-executing state-transition tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medrax.benchmark.chestagentbench_loader import PINNED_REVISION, ChestAgentBenchLoader
from medrax.benchmark.run_state import AtomicCheckpointStore, CheckpointValidationError, RunCheckpoint
from medrax.benchmark.run_transition_plan import (
    TRANSITION_SNAPSHOT_SCHEMA_VERSION, QuestionLifecycle, RunIdentity, RunLifecycle,
    RunTransitionPlan, TransitionPlanError,
)
from medrax.benchmark.runner_contract import BenchmarkRequest, ControllerAttempt
from medrax.controller_config import ControllerConfig


def _dataset(root: Path):
    (root / "figures").mkdir(); (root / "figures" / "a.jpg").write_bytes(b"synthetic")
    rows = []
    for number in ("1", "2"):
        rows.append({"answer": "synthetic", "case_id": f"case-{number}", "categories": "synthetic", "explanation": "synthetic", "full_question_id": f"full-{number}", "image_source_urls": ["https://example.invalid"], "images": ["figures/a.jpg" if number == "1" else "figures/b.jpg"], "question": "synthetic", "question_id": f"q-{number}", "sections": "synthetic", "type": "synthetic"})
    (root / "figures" / "b.jpg").write_bytes(b"synthetic-b")
    (root / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return ChestAgentBenchLoader(root).load()


@pytest.fixture
def requests(tmp_path: Path):
    dataset = _dataset(tmp_path); config = ControllerConfig.from_values(provider="gemini", model="synthetic", temperature=0.2, top_p=0.95)
    return [BenchmarkRequest.from_loader(dataset=dataset, question=q, run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=config) for q in dataset.questions]


def _running(requests):
    identity = RunIdentity.from_request(requests[0])
    return RunTransitionPlan.new(identity).initialize(requests).start()


def test_legal_run_question_and_attempt_transitions_are_immutable(requests) -> None:
    running = _running(requests); started = running.start_attempt(requests[0], "attempt_1")
    completed = ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_1", status="completed", raw_response="synthetic")
    finished = started.record_attempt(completed)
    assert running.questions["q-1"].lifecycle is QuestionLifecycle.PENDING
    assert started.questions["q-1"].lifecycle is QuestionLifecycle.ATTEMPT_STARTED
    assert finished.questions["q-1"].lifecycle is QuestionLifecycle.COMPLETED


def test_illegal_transitions_and_duplicate_completion_are_rejected(requests) -> None:
    plan = RunTransitionPlan.new(RunIdentity.from_request(requests[0]))
    with pytest.raises(TransitionPlanError): plan.start()
    running = _running(requests).start_attempt(requests[0], "attempt_1")
    completed = ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_1", status="completed", raw_response="synthetic")
    finished = running.record_attempt(completed)
    with pytest.raises(TransitionPlanError): finished.record_attempt(completed)
    with pytest.raises(TransitionPlanError): finished.complete()


def test_failed_attempt_needs_explicit_retry_authorization(requests) -> None:
    started = _running(requests).start_attempt(requests[0], "attempt_1")
    failed = ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_1", status="execution_error", error_code="synthetic_failure")
    failed_plan = started.record_attempt(failed)
    with pytest.raises(TransitionPlanError): failed_plan.start_attempt(requests[0], "attempt_2")
    retryable = failed_plan.authorize_retry("q-1")
    assert retryable.start_attempt(requests[0], "attempt_2").questions["q-1"].attempt_count == 2


def test_recovery_marks_only_active_attempts_interrupted(requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_1")
    recovered = plan.recover_interrupted()
    assert recovered.lifecycle is RunLifecycle.INTERRUPTED
    assert recovered.questions["q-1"].lifecycle is QuestionLifecycle.INTERRUPTED
    assert recovered.attempts["attempt_1"].outcome.value == "interrupted"
    with pytest.raises(TransitionPlanError): recovered.checkpoint_projection()


def test_checkpoint_projection_roundtrip_preserves_completed_and_failed(tmp_path: Path, requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_1")
    done = plan.record_attempt(ControllerAttempt.from_request(request=requests[0], attempt_id="attempt_1", status="completed", raw_response="synthetic"))
    failed = done.start_attempt(requests[1], "attempt_2").record_attempt(ControllerAttempt.from_request(request=requests[1], attempt_id="attempt_2", status="execution_error", error_code="synthetic_failure"))
    store = AtomicCheckpointStore(tmp_path / "checkpoint.json", "run_01"); store.write(failed.checkpoint_projection())
    restored = RunTransitionPlan.from_checkpoint(failed.identity, [(q.question_id, q.case_id) for q in failed.questions.values()], store.read())
    assert restored.questions["q-1"].lifecycle is QuestionLifecycle.COMPLETED
    assert restored.questions["q-2"].lifecycle is QuestionLifecycle.FAILED


def test_snapshot_roundtrip_identity_and_schema_fail_closed(requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_1")
    raw = plan.snapshot_json()
    assert RunTransitionPlan.from_snapshot_json(raw, plan.identity).snapshot_json() == raw
    bad_schema = json.loads(raw); bad_schema["schema_version"] = TRANSITION_SNAPSHOT_SCHEMA_VERSION + 1
    with pytest.raises(TransitionPlanError): RunTransitionPlan.from_snapshot_json(json.dumps(bad_schema), plan.identity)
    wrong_identity = RunIdentity("other_run", plan.identity.dataset_revision, plan.identity.controller_config_fingerprint, plan.identity.request_schema_version)
    with pytest.raises(TransitionPlanError): RunTransitionPlan.from_snapshot_json(raw, wrong_identity)
    duplicate = json.loads(raw); duplicate["questions"].append(duplicate["questions"][0])
    with pytest.raises(TransitionPlanError): RunTransitionPlan.from_snapshot_json(json.dumps(duplicate), plan.identity)


def test_checkpoint_identity_and_schema_fail_closed(requests) -> None:
    identity = RunIdentity.from_request(requests[0])
    with pytest.raises(TransitionPlanError): RunTransitionPlan.from_checkpoint(identity, [("q-1", "case-1")], RunCheckpoint.fresh("other"))
    with pytest.raises(TransitionPlanError): RunIdentity(identity.run_id, identity.dataset_revision, identity.controller_config_fingerprint, identity.request_schema_version, checkpoint_schema_version=99)
    with pytest.raises(CheckpointValidationError): RunCheckpoint("run_01", frozenset({"q-1"}), frozenset({"q-1"}))


def test_snapshot_and_attempts_have_no_ground_truth_or_asset_paths(requests) -> None:
    plan = _running(requests).start_attempt(requests[0], "attempt_1")
    payload = json.loads(plan.snapshot_json())
    serialized = json.dumps(payload, sort_keys=True)
    assert '"answer"' not in serialized and '"explanation"' not in serialized
    assert "figures" not in serialized and ".jpg" not in serialized
