"""Fake-only tests for the strictly opt-in controller invocation adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medrax.benchmark.chestagentbench_loader import PINNED_REVISION, ChestAgentBenchLoader
from medrax.benchmark.controller_invocation_adapter import ControllerCallableError, ControllerInvocationAdapter, ControllerInvocationError
from medrax.benchmark.locked_orchestrator import AttemptPreparation
from medrax.benchmark.runner_contract import AttemptStatus, BenchmarkRequest, ControllerAttempt
from medrax.controller_config import ControllerConfig


@pytest.fixture
def preparation(tmp_path: Path) -> AttemptPreparation:
    (tmp_path / "figures").mkdir(); (tmp_path / "figures" / "a.jpg").write_bytes(b"synthetic")
    row = {"answer": "ground-truth", "case_id": "case-a", "categories": "synthetic", "explanation": "ground-truth-explanation", "full_question_id": "full-a", "image_source_urls": ["https://example.invalid"], "images": ["figures/a.jpg"], "question": "synthetic-question", "question_id": "q-a", "sections": "synthetic", "type": "synthetic"}
    (tmp_path / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = ChestAgentBenchLoader(tmp_path).load(); config = ControllerConfig.from_values(provider="gemini", model="synthetic", temperature=0.2, top_p=0.95)
    request = BenchmarkRequest.from_loader(dataset=dataset, question=dataset.questions[0], run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=config)
    return AttemptPreparation(request, "attempt_1")


def test_construction_and_input_access_do_not_invoke(preparation) -> None:
    calls = []
    adapter = ControllerInvocationAdapter(preparation, lambda input: calls.append(input))
    assert adapter.input.question_id == "q-a" and calls == []


def test_explicit_valid_invocation_once_and_boundary_has_no_ground_truth_paths_or_credentials(preparation) -> None:
    received = []
    def fake(input):
        received.append(input)
        return ControllerAttempt.from_request(request=preparation.request, attempt_id="attempt_1", status="completed", raw_response="synthetic")
    adapter = ControllerInvocationAdapter(preparation, fake)
    result = adapter.invoke_once()
    assert result.status is AttemptStatus.COMPLETED and len(received) == 1
    boundary = received[0]
    assert not hasattr(boundary, "answer") and not hasattr(boundary, "explanation")
    assert "figures" not in repr(boundary) and ".jpg" not in repr(boundary) and not hasattr(boundary, "case_context") and not hasattr(boundary, "api_key")
    with pytest.raises(ControllerInvocationError): adapter.invoke_once()
    assert len(received) == 1


def test_invalid_preparation_rejected_before_callable_runs() -> None:
    calls = []
    with pytest.raises(ControllerInvocationError): ControllerInvocationAdapter(object(), lambda input: calls.append(input))
    assert calls == []


@pytest.mark.parametrize("factory", [
    lambda p: object(),
    lambda p: ControllerAttempt.from_request(request=p.request, attempt_id="other", status="completed", raw_response="synthetic"),
    lambda p: ControllerAttempt.from_request(request=p.request, attempt_id=p.attempt_id, status="not_executed"),
])
def test_invalid_output_identity_and_status_fail_closed(preparation, factory) -> None:
    adapter = ControllerInvocationAdapter(preparation, lambda input: factory(preparation))
    with pytest.raises(ControllerInvocationError): adapter.invoke_once()
    with pytest.raises(ControllerInvocationError): adapter.invoke_once()


def test_callable_exception_is_explicit_and_not_retried(preparation) -> None:
    calls = []
    def broken(input): calls.append(input); raise RuntimeError("synthetic")
    adapter = ControllerInvocationAdapter(preparation, broken)
    with pytest.raises(ControllerCallableError): adapter.invoke_once()
    with pytest.raises(ControllerInvocationError): adapter.invoke_once()
    assert len(calls) == 1


def test_forged_mismatched_question_identity_is_rejected(preparation) -> None:
    forged = object.__new__(ControllerAttempt)
    object.__setattr__(forged, "request_identity", ("run_01", PINNED_REVISION, "other-question", "case-a"))
    object.__setattr__(forged, "attempt_id", preparation.attempt_id)
    object.__setattr__(forged, "status", AttemptStatus.COMPLETED)
    object.__setattr__(forged, "raw_response", "synthetic")
    object.__setattr__(forged, "error_code", None)
    with pytest.raises(ControllerInvocationError):
        ControllerInvocationAdapter(preparation, lambda input: forged).invoke_once()


def test_missing_or_unsupported_result_contract_fails_closed(preparation) -> None:
    adapter = ControllerInvocationAdapter(preparation, lambda input: ControllerAttempt.from_request(request=preparation.request, attempt_id=preparation.attempt_id, status="completed", raw_response=None))
    with pytest.raises(ControllerCallableError): adapter.invoke_once()
