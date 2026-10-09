"""CPU-only contract tests; no controller, model, or benchmark execution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medrax.benchmark.chestagentbench_loader import (
    PINNED_REVISION,
    ChestAgentBenchLoader,
    EvaluationRecord,
)
from medrax.benchmark.runner_contract import (
    RUNNER_CONTRACT_SCHEMA_VERSION,
    AttemptStatus,
    BenchmarkContractError,
    BenchmarkRequest,
    ControllerAttempt,
)
from medrax.controller_config import ControllerConfig


def _write_synthetic_dataset(root: Path) -> None:
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "synthetic.jpg").write_bytes(b"synthetic")
    row = {
        "answer": "synthetic-answer", "case_id": "case-a", "categories": "synthetic-category",
        "explanation": "synthetic-explanation", "full_question_id": "full-q-1",
        "image_source_urls": ["https://example.invalid/synthetic"], "images": ["figures/synthetic.jpg"],
        "question": "synthetic-question", "question_id": "q-1", "sections": "synthetic-section", "type": "synthetic-type",
    }
    (root / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.fixture
def loaded_dataset(tmp_path: Path):
    _write_synthetic_dataset(tmp_path)
    return ChestAgentBenchLoader(tmp_path).load()


@pytest.fixture
def controller_config() -> ControllerConfig:
    return ControllerConfig.from_values(provider="gemini", model="synthetic-controller", temperature=0.2, top_p=0.95)


@pytest.fixture
def contract_request(loaded_dataset, controller_config) -> BenchmarkRequest:
    return BenchmarkRequest.from_loader(
        dataset=loaded_dataset, question=loaded_dataset.questions[0], run_id="run_01",
        dataset_revision=PINNED_REVISION, controller_config=controller_config,
    )


def test_valid_loader_issued_request_has_stable_identity_and_trusted_context(contract_request, loaded_dataset) -> None:
    assert contract_request.identity == ("run_01", PINNED_REVISION, "q-1", "case-a")
    assert contract_request.case_context == loaded_dataset.context_for(loaded_dataset.questions[0])


@pytest.mark.parametrize("run_id,revision,schema", [
    ("../run", PINNED_REVISION, RUNNER_CONTRACT_SCHEMA_VERSION),
    ("run_01", "un-pinned", RUNNER_CONTRACT_SCHEMA_VERSION),
    ("run_01", PINNED_REVISION, "wrong-schema"),
])
def test_invalid_run_identity_or_schema_is_rejected(loaded_dataset, controller_config, run_id, revision, schema) -> None:
    with pytest.raises(BenchmarkContractError):
        BenchmarkRequest.from_loader(dataset=loaded_dataset, question=loaded_dataset.questions[0], run_id=run_id, dataset_revision=revision, controller_config=controller_config, schema_version=schema)


def test_fabricated_question_and_evaluation_record_cannot_be_controller_requests(loaded_dataset, controller_config) -> None:
    evaluation = EvaluationRecord("q-1", "synthetic-answer", "synthetic-explanation")
    for invalid in (object(), evaluation):
        with pytest.raises(BenchmarkContractError):
            BenchmarkRequest.from_loader(dataset=loaded_dataset, question=invalid, run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=controller_config)  # type: ignore[arg-type]


def test_question_from_another_dataset_load_is_rejected(loaded_dataset, controller_config) -> None:
    other = ChestAgentBenchLoader(loaded_dataset.adapter._root).load()
    with pytest.raises(BenchmarkContractError):
        BenchmarkRequest.from_loader(dataset=loaded_dataset, question=other.questions[0], run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=controller_config)


@pytest.mark.parametrize("config", [
    object(),
    ControllerConfig(provider="gemini", model="synthetic", temperature=9.0, top_p=0.5),
])
def test_invalid_provider_configuration_is_rejected(loaded_dataset, config) -> None:
    with pytest.raises(BenchmarkContractError):
        BenchmarkRequest.from_loader(dataset=loaded_dataset, question=loaded_dataset.questions[0], run_id="run_01", dataset_revision=PINNED_REVISION, controller_config=config)  # type: ignore[arg-type]


def test_serialization_is_deterministic_and_excludes_ground_truth_paths_and_context(contract_request) -> None:
    first = contract_request.controller_json(); second = contract_request.controller_json()
    assert first == second
    payload = json.loads(first)
    assert "answer" not in first and "explanation" not in first and "figures" not in first
    assert "_case_context" not in first and "synthetic.jpg" not in first
    assert payload["question"]["image_ids"] and all("/" not in value for value in payload["question"]["image_ids"])


@pytest.mark.parametrize("status,raw,error", [
    (AttemptStatus.COMPLETED, "synthetic-response", None),
    (AttemptStatus.EXECUTION_ERROR, None, "tool_failure"),
    (AttemptStatus.INVALID_RESPONSE, "synthetic-malformed", "invalid_response"),
    (AttemptStatus.NOT_EXECUTED, None, None),
    (AttemptStatus.INTERRUPTED, None, None),
])
def test_all_output_statuses_have_valid_separate_payloads(contract_request, status, raw, error) -> None:
    attempt = ControllerAttempt.from_request(request=contract_request, attempt_id="attempt_01", status=status, raw_response=raw, error_code=error)
    persisted = json.loads(attempt.persisted_json())
    assert persisted["status"] == status.value and "answer" not in persisted and "explanation" not in persisted


@pytest.mark.parametrize("status,raw,error", [
    ("unknown", None, None), (AttemptStatus.COMPLETED, None, None),
    (AttemptStatus.EXECUTION_ERROR, "unexpected", "tool_failure"),
    (AttemptStatus.INVALID_RESPONSE, "bad", None), (AttemptStatus.INTERRUPTED, None, "why"),
])
def test_invalid_output_status_combinations_are_rejected(contract_request, status, raw, error) -> None:
    with pytest.raises(BenchmarkContractError):
        ControllerAttempt.from_request(request=contract_request, attempt_id="attempt_01", status=status, raw_response=raw, error_code=error)


def test_path_like_attempt_identifier_is_rejected(contract_request) -> None:
    with pytest.raises(BenchmarkContractError):
        ControllerAttempt.from_request(request=contract_request, attempt_id="../attempt", status="not_executed")


def test_persisted_attempt_is_deterministic_and_references_only_identity(contract_request) -> None:
    first = ControllerAttempt.from_request(request=contract_request, attempt_id="attempt_01", status="completed", raw_response="synthetic-response")
    second = ControllerAttempt.from_request(request=contract_request, attempt_id="attempt_01", status="completed", raw_response="synthetic-response")
    assert first.persisted_json() == second.persisted_json()
    assert "synthetic-question" not in first.persisted_json()
