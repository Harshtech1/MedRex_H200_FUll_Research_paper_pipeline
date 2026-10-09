"""Synthetic integration tests; never imports or invokes concrete medical tools."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from medrax.benchmark.chestagentbench_loader import PINNED_REVISION, ChestAgentBenchLoader
from medrax.benchmark.locked_orchestrator import LockedRunOrchestrator
from medrax.benchmark.mock_controller_tool_integration import MockControllerToolIntegration, MockIntegrationError, MockToolDecision, MockToolOutput
from medrax.benchmark.run_state import RunNamespaceLock
from medrax.benchmark.run_transition_plan import RunIdentity, RunTransitionPlan
from medrax.benchmark.runner_contract import AttemptStatus, BenchmarkRequest
from medrax.benchmark.transition_snapshot_store import AtomicTransitionSnapshotStore
from medrax.benchmark.trusted_tool_dispatch import TrustedToolDispatcher
from medrax.controller_config import ControllerConfig
from medrax.tool_registry import PAPER_TOOL_NAMES


@pytest.fixture
def prepared(tmp_path: Path):
    root = tmp_path / "dataset"; (root / "figures").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"): (root / "figures" / name).write_bytes(b"synthetic")
    rows = []
    for case, images in (("case-a", ["a.jpg", "b.jpg"]), ("case-b", ["c.jpg"])):
        rows.append({"answer":"ground-truth", "case_id":case, "categories":"synthetic", "explanation":"ground-truth-explanation", "full_question_id":"full-"+case, "image_source_urls":["https://example.invalid"], "images":["figures/"+image for image in images], "question":"synthetic-question", "question_id":"q-"+case, "sections":"synthetic", "type":"synthetic"})
    (root / "metadata.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    dataset = ChestAgentBenchLoader(root).load(); config = ControllerConfig.from_values(provider="gemini", model="synthetic", temperature=0.2, top_p=0.95)
    request = BenchmarkRequest.from_loader(dataset=dataset, question=dataset.questions[0], run_id="run_17", dataset_revision=PINNED_REVISION, controller_config=config)
    plan = RunTransitionPlan.new(RunIdentity.from_request(request)).initialize([request])
    store = AtomicTransitionSnapshotStore(tmp_path / "state" / "transition.json", plan.identity)
    orchestrator = LockedRunOrchestrator(lock=RunNamespaceLock(tmp_path / "locks", "run_17"), plan=plan, snapshot_store=store)
    return dataset, orchestrator, request, TrustedToolDispatcher(dataset.adapter)


@pytest.mark.parametrize("name,arguments,keys", [
    ("chest_xray_report_generator", {}, {"image_path"}), ("chest_xray_classifier", {}, {"image_path"}),
    ("chest_xray_segmentation", {"organs":["Heart"]}, {"image_path", "organs"}), ("xray_phrase_grounding", {"phrase":"x"}, {"image_path", "phrase", "max_new_tokens"}),
    ("chest_xray_expert", {"prompt":"x"}, {"image_paths", "prompt", "max_new_tokens"}), ("llava_med_qa", {"question":"x"}, {"question", "image_path"}),
])
def test_each_six_tool_has_an_independent_mock_route(prepared, name, arguments, keys) -> None:
    _, orchestrator, request, dispatcher = prepared; captured = []
    tools = {tool: (lambda values, tool=tool: (captured.append((tool, dict(values))), MockToolOutput("mocked"))[1]) for tool in PAPER_TOOL_NAMES}
    ids = request.question.image_ids if name == "chest_xray_expert" else (request.question.image_ids[0],)
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: MockToolDecision(name, ids, arguments), dispatcher=dispatcher, mock_tools=tools)
    result = integration.invoke_once()
    assert set(PAPER_TOOL_NAMES) == {"chest_xray_report_generator", "chest_xray_classifier", "chest_xray_segmentation", "xray_phrase_grounding", "chest_xray_expert", "llava_med_qa"}
    assert result.status is AttemptStatus.COMPLETED and captured[0][0] == name and set(captured[0][1]) == keys
    assert "ground-truth" not in repr(integration.controller_input) and "figures" not in repr(integration.controller_input) and not hasattr(integration.controller_input, "api_key")
    assert not any(module.startswith("medrax.tools.") for module in sys.modules)


@pytest.mark.parametrize("kind", ["unknown_tool", "unknown_id", "cross_case", "raw_path", "ground_truth"])
def test_invalid_decisions_fail_closed_before_any_fake_tool(prepared, kind) -> None:
    dataset, orchestrator, request, dispatcher = prepared; calls = []; image = request.question.image_ids[0]
    decisions = {
        "unknown_tool": MockToolDecision("unknown", (image,), {}),
        "unknown_id": MockToolDecision("chest_xray_classifier", ("unknown",), {}),
        "cross_case": MockToolDecision("chest_xray_classifier", (dataset.questions[1].image_ids[0],), {}),
        "raw_path": MockToolDecision("chest_xray_classifier", (image,), {"image_path":"/tmp/x"}),
        "ground_truth": MockToolDecision("llava_med_qa", (), {"question":"x", "answer":"leak"}),
    }
    decision = decisions[kind]
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={name: lambda values: calls.append(values) for name in PAPER_TOOL_NAMES})
    assert integration.invoke_once().status is AttemptStatus.INVALID_RESPONSE and calls == []


def test_integration_returns_attempt_for_existing_orchestrator_to_persist(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared
    decision = MockToolDecision("chest_xray_classifier", (request.question.image_ids[0],), {})
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={"chest_xray_classifier": lambda values: MockToolOutput("mocked")})
    result = integration.invoke_once()
    assert orchestrator.plan.questions[request.question.question_id].lifecycle.value == "attempt_started"
    orchestrator.record_outcome(result)
    assert orchestrator.snapshot_store.read().questions[request.question.question_id].lifecycle.value == "completed"


def test_malformed_output_exception_no_retry_and_duplicate_rejection(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared; calls = []
    decision = MockToolDecision("chest_xray_classifier", (request.question.image_ids[0],), {})
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={"chest_xray_classifier": lambda values: calls.append(values)})
    assert integration.invoke_once().status is AttemptStatus.INVALID_RESPONSE and len(calls) == 1
    with pytest.raises(MockIntegrationError): integration.invoke_once()
    assert len(calls) == 1



def test_fake_tool_exception_becomes_explicit_execution_failure(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared
    decision = MockToolDecision("chest_xray_classifier", (request.question.image_ids[0],), {})
    failing = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={"chest_xray_classifier": lambda values: (_ for _ in ()).throw(RuntimeError("synthetic"))})
    assert failing.invoke_once().status is AttemptStatus.EXECUTION_ERROR


def test_missing_tool_has_no_fallback(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared; calls = []
    decision = MockToolDecision("chest_xray_report_generator", (request.question.image_ids[0],), {})
    unavailable = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={"chest_xray_classifier": lambda values: calls.append(values)})
    assert unavailable.invoke_once().status is AttemptStatus.EXECUTION_ERROR and calls == []
    with pytest.raises(MockIntegrationError): unavailable.invoke_once()


def test_fake_controller_exception_is_terminal_and_not_retried(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared; calls = []
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: (calls.append(input), (_ for _ in ()).throw(RuntimeError("synthetic")))[1], dispatcher=dispatcher, mock_tools={})
    assert integration.invoke_once().status is AttemptStatus.EXECUTION_ERROR and len(calls) == 1
    with pytest.raises(MockIntegrationError): integration.invoke_once()
    assert len(calls) == 1


def test_fake_output_cannot_echo_trusted_path(prepared) -> None:
    _, orchestrator, request, dispatcher = prepared
    decision = MockToolDecision("chest_xray_classifier", (request.question.image_ids[0],), {})
    integration = MockControllerToolIntegration(preparation=orchestrator.prepare_attempt(request, "attempt_17"), controller=lambda input: decision, dispatcher=dispatcher, mock_tools={"chest_xray_classifier": lambda values: MockToolOutput(str(values["image_path"]))})
    assert integration.invoke_once().status is AttemptStatus.INVALID_RESPONSE
