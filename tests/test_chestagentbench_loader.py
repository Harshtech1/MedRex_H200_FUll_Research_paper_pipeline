"""CPU-only tests for read-only ChestAgentBench loading and adapter integration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from medrax.benchmark import chestagentbench_loader as loader_module
from medrax.benchmark.chestagentbench_loader import (
    ChestAgentBenchLoader,
    ChestAgentBenchLoaderError,
    ManifestIntegrityError,
    ManifestRowError,
    OpaqueImageIdCollisionError,
)
from medrax.benchmark.image_identifier_adapter import WrongCaseImageIdentifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = PROJECT_ROOT / "chestagentbench"
EXPECTED_METADATA_SHA256 = "45e429e3a3b8e4dbfa064cbdaa3f2960bee90cad9e0f25501e38da67597bc2a8"
EXPECTED_FIGURES_SHA256 = "8d5ac156c267046aadf54f20d961896c6e73d31ec59b8f635d1b208fa1444226"


def _row(*, question_id: str = "q-1", full_question_id: str = "fq-1", case_id: str = "case-a", images: object = None) -> dict[str, object]:
    return {
        "question_id": question_id, "full_question_id": full_question_id, "case_id": case_id,
        "question": "synthetic question", "categories": "synthetic-category", "sections": "synthetic-section",
        "type": "synthetic-type", "images": ["figures/a.jpg"] if images is None else images,
        "image_source_urls": ["https://example.invalid/synthetic"], "answer": "synthetic answer",
        "explanation": "synthetic explanation",
    }


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> None:
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "figures" / "a.jpg").write_bytes(b"synthetic-a")
    (root / "figures" / "b.jpg").write_bytes(b"synthetic-b")
    (root / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_synthetic_load_preserves_order_repeated_cases_and_ground_truth_separation(tmp_path: Path) -> None:
    rows = [_row(question_id="q-1", full_question_id="fq-1"), _row(question_id="q-2", full_question_id="fq-2")]
    _write_manifest(tmp_path, rows)
    dataset = ChestAgentBenchLoader(tmp_path).load()
    assert [question.question_id for question in dataset.questions] == ["q-1", "q-2"]
    assert [question.case_id for question in dataset.questions] == ["case-a", "case-a"]
    question = dataset.questions[0]
    assert not hasattr(question, "answer") and not hasattr(question, "explanation")
    assert "figures" not in repr(question) and "/" not in question.image_ids[0]
    assert dataset.evaluation_for("q-1").answer == "synthetic answer"


def test_deterministic_opaque_ids_and_same_case_resolution(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [_row()])
    first = ChestAgentBenchLoader(tmp_path).load(); second = ChestAgentBenchLoader(tmp_path).load()
    assert first.questions[0].image_ids == second.questions[0].image_ids
    context = first.context_for(first.questions[0])
    assert first.adapter.resolve(context, first.questions[0].image_ids[0]).is_file()


def test_cross_case_identifier_is_rejected(tmp_path: Path) -> None:
    rows = [_row(), _row(question_id="q-2", full_question_id="fq-2", case_id="case-b", images=["figures/b.jpg"])]
    _write_manifest(tmp_path, rows)
    dataset = ChestAgentBenchLoader(tmp_path).load()
    with pytest.raises(WrongCaseImageIdentifier):
        dataset.adapter.resolve(dataset.context_for(dataset.questions[1]), dataset.questions[0].image_ids[0])


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(question_id=""),
    lambda row: row.pop("question"),
    lambda row: row.update(images="figures/a.jpg"),
])
def test_missing_or_wrong_required_fields_are_row_indexed(tmp_path: Path, mutation) -> None:
    row = _row(); mutation(row); _write_manifest(tmp_path, [row])
    with pytest.raises(ManifestRowError, match="Row 1"):
        ChestAgentBenchLoader(tmp_path).load()


@pytest.mark.parametrize("rows", [
    [_row(), _row(full_question_id="fq-2")],
    [_row(), _row(question_id="q-2")],
])
def test_malformed_json_and_duplicate_question_identifiers_are_rejected(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    (tmp_path / "figures").mkdir(); (tmp_path / "figures" / "a.jpg").write_bytes(b"synthetic")
    (tmp_path / "metadata.jsonl").write_text("{\n", encoding="utf-8")
    with pytest.raises(ManifestRowError, match="Row 1"):
        ChestAgentBenchLoader(tmp_path).load()
    _write_manifest(tmp_path, rows)
    with pytest.raises(ManifestRowError, match="duplicate question"):
        ChestAgentBenchLoader(tmp_path).load()


@pytest.mark.parametrize("reference", ["../outside.jpg", "/tmp/outside.jpg", "figures/missing.jpg"])
def test_out_of_root_and_missing_images_are_rejected(tmp_path: Path, reference: str) -> None:
    _write_manifest(tmp_path, [_row(images=[reference])])
    with pytest.raises(ManifestRowError, match="Row 1"):
        ChestAgentBenchLoader(tmp_path).load()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [_row(images=["figures/escape.jpg"])])
    outside = tmp_path / "outside.jpg"; outside.write_bytes(b"outside")
    try:
        os.symlink(outside, tmp_path / "figures" / "escape.jpg")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ManifestRowError, match="Row 1"):
        ChestAgentBenchLoader(tmp_path).load()


def test_shared_asset_across_cases_is_rejected_without_broadening_authorization(tmp_path: Path) -> None:
    rows = [_row(), _row(question_id="q-2", full_question_id="fq-2", case_id="case-b")]
    _write_manifest(tmp_path, rows)
    with pytest.raises(ManifestRowError, match="multiple cases"):
        ChestAgentBenchLoader(tmp_path).load()


def test_opaque_identifier_collision_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(images=["figures/a.jpg", "figures/b.jpg"])]
    _write_manifest(tmp_path, rows)
    monkeypatch.setattr(loader_module, "_opaque_image_id", lambda reference: "cabimg_collision")
    with pytest.raises(OpaqueImageIdCollisionError):
        ChestAgentBenchLoader(tmp_path).load()


def test_context_requires_exact_loader_issued_question(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [_row()]); dataset = ChestAgentBenchLoader(tmp_path).load()
    forged = object()
    with pytest.raises(ChestAgentBenchLoaderError):
        dataset.context_for(forged)  # type: ignore[arg-type]


def test_real_pinned_manifest_loads_read_only_and_preserves_expected_contract() -> None:
    before_metadata = hashlib.sha256((REAL_ROOT / "metadata.jsonl").read_bytes()).hexdigest()
    before_archive = hashlib.sha256((REAL_ROOT / "figures.zip").read_bytes()).hexdigest()
    dataset = ChestAgentBenchLoader(REAL_ROOT).load()
    assert len(dataset.questions) == 2500 and len(dataset.evaluation_records) == 2500
    assert len({question.question_id for question in dataset.questions}) == 2500
    assert len({question.full_question_id for question in dataset.questions}) == 2500
    assert len({question.case_id for question in dataset.questions}) == 609
    assert sum(len(question.image_ids) for question in dataset.questions) == 4629
    assert len({image_id for question in dataset.questions for image_id in question.image_ids}) == 1346
    assert before_metadata == EXPECTED_METADATA_SHA256 == hashlib.sha256((REAL_ROOT / "metadata.jsonl").read_bytes()).hexdigest()
    assert before_archive == EXPECTED_FIGURES_SHA256 == hashlib.sha256((REAL_ROOT / "figures.zip").read_bytes()).hexdigest()


def test_real_provenance_mismatch_is_rejected_without_reading_images(tmp_path: Path) -> None:
    _write_manifest(tmp_path, [_row()])
    (tmp_path / "ACQUISITION_PROVENANCE.json").write_text(json.dumps({
        "dataset_repository": "wrong/repository", "revision": "wrong", "files": {"metadata.jsonl": {"sha256": "bad"}}
    }), encoding="utf-8")
    with pytest.raises(ManifestIntegrityError):
        ChestAgentBenchLoader(tmp_path).load()
