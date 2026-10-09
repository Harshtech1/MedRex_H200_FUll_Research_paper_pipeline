"""Read-only, path-safe loading for a local pinned ChestAgentBench manifest.

The loader deliberately has no model, network, image-decoding, or controller
dependencies.  It exposes opaque image IDs to controller-facing records while
retaining canonical paths only inside ``CaseImageIdentifierAdapter``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .image_identifier_adapter import (
    CaseContext,
    CaseImageIdentifierAdapter,
    ImageAssetEntry,
    ManifestValidationError,
)


CANONICAL_REPOSITORY = "wanglab/chest-agent-bench"
PINNED_REVISION = "921e60440927f9893228d843c0206b755744252a"
_REQUIRED_FIELDS = {
    "answer": str,
    "case_id": str,
    "categories": str,
    "explanation": str,
    "full_question_id": str,
    "image_source_urls": list,
    "images": list,
    "question": str,
    "question_id": str,
    "sections": str,
    "type": str,
}


class ChestAgentBenchLoaderError(ValueError):
    """Base error for manifest validation failures without record disclosure."""


class ManifestRowError(ChestAgentBenchLoaderError):
    """A malformed row, reported by one-based row number only."""


class ManifestIntegrityError(ChestAgentBenchLoaderError):
    """An invalid pinned provenance record or unsafe dataset asset."""


class OpaqueImageIdCollisionError(ChestAgentBenchLoaderError):
    """A deterministic opaque-ID collision across distinct image assets."""


@dataclass(frozen=True)
class ControllerQuestion:
    """Inference input with no raw paths, answer, or explanation fields."""

    question_id: str
    full_question_id: str
    case_id: str
    question: str
    categories: str
    sections: str
    question_type: str
    image_ids: tuple[str, ...]
    _capability: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class EvaluationRecord:
    """Evaluation-only ground truth, deliberately separate from inference input."""

    question_id: str
    answer: str
    explanation: str


@dataclass(frozen=True)
class ChestAgentBenchDataset:
    """Loaded questions and evaluation records with a trusted adapter seam."""

    questions: tuple[ControllerQuestion, ...]
    evaluation_records: tuple[EvaluationRecord, ...]
    adapter: CaseImageIdentifierAdapter = field(repr=False)
    metadata_sha256: str
    _capability: object = field(repr=False, compare=False)
    _questions_by_id: Mapping[str, ControllerQuestion] = field(repr=False, compare=False)
    _evaluation_by_id: Mapping[str, EvaluationRecord] = field(repr=False, compare=False)

    def context_for(self, question: ControllerQuestion) -> CaseContext:
        """Return context only for the exact controller record issued by this load."""
        known = self._questions_by_id.get(getattr(question, "question_id", ""))
        if known is not question or question._capability is not self._capability:
            raise ChestAgentBenchLoaderError("Question record was not issued by this dataset load.")
        return self.adapter.context_for(question.case_id)

    def evaluation_for(self, question_id: str) -> EvaluationRecord:
        """Retrieve ground truth through the explicit evaluation-only API."""
        try:
            return self._evaluation_by_id[question_id]
        except KeyError as exc:
            raise ChestAgentBenchLoaderError("Unknown question identifier.") from exc


class ChestAgentBenchLoader:
    """Load and validate a local dataset tree without changing it.

    Opaque IDs are ``cabimg_`` plus SHA-256 of a domain-separated canonical
    root-relative reference.  The digest is deterministic but does not expose a
    readable or reversible filesystem path.  Every generated ID is checked for
    collision before adapter construction.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    def load(self) -> ChestAgentBenchDataset:
        root = self._trusted_root()
        manifest = root / "metadata.jsonl"
        if not manifest.is_file() or manifest.is_symlink():
            raise ManifestIntegrityError("Pinned metadata manifest is unavailable.")
        metadata_hash = _sha256(manifest)
        self._validate_provenance(root, metadata_hash)

        questions: list[ControllerQuestion] = []
        evaluations: list[EvaluationRecord] = []
        entries: list[ImageAssetEntry] = []
        seen_question_ids: set[str] = set()
        seen_full_question_ids: set[str] = set()
        reference_ids: dict[str, str] = {}
        reference_cases: dict[str, str] = {}
        capability = object()

        try:
            lines = manifest.open(encoding="utf-8")
        except OSError as exc:
            raise ManifestIntegrityError("Pinned metadata manifest cannot be read.") from exc
        with lines:
            for row_index, line in enumerate(lines, start=1):
                row = _parse_row(line, row_index)
                _validate_row(row, row_index)
                question_id = row["question_id"]
                full_question_id = row["full_question_id"]
                if question_id in seen_question_ids or full_question_id in seen_full_question_ids:
                    raise ManifestRowError(f"Row {row_index}: duplicate question identifier.")
                seen_question_ids.add(question_id)
                seen_full_question_ids.add(full_question_id)

                image_ids: list[str] = []
                for reference in row["images"]:
                    _validate_asset_reference(root, reference, row_index)
                    image_id = reference_ids.get(reference)
                    if image_id is None:
                        image_id = _opaque_image_id(reference)
                        existing_reference = next((key for key, value in reference_ids.items() if value == image_id), None)
                        if existing_reference is not None and existing_reference != reference:
                            raise OpaqueImageIdCollisionError("Opaque image identifier collision detected.")
                        reference_ids[reference] = image_id
                    owner = reference_cases.setdefault(reference, row["case_id"])
                    if owner != row["case_id"]:
                        raise ManifestRowError(f"Row {row_index}: an image asset is assigned to multiple cases.")
                    image_ids.append(image_id)

                questions.append(
                    ControllerQuestion(
                        question_id=question_id,
                        full_question_id=full_question_id,
                        case_id=row["case_id"],
                        question=row["question"],
                        categories=row["categories"],
                        sections=row["sections"],
                        question_type=row["type"],
                        image_ids=tuple(image_ids),
                        _capability=capability,
                    )
                )
                evaluations.append(EvaluationRecord(question_id, row["answer"], row["explanation"]))

        if not questions:
            raise ManifestIntegrityError("Pinned metadata manifest contains no rows.")
        for reference, image_id in reference_ids.items():
            entries.append(ImageAssetEntry(reference_cases[reference], image_id, reference))
        try:
            adapter = CaseImageIdentifierAdapter.from_entries(root, entries)
        except ManifestValidationError as exc:
            raise ManifestIntegrityError("Validated manifest assets could not be registered.") from exc
        question_map = MappingProxyType({question.question_id: question for question in questions})
        evaluation_map = MappingProxyType({record.question_id: record for record in evaluations})
        return ChestAgentBenchDataset(
            tuple(questions), tuple(evaluations), adapter, metadata_hash, capability, question_map, evaluation_map
        )

    def _trusted_root(self) -> Path:
        if self.dataset_root.is_symlink() or not self.dataset_root.is_dir():
            raise ManifestIntegrityError("Dataset root must be an existing non-symlink directory.")
        return self.dataset_root.resolve(strict=True)

    @staticmethod
    def _validate_provenance(root: Path, metadata_hash: str) -> None:
        provenance = root / "ACQUISITION_PROVENANCE.json"
        if not provenance.exists():
            return  # Synthetic manifests intentionally do not require acquisition metadata.
        if provenance.is_symlink() or not provenance.is_file():
            raise ManifestIntegrityError("Acquisition provenance is invalid.")
        try:
            data = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestIntegrityError("Acquisition provenance cannot be validated.") from exc
        try:
            expected_hash = data["files"]["metadata.jsonl"]["sha256"]
        except (KeyError, TypeError) as exc:
            raise ManifestIntegrityError("Acquisition provenance is incomplete.") from exc
        if data.get("dataset_repository") != CANONICAL_REPOSITORY or data.get("revision") != PINNED_REVISION:
            raise ManifestIntegrityError("Acquisition provenance does not identify the pinned dataset.")
        if not isinstance(expected_hash, str) or expected_hash != metadata_hash:
            raise ManifestIntegrityError("Metadata checksum does not match acquisition provenance.")


def _parse_row(line: str, row_index: int) -> dict[str, object]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ManifestRowError(f"Row {row_index}: malformed JSON.") from exc
    if not isinstance(row, dict):
        raise ManifestRowError(f"Row {row_index}: expected an object.")
    return row


def _validate_row(row: Mapping[str, object], row_index: int) -> None:
    for field_name, expected_type in _REQUIRED_FIELDS.items():
        value = row.get(field_name)
        if not isinstance(value, expected_type) or (expected_type is str and not value):
            raise ManifestRowError(f"Row {row_index}: invalid required field '{field_name}'.")
    for list_name in ("images", "image_source_urls"):
        values = row[list_name]
        if any(not isinstance(value, str) or not value for value in values):
            raise ManifestRowError(f"Row {row_index}: invalid '{list_name}' value.")


def _validate_asset_reference(root: Path, reference: str, row_index: int) -> None:
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts or "\\" in reference:
        raise ManifestRowError(f"Row {row_index}: image reference is not root-relative.")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ManifestRowError(f"Row {row_index}: image reference is not permitted.")
    candidate = root / path
    if not candidate.is_file():
        raise ManifestRowError(f"Row {row_index}: image asset is unavailable.")
    try:
        candidate.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ManifestRowError(f"Row {row_index}: image reference escapes dataset root.") from exc


def _opaque_image_id(reference: str) -> str:
    return "cabimg_" + hashlib.sha256(("chestagentbench-image-v1:\0" + reference).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
