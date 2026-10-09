"""Trusted preparation of six-tool arguments from opaque authorized image IDs.

This module prepares arguments only. It neither imports concrete tools nor
executes them. Paths are resolved exclusively by ``CaseImageIdentifierAdapter``
under a loader-issued ``CaseContext`` and are hidden from normal representations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from medrax.tool_registry import PAPER_TOOL_NAMES

from .image_identifier_adapter import CaseContext, CaseImageIdentifierAdapter, ImageIdentifierError


class TrustedToolDispatchError(ValueError):
    """Rejected authorization or tool-argument preparation without path disclosure."""


_SINGLE_IMAGE = {"chest_xray_report_generator", "chest_xray_classifier", "chest_xray_segmentation", "xray_phrase_grounding"}
_GROUND_TRUTH = {"answer", "explanation", "ground_truth", "label", "labels"}
_SEGMENTATION_ORGANS = frozenset({"Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Left Lung", "Right Lung", "Left Hilus Pulmonis", "Right Hilus Pulmonis", "Heart", "Aorta", "Facies Diaphragmatica", "Mediastinum", "Weasand", "Spine"})


@dataclass(frozen=True)
class PreparedToolCall:
    """Trusted, unexecuted tool arguments; local paths are intentionally private."""

    tool_name: str
    image_ids: tuple[str, ...]
    _arguments: Mapping[str, object] = field(repr=False, compare=False)

    @property
    def public_metadata(self) -> Mapping[str, object]:
        return MappingProxyType({"tool_name": self.tool_name, "image_ids": self.image_ids})

    def execution_arguments(self) -> Mapping[str, object]:
        """For a future separately-authorized execution adapter only."""
        return MappingProxyType(dict(self._arguments))


class TrustedToolDispatcher:
    """Prepare explicit source-audited mappings using an existing image adapter."""

    def __init__(self, adapter: CaseImageIdentifierAdapter) -> None:
        if not isinstance(adapter, CaseImageIdentifierAdapter):
            raise TrustedToolDispatchError("A trusted CaseImageIdentifierAdapter is required.")
        self._adapter = adapter

    def prepare(self, *, context: CaseContext, tool_name: str, image_ids: tuple[str, ...] | list[str], arguments: Mapping[str, object]) -> PreparedToolCall:
        if tool_name not in PAPER_TOOL_NAMES:
            raise TrustedToolDispatchError("Tool name is not in the six-tool allowlist.")
        if not isinstance(image_ids, (tuple, list)) or not all(isinstance(image_id, str) for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
            raise TrustedToolDispatchError("Image identifiers must be a duplicate-free sequence of opaque identifiers.")
        if not isinstance(arguments, Mapping) or _GROUND_TRUTH & set(arguments):
            raise TrustedToolDispatchError("Arguments are invalid or include evaluation-only fields.")
        ids = tuple(image_ids)
        self._check_image_count(tool_name, ids)
        paths = self._resolve(context, ids)
        prepared = self._map_arguments(tool_name, paths, arguments)
        return PreparedToolCall(tool_name, ids, MappingProxyType(prepared))

    def _resolve(self, context: CaseContext, image_ids: tuple[str, ...]) -> tuple[str, ...]:
        paths = []
        for image_id in image_ids:
            try:
                path = self._adapter.resolve(context, image_id)
            except ImageIdentifierError as exc:
                raise TrustedToolDispatchError("Image identifier is not authorized for this trusted case context.") from exc
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self._adapter._root)
            except (OSError, ValueError):
                raise TrustedToolDispatchError("Authorized image asset is no longer within the approved dataset root.")
            if resolved != path or path.is_symlink() or not path.is_file():
                raise TrustedToolDispatchError("Authorized image asset is no longer an available regular file.")
            paths.append(str(path))
        return tuple(paths)

    @staticmethod
    def _check_image_count(tool_name: str, image_ids: tuple[str, ...]) -> None:
        if tool_name in _SINGLE_IMAGE and len(image_ids) != 1:
            raise TrustedToolDispatchError("This tool requires exactly one authorized image identifier.")
        if tool_name == "chest_xray_expert" and not image_ids:
            raise TrustedToolDispatchError("CheXagent VQA requires one or more authorized image identifiers.")
        if tool_name == "llava_med_qa" and len(image_ids) > 1:
            raise TrustedToolDispatchError("LLaVA-Med accepts at most one authorized image identifier.")

    @staticmethod
    def _map_arguments(tool_name: str, paths: tuple[str, ...], arguments: Mapping[str, object]) -> dict[str, object]:
        if tool_name in {"chest_xray_report_generator", "chest_xray_classifier"}:
            _exact(arguments, set()); return {"image_path": paths[0]}
        if tool_name == "chest_xray_segmentation":
            _exact(arguments, {"organs"}); organs = arguments.get("organs")
            if organs is not None and (not isinstance(organs, list) or any(not isinstance(item, str) or item not in _SEGMENTATION_ORGANS for item in organs)):
                raise TrustedToolDispatchError("Segmentation organs are invalid.")
            return {"image_path": paths[0], "organs": organs}
        if tool_name == "xray_phrase_grounding":
            phrase, maximum = _text_and_limit(arguments, "phrase", 300)
            return {"image_path": paths[0], "phrase": phrase, "max_new_tokens": maximum}
        if tool_name == "chest_xray_expert":
            prompt, maximum = _text_and_limit(arguments, "prompt", 512)
            return {"image_paths": list(paths), "prompt": prompt, "max_new_tokens": maximum}
        question, _ = _text_and_limit(arguments, "question", None)
        return {"question": question, "image_path": paths[0] if paths else None}


def _exact(arguments: Mapping[str, object], allowed: set[str]) -> None:
    if set(arguments) - allowed:
        raise TrustedToolDispatchError("Unsupported tool arguments were supplied.")


def _text_and_limit(arguments: Mapping[str, object], text_key: str, default: int | None) -> tuple[str, int | None]:
    allowed = {text_key} | ({"max_new_tokens"} if default is not None else set())
    _exact(arguments, allowed)
    text = arguments.get(text_key)
    if not isinstance(text, str):
        raise TrustedToolDispatchError(f"Required {text_key} argument is invalid.")
    if default is None:
        return text, None
    maximum = arguments.get("max_new_tokens", default)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise TrustedToolDispatchError("max_new_tokens must be an integer.")
    return text, maximum
