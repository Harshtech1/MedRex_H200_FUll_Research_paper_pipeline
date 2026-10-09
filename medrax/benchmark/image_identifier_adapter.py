"""Case-scoped opaque image-ID resolution for trusted benchmark dispatch."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
class ImageIdentifierError(ValueError): pass
class InvalidImageIdentifier(ImageIdentifierError): pass
class UnknownImageIdentifier(ImageIdentifierError): pass
class WrongCaseImageIdentifier(ImageIdentifierError): pass
class InvalidCaseContext(ImageIdentifierError): pass
class ManifestValidationError(ImageIdentifierError): pass

@dataclass(frozen=True)
class ImageAssetEntry:
    case_id: str
    image_id: str
    asset_path: str

@dataclass(frozen=True)
class CaseContext:
    case_id: str
    _token: object = field(repr=False, compare=False)

@dataclass(frozen=True)
class CaseImageIdentifierAdapter:
    _root: Path
    _cases: Mapping[str, Mapping[str, Path]]
    _owners: Mapping[str, str]
    _token: object = field(repr=False, compare=False)

    @classmethod
    def from_entries(cls, asset_root: str | Path, entries: Iterable[ImageAssetEntry]):
        root_path = Path(asset_root)
        if root_path.is_symlink() or not root_path.is_dir():
            raise ManifestValidationError("Dataset asset root must be an existing non-symlink directory.")
        root = root_path.resolve(strict=True); cases: dict[str, dict[str, Path]] = {}; owners: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, ImageAssetEntry): raise ManifestValidationError("Manifest entries must be ImageAssetEntry instances.")
            _valid("case ID", entry.case_id); _valid("image identifier", entry.image_id)
            if entry.image_id in owners: raise ManifestValidationError("Duplicate image identifier in manifest.")
            cases.setdefault(entry.case_id, {})[entry.image_id] = _asset(root, entry.asset_path); owners[entry.image_id] = entry.case_id
        if not cases: raise ManifestValidationError("Manifest must contain at least one registered asset.")
        return cls(root, MappingProxyType({k: MappingProxyType(dict(v)) for k,v in cases.items()}), MappingProxyType(dict(owners)), object())

    def context_for(self, case_id: str) -> CaseContext:
        _valid("case ID", case_id)
        if case_id not in self._cases: raise InvalidCaseContext("Active benchmark case is not registered.")
        return CaseContext(case_id, self._token)

    def resolve(self, context: CaseContext, image_id: str) -> Path:
        if not isinstance(context, CaseContext) or context._token is not self._token: raise InvalidCaseContext("Active benchmark case context is invalid.")
        _valid("image identifier", image_id); owner = self._owners.get(image_id)
        if owner is None: raise UnknownImageIdentifier("Image identifier is not registered.")
        if owner != context.case_id: raise WrongCaseImageIdentifier("Image identifier is not authorized for the active case.")
        return self._cases[context.case_id][image_id]

    def registered_ids(self, context: CaseContext) -> tuple[str, ...]:
        if not isinstance(context, CaseContext) or context._token is not self._token: raise InvalidCaseContext("Active benchmark case context is invalid.")
        return tuple(self._cases[context.case_id])

def _valid(label: str, value: object) -> None:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value): raise InvalidImageIdentifier(f"{label.capitalize()} must be a non-empty opaque token.")

def _asset(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip(): raise ManifestValidationError("Manifest asset path must be a non-empty root-relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value: raise ManifestValidationError("Manifest asset path is not permitted.")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink(): raise ManifestValidationError("Manifest asset path is not permitted.")
    candidate = root / path
    if not candidate.is_file(): raise ManifestValidationError("Registered asset file is unavailable.")
    resolved = candidate.resolve(strict=True)
    try: resolved.relative_to(root)
    except ValueError as exc: raise ManifestValidationError("Manifest asset path is outside the dataset asset root.") from exc
    return resolved
