"""Offline validation for explicit MedRAX controller selection.

This module intentionally contains no provider SDK imports and constructs no clients.
Gemini is represented as the documented reconstruction substitution for the paper's
GPT-4o controller; selecting it here does not assert that a Gemini client is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from collections.abc import Mapping


DOCUMENTED_GEMINI_MODEL = "gemini-3.5-flash-lite"
DOCUMENTED_GEMINI_TEMPERATURE = 0.2
DOCUMENTED_GEMINI_TOP_P = 0.95


class ControllerConfigurationError(ValueError):
    """Raised for a safe, actionable controller configuration error."""


class ControllerProvider(StrEnum):
    """Providers represented by the current controller configuration contract."""

    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass(frozen=True)
class ControllerConfig:
    """Validated controller settings, separate from client construction.

    Credentials are deliberately not read or stored. A later provider-integration
    layer must validate the selected provider's credential without changing this
    provider or model.
    """

    provider: ControllerProvider
    model: str
    temperature: float
    top_p: float

    @classmethod
    def from_values(
        cls,
        *,
        provider: object,
        model: object,
        temperature: object,
        top_p: object,
    ) -> "ControllerConfig":
        return cls(
            provider=_parse_provider(provider),
            model=_parse_model(model),
            temperature=_parse_float("CONTROLLER_TEMPERATURE", temperature, 0.0, 2.0),
            top_p=_parse_float("CONTROLLER_TOP_P", top_p, 0.0, 1.0, lower_exclusive=True),
        )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "ControllerConfig":
        """Build settings from explicit, non-secret controller environment values."""
        provider = _parse_provider(environ.get("CONTROLLER_PROVIDER"))
        model_key = f"{provider.value.upper()}_MODEL"
        return cls.from_values(
            provider=provider,
            model=environ.get(model_key),
            temperature=environ.get("CONTROLLER_TEMPERATURE"),
            top_p=environ.get("CONTROLLER_TOP_P"),
        )


def _parse_provider(value: object) -> ControllerProvider:
    if not isinstance(value, str) or not value.strip():
        raise ControllerConfigurationError(
            "CONTROLLER_PROVIDER must explicitly be set to gemini or openai."
        )
    try:
        return ControllerProvider(value.strip().lower())
    except ValueError as exc:
        raise ControllerConfigurationError(
            "CONTROLLER_PROVIDER must be one of: gemini, openai."
        ) from exc


def _parse_model(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControllerConfigurationError("The selected provider model must be a non-empty string.")
    return value.strip()


def _parse_float(
    name: str,
    value: object,
    lower: float,
    upper: float,
    *,
    lower_exclusive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ControllerConfigurationError(f"{name} must be a finite number.")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ControllerConfigurationError(f"{name} must be a finite number.") from exc
    if not math.isfinite(parsed) or parsed > upper or (parsed <= lower if lower_exclusive else parsed < lower):
        comparator = f"({lower}, {upper}]" if lower_exclusive else f"[{lower}, {upper}]"
        raise ControllerConfigurationError(f"{name} must be in {comparator}.")
    return parsed
