"""Offline tests for explicit controller-selection configuration."""

import importlib
import socket
import sys

import pytest

from medrax.controller_config import (
    DOCUMENTED_GEMINI_MODEL,
    DOCUMENTED_GEMINI_TEMPERATURE,
    DOCUMENTED_GEMINI_TOP_P,
    ControllerConfig,
    ControllerConfigurationError,
    ControllerProvider,
)


def gemini_environment(**overrides: str) -> dict[str, str]:
    values = {
        "CONTROLLER_PROVIDER": "gemini",
        "GEMINI_MODEL": DOCUMENTED_GEMINI_MODEL,
        "CONTROLLER_TEMPERATURE": str(DOCUMENTED_GEMINI_TEMPERATURE),
        "CONTROLLER_TOP_P": str(DOCUMENTED_GEMINI_TOP_P),
    }
    values.update(overrides)
    return values


def test_explicit_gemini_selection_preserves_documented_configuration() -> None:
    config = ControllerConfig.from_values(
        provider="gemini",
        model=DOCUMENTED_GEMINI_MODEL,
        temperature=DOCUMENTED_GEMINI_TEMPERATURE,
        top_p=DOCUMENTED_GEMINI_TOP_P,
    )
    assert config.provider is ControllerProvider.GEMINI
    assert config.model == DOCUMENTED_GEMINI_MODEL
    assert config.temperature == DOCUMENTED_GEMINI_TEMPERATURE
    assert config.top_p == DOCUMENTED_GEMINI_TOP_P


def test_gemini_environment_contract_is_explicit_and_non_secret() -> None:
    config = ControllerConfig.from_environment(gemini_environment())
    assert config.provider is ControllerProvider.GEMINI
    assert config.model == DOCUMENTED_GEMINI_MODEL


@pytest.mark.parametrize("provider", [None, "", "   "])
def test_missing_or_empty_provider_fails_closed(provider: object) -> None:
    with pytest.raises(ControllerConfigurationError, match="CONTROLLER_PROVIDER"):
        ControllerConfig.from_values(
            provider=provider,
            model=DOCUMENTED_GEMINI_MODEL,
            temperature=0.2,
            top_p=0.95,
        )


def test_unsupported_provider_does_not_echo_untrusted_input() -> None:
    secret_like_value = "unsupported-provider-secret"
    with pytest.raises(ControllerConfigurationError) as error:
        ControllerConfig.from_values(
            provider=secret_like_value,
            model=DOCUMENTED_GEMINI_MODEL,
            temperature=0.2,
            top_p=0.95,
        )
    assert secret_like_value not in str(error.value)


@pytest.mark.parametrize(
    ("model", "temperature", "top_p"),
    [
        ("", 0.2, 0.95),
        (DOCUMENTED_GEMINI_MODEL, "not-a-number", 0.95),
        (DOCUMENTED_GEMINI_MODEL, True, 0.95),
        (DOCUMENTED_GEMINI_MODEL, 2.1, 0.95),
        (DOCUMENTED_GEMINI_MODEL, 0.2, 0),
        (DOCUMENTED_GEMINI_MODEL, 0.2, 1.1),
    ],
)
def test_malformed_configuration_fails_closed(
    model: object, temperature: object, top_p: object
) -> None:
    with pytest.raises(ControllerConfigurationError):
        ControllerConfig.from_values(
            provider="gemini", model=model, temperature=temperature, top_p=top_p
        )


def test_absent_credentials_do_not_block_pure_configuration_validation() -> None:
    config = ControllerConfig.from_environment(gemini_environment())
    assert config.provider is ControllerProvider.GEMINI


def test_missing_gemini_credential_never_falls_back_to_openai() -> None:
    environment = gemini_environment(OPENAI_API_KEY="present-but-unused")
    config = ControllerConfig.from_environment(environment)
    assert config.provider is ControllerProvider.GEMINI


def test_missing_openai_credential_never_falls_back_to_gemini() -> None:
    config = ControllerConfig.from_environment(
        {
            "CONTROLLER_PROVIDER": "openai",
            "OPENAI_MODEL": "gpt-4o",
            "CONTROLLER_TEMPERATURE": "0.2",
            "CONTROLLER_TOP_P": "0.95",
            "GEMINI_API_KEY": "present-but-unused",
        }
    )
    assert config.provider is ControllerProvider.OPENAI
    assert config.model == "gpt-4o"


def test_explicit_model_is_never_substituted() -> None:
    config = ControllerConfig.from_values(
        provider="gemini", model="chosen-model", temperature=0.2, top_p=0.95
    )
    assert config.model == "chosen-model"


def test_error_text_does_not_leak_model_value() -> None:
    secret_like_model = "model-secret-value"
    with pytest.raises(ControllerConfigurationError) as error:
        ControllerConfig.from_values(
            provider="gemini", model=secret_like_model, temperature="invalid", top_p=0.95
        )
    assert secret_like_model not in str(error.value)


def test_import_has_no_network_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("configuration import attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_connection)
    sys.modules.pop("medrax.controller_config", None)
    module = importlib.import_module("medrax.controller_config")
    assert module.DOCUMENTED_GEMINI_MODEL == DOCUMENTED_GEMINI_MODEL
