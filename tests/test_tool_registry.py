"""Offline mock-registry and static six-tool contract tests."""

from __future__ import annotations

import ast
import builtins
import importlib
from pathlib import Path
import socket
import sys
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from medrax.tool_registry import PAPER_TOOL_NAMES, ToolRegistry, ToolRegistryError


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACTS = {
    "chest_xray_report_generator": {
        "path": "medrax/tools/report_generation.py",
        "input": "ChestXRayInput",
        "fields": {"image_path": (True, None)},
    },
    "chest_xray_classifier": {
        "path": "medrax/tools/classification.py",
        "input": "ChestXRayInput",
        "fields": {"image_path": (True, None)},
    },
    "chest_xray_segmentation": {
        "path": "medrax/tools/segmentation.py",
        "input": "ChestXRaySegmentationInput",
        "fields": {"image_path": (True, None), "organs": (False, None)},
    },
    "xray_phrase_grounding": {
        "path": "medrax/tools/grounding.py",
        "input": "XRayPhraseGroundingInput",
        "fields": {"image_path": (True, None), "phrase": (True, None), "max_new_tokens": (False, 300)},
    },
    "chest_xray_expert": {
        "path": "medrax/tools/xray_vqa.py",
        "input": "XRayVQAToolInput",
        "fields": {"image_paths": (True, None), "prompt": (True, None), "max_new_tokens": (False, 512)},
    },
    "llava_med_qa": {
        "path": "medrax/tools/llava_med.py",
        "input": "LlavaMedInput",
        "fields": {"question": (True, None), "image_path": (False, None)},
    },
}


class ImagePathInput(BaseModel):
    image_path: str


class SegmentationInput(BaseModel):
    image_path: str
    organs: list[str] | None = None


class GroundingInput(BaseModel):
    image_path: str
    phrase: str
    max_new_tokens: int = 300


class VqaInput(BaseModel):
    image_paths: list[str]
    prompt: str
    max_new_tokens: int = 512


class LlavaInput(BaseModel):
    question: str
    image_path: str | None = None


SCHEMA_FIXTURES: dict[str, tuple[type[BaseModel], dict[str, Any], str]] = {
    "chest_xray_report_generator": (ImagePathInput, {"image_path": "synthetic.png"}, "image_path"),
    "chest_xray_classifier": (ImagePathInput, {"image_path": "synthetic.png"}, "image_path"),
    "chest_xray_segmentation": (SegmentationInput, {"image_path": "synthetic.png"}, "image_path"),
    "xray_phrase_grounding": (
        GroundingInput,
        {"image_path": "synthetic.png", "phrase": "synthetic finding"},
        "image_path",
    ),
    "chest_xray_expert": (
        VqaInput,
        {"image_paths": ["synthetic.png"], "prompt": "synthetic question"},
        "image_paths",
    ),
    "llava_med_qa": (LlavaInput, {"question": "synthetic question"}, "question"),
}


class MockTool(BaseTool):
    """A local-only tool whose result follows MedRAX's (output, metadata) convention."""

    name: str
    description: str = "Synthetic test tool"
    args_schema: type[BaseModel]
    response: tuple[dict[str, str], dict[str, str]] = (
        {"response": "synthetic"},
        {"analysis_status": "synthetic"},
    )
    calls: list[dict[str, Any]] = Field(default_factory=list)
    fail: bool = False

    def _run(self, **kwargs: Any) -> tuple[dict[str, str], dict[str, str]]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("synthetic failure")
        return self.response

    async def _arun(self, **kwargs: Any) -> tuple[dict[str, str], dict[str, str]]:
        return self._run(**kwargs)


def fake_tool(name: str, *, fail: bool = False) -> MockTool:
    schema, _, _ = SCHEMA_FIXTURES[name]
    return MockTool(name=name, args_schema=schema, fail=fail)


def six_fake_tools() -> list[MockTool]:
    return [fake_tool(name) for name in PAPER_TOOL_NAMES]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assigned_constant(class_node: ast.ClassDef, field: str) -> Any:
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field:
            if isinstance(node.value, ast.Constant):
                return node.value.value
    raise AssertionError(f"{field} constant not found")


@pytest.mark.parametrize("tool_name", PAPER_TOOL_NAMES)
def test_static_source_contract_matches_canonical_name_and_input_fields(tool_name: str) -> None:
    contract = SOURCE_CONTRACTS[tool_name]
    tree = ast.parse((ROOT / contract["path"]).read_text())
    tool_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "name"
            for item in node.body
        )
    )
    assert _assigned_constant(tool_class, "name") == tool_name
    input_class = _class(tree, contract["input"])
    source_fields = {
        node.target.id: node.value
        for node in input_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert set(source_fields) == set(contract["fields"])
    for field, (required, default) in contract["fields"].items():
        value = source_fields[field]
        if required:
            assert isinstance(value, ast.Call)
            assert any(isinstance(argument, ast.Constant) and argument.value is Ellipsis for argument in value.args)
        elif default is not None:
            assert isinstance(value, ast.Call)
            positional_default = any(
                isinstance(argument, ast.Constant) and argument.value == default for argument in value.args
            )
            keyword_default = any(
                isinstance(keyword.value, ast.Constant) and keyword.value.value == default
                for keyword in value.keywords
            )
            assert positional_default or keyword_default


@pytest.mark.parametrize("tool_name", PAPER_TOOL_NAMES)
def test_schema_fixtures_validate_minimal_input_and_reject_missing_required_field(tool_name: str) -> None:
    schema, valid_input, missing_field = SCHEMA_FIXTURES[tool_name]
    model = schema.model_validate(valid_input)
    assert model.model_dump().items() >= valid_input.items()
    invalid = dict(valid_input)
    invalid.pop(missing_field)
    with pytest.raises(Exception):
        schema.model_validate(invalid)


@pytest.mark.parametrize("tool_name", PAPER_TOOL_NAMES)
def test_schema_fixtures_follow_current_default_extra_handling(tool_name: str) -> None:
    schema, valid_input, _ = SCHEMA_FIXTURES[tool_name]
    model = schema.model_validate({**valid_input, "unknown_argument": "ignored-by-default"})
    assert "unknown_argument" not in model.model_dump()


def test_complete_six_tool_registry_has_unique_names_and_valid_schemas() -> None:
    registry = ToolRegistry.for_six_paper_tools(six_fake_tools())
    assert registry.names == PAPER_TOOL_NAMES


def test_dispatch_invokes_only_the_selected_mock_and_preserves_tuple_result() -> None:
    tools = six_fake_tools()
    selected = tools[0]
    result = ToolRegistry.for_six_paper_tools(tools).dispatch(
        selected.name, {"image_path": "synthetic.png"}
    )
    assert result.succeeded
    assert result.result == selected.response
    assert selected.calls == [{"image_path": "synthetic.png"}]
    assert all(not tool.calls for tool in tools[1:])


def test_duplicate_names_fail_instead_of_overwriting() -> None:
    duplicate = [fake_tool("chest_xray_classifier"), fake_tool("chest_xray_classifier")]
    with pytest.raises(ToolRegistryError, match="Duplicate"):
        ToolRegistry(duplicate)


def test_empty_and_incomplete_registries_fail_clearly() -> None:
    with pytest.raises(ToolRegistryError, match="at least one"):
        ToolRegistry([])
    with pytest.raises(ToolRegistryError, match="required canonical"):
        ToolRegistry.for_six_paper_tools(six_fake_tools()[:-1])


def test_lookup_failure_and_selected_tool_exception_are_explicit_safe_failures() -> None:
    tool = fake_tool("chest_xray_classifier", fail=True)
    registry = ToolRegistry([tool])
    with pytest.raises(ToolRegistryError, match="No tool"):
        registry.get("not-registered")
    result = registry.dispatch("chest_xray_classifier", {"image_path": "synthetic.png"})
    assert not result.succeeded
    assert result.result is None
    assert result.error == "Selected tool execution failed."


def test_registry_import_has_no_network_or_heavy_dependency_side_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {"torch", "transformers", "openai", "google", "gradio", "requests"}
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in blocked:
            raise AssertionError(f"registry import attempted heavyweight import: {name}")
        return original_import(name, *args, **kwargs)

    def forbidden_connection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("registry import attempted network access")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", forbidden_connection)
    sys.modules.pop("medrax.tool_registry", None)
    module = importlib.import_module("medrax.tool_registry")
    assert module.PAPER_TOOL_NAMES == PAPER_TOOL_NAMES
