"""AST-only audit tests; concrete medical-tool modules are never imported."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from medrax.tool_registry import PAPER_TOOL_NAMES


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "chest_xray_report_generator": ("medrax/tools/report_generation.py", "ChestXRayReportGeneratorTool", "ChestXRayInput", {"image_path"}, {"torch", "transformers"}),
    "chest_xray_classifier": ("medrax/tools/classification.py", "ChestXRayClassifierTool", "ChestXRayInput", {"image_path"}, {"torch", "torchvision", "torchxrayvision"}),
    "chest_xray_segmentation": ("medrax/tools/segmentation.py", "ChestXRaySegmentationTool", "ChestXRaySegmentationInput", {"image_path", "organs"}, {"torch", "torchvision", "torchxrayvision"}),
    "xray_phrase_grounding": ("medrax/tools/grounding.py", "XRayPhraseGroundingTool", "XRayPhraseGroundingInput", {"image_path", "phrase", "max_new_tokens"}, {"torch", "transformers"}),
    "chest_xray_expert": ("medrax/tools/xray_vqa.py", "XRayVQATool", "XRayVQAToolInput", {"image_paths", "prompt", "max_new_tokens"}, {"torch", "transformers"}),
    "llava_med_qa": ("medrax/tools/llava_med.py", "LlavaMedTool", "LlavaMedInput", {"question", "image_path"}, {"torch"}),
}


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import): names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module: names.add(node.module.split(".")[0])
    return names


def _constant(class_node: ast.ClassDef, name: str) -> str:
    node = next(item for item in class_node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name)
    assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    return node.value.value


@pytest.mark.parametrize("tool_name", PAPER_TOOL_NAMES)
def test_registry_names_and_static_source_schema_match(tool_name: str) -> None:
    path, class_name, schema_name, fields, heavy = CONTRACTS[tool_name]
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tool_class, schema = _class(tree, class_name), _class(tree, schema_name)
    assert _constant(tool_class, "name") == tool_name
    schema_fields = {node.target.id for node in schema.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert schema_fields == fields and heavy <= _imports(tree)
    methods = {node.name for node in tool_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"__init__", "_run", "_arun"} <= methods


@pytest.mark.parametrize("tool_name", PAPER_TOOL_NAMES)
def test_every_concrete_constructor_has_static_model_loading_boundary(tool_name: str) -> None:
    path, class_name, _, _, _ = CONTRACTS[tool_name]
    constructor = next(node for node in _class(ast.parse((ROOT / path).read_text()), class_name).body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    source = ast.unparse(constructor)
    assert any(marker in source for marker in ("from_pretrained", "DenseNet", "PSPNet", "load_pretrained_model"))


def test_lightweight_registry_source_does_not_import_concrete_tools() -> None:
    source = (ROOT / "medrax/tool_registry.py").read_text(encoding="utf-8")
    assert "medrax.tools" not in source and "torch" not in source and "transformers" not in source


def test_invocation_boundary_fields_exclude_evaluation_paths_and_credentials() -> None:
    tree = ast.parse((ROOT / "medrax/benchmark/controller_invocation_adapter.py").read_text(encoding="utf-8"))
    fields = {node.target.id for node in _class(tree, "ControllerInvocationInput").body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert {"question", "image_ids", "provider", "model"} <= fields
    assert not fields & {"answer", "explanation", "image_path", "image_paths", "api_key", "case_context"}
