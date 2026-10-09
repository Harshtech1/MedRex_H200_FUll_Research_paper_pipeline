"""Synthetic path-safe preparation tests; no concrete tools are imported."""
from __future__ import annotations

import json, os, sys
from pathlib import Path
import pytest

from medrax.benchmark.chestagentbench_loader import ChestAgentBenchLoader
from medrax.benchmark.image_identifier_adapter import CaseContext, CaseImageIdentifierAdapter
from medrax.benchmark.trusted_tool_dispatch import TrustedToolDispatcher, TrustedToolDispatchError
from medrax.tool_registry import PAPER_TOOL_NAMES


def _dataset(root: Path):
    (root / "figures").mkdir(parents=True)
    for image in ("a.jpg", "b.jpg", "c.jpg"): (root / "figures" / image).write_bytes(b"synthetic")
    rows=[]
    for case, images in (("case-a", ["a.jpg", "b.jpg"]), ("case-b", ["c.jpg"])):
        rows.append({"answer":"synthetic-answer","case_id":case,"categories":"synthetic","explanation":"synthetic-explanation","full_question_id":"full-"+case,"image_source_urls":["https://example.invalid"],"images":["figures/"+image for image in images],"question":"synthetic-question","question_id":"q-"+case,"sections":"synthetic","type":"synthetic"})
    (root / "metadata.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    return ChestAgentBenchLoader(root).load()


@pytest.fixture
def dispatch(tmp_path: Path):
    dataset=_dataset(tmp_path); return dataset, TrustedToolDispatcher(dataset.adapter)


def test_all_six_mappings_and_no_concrete_imports(dispatch) -> None:
    dataset, dispatcher=dispatch; context=dataset.context_for(dataset.questions[0]); image=dataset.questions[0].image_ids[0]
    specs={"chest_xray_report_generator":({}, {"image_path"}),"chest_xray_classifier":({}, {"image_path"}),"chest_xray_segmentation":({"organs":["Heart"]}, {"image_path","organs"}),"xray_phrase_grounding":({"phrase":"synthetic"}, {"image_path","phrase","max_new_tokens"}),"chest_xray_expert":({"prompt":"synthetic"}, {"image_paths","prompt","max_new_tokens"}),"llava_med_qa":({"question":"synthetic"}, {"question","image_path"})}
    assert set(specs)==set(PAPER_TOOL_NAMES)
    for name,(arguments,keys) in specs.items(): assert set(dispatcher.prepare(context=context,tool_name=name,image_ids=(image,),arguments=arguments).execution_arguments())==keys
    assert not any(name.startswith("medrax.tools.") for name in sys.modules)


def test_case_authorization_unknown_duplicate_and_raw_path_rejected(dispatch) -> None:
    dataset, dispatcher=dispatch; a,b=dataset.questions; context=dataset.context_for(a)
    for ids in ((b.image_ids[0],),("unknown",),(a.image_ids[0],a.image_ids[0])):
        with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=ids,arguments={})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(a.image_ids[0],),arguments={"image_path":"/tmp/x"})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(a.image_ids[0],),arguments={"case_id":"case-b"})
    forged = CaseContext("case-a", object())
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=forged,tool_name="chest_xray_classifier",image_ids=(a.image_ids[0],),arguments={})


def test_path_escape_symlink_missing_and_redaction(dispatch, tmp_path: Path) -> None:
    dataset, dispatcher=dispatch; question=dataset.questions[0]; context=dataset.context_for(question)
    path=dataset.adapter.resolve(context,question.image_ids[0]); path.unlink()
    with pytest.raises(TrustedToolDispatchError) as err: dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(question.image_ids[0],),arguments={})
    assert str(tmp_path) not in str(err.value)
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=("../x",),arguments={})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(str(tmp_path / "outside"),),arguments={})
    path.mkdir()
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(question.image_ids[0],),arguments={})
    path.rmdir()
    outside=tmp_path / "outside"; outside.write_bytes(b"outside")
    try: os.symlink(outside, path)
    except OSError: pytest.skip("symlinks unavailable")
    if path.is_symlink():
        with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_classifier",image_ids=(question.image_ids[0],),arguments={})


def test_argument_validation_multi_image_and_ground_truth_exclusion(dispatch) -> None:
    dataset, dispatcher=dispatch; a,b=dataset.questions; context=dataset.context_for(a); image=a.image_ids[0]
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="unknown",image_ids=(image,),arguments={})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="xray_phrase_grounding",image_ids=(image,),arguments={})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_segmentation",image_ids=(image,),arguments={"organs":["not-an-organ"]})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_expert",image_ids=(),arguments={"prompt":"x"})
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="chest_xray_expert",image_ids=(image,),arguments={"prompt":"x","credential":"secret"})
    assert len(dispatcher.prepare(context=context,tool_name="chest_xray_expert",image_ids=a.image_ids,arguments={"prompt":"x"}).execution_arguments()["image_paths"]) == 2
    with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="llava_med_qa",image_ids=(image,image),arguments={"question":"x"})
    for field in ("answer", "explanation", "ground_truth", "label", "labels"):
        with pytest.raises(TrustedToolDispatchError): dispatcher.prepare(context=context,tool_name="llava_med_qa",image_ids=(),arguments={"question":"x",field:"leak"})


def test_prepared_representation_hides_paths_and_has_no_execution(dispatch) -> None:
    dataset, dispatcher=dispatch; question=dataset.questions[0]; call=dispatcher.prepare(context=dataset.context_for(question),tool_name="llava_med_qa",image_ids=(),arguments={"question":"synthetic"})
    assert "figures" not in repr(call) and ".jpg" not in repr(call) and call.public_metadata=={"tool_name":"llava_med_qa","image_ids":()}
