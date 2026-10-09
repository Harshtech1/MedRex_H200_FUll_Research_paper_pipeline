"""Synthetic case-scoped identifier security tests."""
import os
from pathlib import Path
import pytest
from medrax.benchmark.image_identifier_adapter import *

def put(path: Path): path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic", encoding="utf-8")

@pytest.fixture
def adapter(tmp_path):
    root=tmp_path/"assets"; put(root/"shared.txt")
    return CaseImageIdentifierAdapter.from_entries(root,[ImageAssetEntry("case-a","img-a","shared.txt"),ImageAssetEntry("case-b","img-b","shared.txt")])

def test_valid_deterministic_same_case(adapter):
    c=adapter.context_for("case-a"); assert adapter.resolve(c,"img-a")==adapter.resolve(c,"img-a"); assert adapter.registered_ids(c)==("img-a",)
def test_unknown_and_cross_case(adapter):
    with pytest.raises(UnknownImageIdentifier): adapter.resolve(adapter.context_for("case-a"),"fake")
    assert adapter.resolve(adapter.context_for("case-a"),"img-a")==adapter.resolve(adapter.context_for("case-b"),"img-b")
    with pytest.raises(WrongCaseImageIdentifier): adapter.resolve(adapter.context_for("case-b"),"img-a")
@pytest.mark.parametrize("bad",["/tmp/x","../x","..\\x","https://x","","a/b"])
def test_path_like_ids_rejected(adapter,bad):
    with pytest.raises(InvalidImageIdentifier): adapter.resolve(adapter.context_for("case-a"),bad)
def test_empty_case_and_context_override_rejected(adapter):
    with pytest.raises(InvalidImageIdentifier): adapter.context_for("")
    with pytest.raises(InvalidCaseContext): adapter.resolve(object(),"img-a")
    with pytest.raises(TypeError): adapter.resolve(adapter.context_for("case-a"),"img-a",case_id="case-b")
def test_manifest_validation_and_redaction(tmp_path):
    root=tmp_path/"assets"; root.mkdir(); outside=tmp_path/"outside.txt"; put(outside)
    for entry in [ImageAssetEntry("case-a","id",str(outside)),ImageAssetEntry("case-a","id","missing.txt")]:
        with pytest.raises(ManifestValidationError) as err: CaseImageIdentifierAdapter.from_entries(root,[entry])
        assert str(root) not in str(err.value) and str(outside) not in str(err.value)
    put(root/"one.txt")
    with pytest.raises(ManifestValidationError): CaseImageIdentifierAdapter.from_entries(root,[ImageAssetEntry("case-a","same","one.txt"),ImageAssetEntry("case-b","same","one.txt")])
    with pytest.raises(ManifestValidationError): CaseImageIdentifierAdapter.from_entries(root,[object()])
def test_symlink_and_immutable_mapping(tmp_path):
    root=tmp_path/"assets"; root.mkdir(); outside=tmp_path/"out.txt"; put(outside)
    try: os.symlink(outside,root/"escape.txt")
    except OSError: pytest.skip("symlinks unavailable")
    with pytest.raises(ManifestValidationError): CaseImageIdentifierAdapter.from_entries(root,[ImageAssetEntry("case-a","escape","escape.txt")])
    put(root/"one.txt"); entries=[ImageAssetEntry("case-a","img-a","one.txt")]; a=CaseImageIdentifierAdapter.from_entries(root,entries); entries.append(ImageAssetEntry("case-b","img-b","one.txt"))
    assert a.registered_ids(a.context_for("case-a"))==("img-a",)
    with pytest.raises(TypeError): a._cases["case-a"]["new"] = root/"one.txt"
@pytest.mark.parametrize("case,image,error",[("case-b","img-a",WrongCaseImageIdentifier),("case-a","img-b",WrongCaseImageIdentifier),("case-a","fake",UnknownImageIdentifier),("case-a","/tmp/x",InvalidImageIdentifier),("case-a","https://x",InvalidImageIdentifier),("case-a","../x",InvalidImageIdentifier)])
def test_adversarial_matrix(adapter,case,image,error):
    with pytest.raises(error): adapter.resolve(adapter.context_for(case),image)
