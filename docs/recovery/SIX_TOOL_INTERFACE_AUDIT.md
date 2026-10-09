# Six-Tool Interface and Dependency Audit

Scope: static source inspection and CPU-only AST/registry contract tests on
2026-10-10. No concrete tool module was imported, no constructor was called,
and no model, image, API, credential, or network operation was executed.

## Evidence and safety boundary

`medrax/tool_registry.py` is safe to import in the CPU environment: it imports
only Pydantic, LangChain Core, and supplied `BaseTool` instances. It does not
import `medrax.tools.*`, construct tools, load weights, or select a provider.
The six concrete modules are not safe audit imports: each has top-level imports
of unavailable/heavy frameworks, and every tool constructor initializes model
state. The current Python 3.12 CPU environment has no authorization or installed
model stack for their execution.

All source paths below return `(output, metadata)` from `_run` and use `_arun`
as a synchronous forwarding wrapper. Error payload details are source-defined;
they were not exercised. Several metadata dictionaries retain caller-supplied
image paths, which is incompatible with the path-free Task 14 controller
boundary. A future integration must resolve opaque IDs inside trusted dispatch,
then pass only approved local paths directly to a tool adapter—not to the
generic controller invocation input.

| Registry name / source | Public class, schema, callable contract | Constructor / weights / imports | CPU audit status and unverified evidence |
|---|---|---|---|
| `chest_xray_report_generator` — `medrax/tools/report_generation.py` | `ChestXRayReportGeneratorTool`; `ChestXRayInput(image_path: str)`; `_run(image_path, run_manager=None) -> tuple[str, dict]`. | Top-level `torch`, PIL, `transformers`; constructor loads `IAMJB/chexpert-mimic-cxr-findings-baseline` and `IAMJB/chexpert-mimic-cxr-impression-baseline` via `from_pretrained`, defaults to `/model-weights`, `cuda`. | **STATIC_ONLY / NOT_TESTED**. Constructor can load/download weights. Verify later with approved cached CPU weights and synthetic image. |
| `chest_xray_classifier` — `medrax/tools/classification.py` | `ChestXRayClassifierTool`; `ChestXRayInput(image_path: str)`; `_run(image_path, run_manager=None) -> tuple[dict[str,float], dict]`. | Top-level `torch`, `torchvision`, `skimage`, `torchxrayvision`; constructor creates `xrv.models.DenseNet(weights="densenet121-res224-all")`, defaults `cuda`. | **STATIC_ONLY / NOT_TESTED**. Weight retrieval/cached behavior and CPU compatibility require a separately authorized environment. |
| `chest_xray_segmentation` — `medrax/tools/segmentation.py` | `ChestXRaySegmentationTool`; `ChestXRaySegmentationInput(image_path: str, organs: list[str] | None = None)`; `_run(image_path, organs=None, run_manager=None) -> tuple[dict, dict]`. Invalid organ labels raise internally and are returned as failed metadata. | Top-level NumPy, Torch, TorchVision, TorchXRayVision, Matplotlib, scikit-image; constructor builds `xrv.baseline_models.chestx_det.PSPNet()`, defaults `cuda`, creates `temp/`. | **STATIC_ONLY / NOT_TESTED**. Constructor has model and filesystem side effects; model-weight behavior is not verified. |
| `xray_phrase_grounding` — `medrax/tools/grounding.py` | `XRayPhraseGroundingTool`; `XRayPhraseGroundingInput(image_path: str, phrase: str, max_new_tokens: int = 300)`; `_run(...) -> tuple[dict, dict]`. | Top-level Torch, PIL, Matplotlib, Transformers/`BitsAndBytesConfig`; constructor loads `microsoft/maira-2` with `trust_remote_code=True`, optional 4/8-bit quantization, default `cuda`, temp directory. | **STATIC_ONLY / NOT_TESTED**. Remote code/model download and quantization paths must be separately security-reviewed and executed only with approval. |
| `chest_xray_expert` — `medrax/tools/xray_vqa.py` | `XRayVQATool`; `XRayVQAToolInput(image_paths: list[str], prompt: str, max_new_tokens: int = 512)`; `_run(...) -> tuple[dict, dict]`, output key `response`. Missing paths are caught into failed metadata. | Top-level Torch/Transformers; constructor loads `StanfordAIMI/CheXagent-2-3b` tokenizer/model with `trust_remote_code=True`, alters Transformers version temporarily, defaults `cuda`/`bfloat16`. | **STATIC_ONLY / NOT_TESTED**. Model load/network and device behavior are not verified. |
| `llava_med_qa` — `medrax/tools/llava_med.py` | `LlavaMedTool`; `LlavaMedInput(question: str, image_path: str | None = None)`; `_run(...) -> tuple[str, dict]`. | Top-level Torch, PIL, internal LLaVA-Med model utilities; constructor calls `load_pretrained_model` for `microsoft/llava-med-v1.5-mistral-7b`, defaults `/model-weights`, `cuda`, `bfloat16`. `_process_input` explicitly calls `.cuda()`. | **STATIC_ONLY / NOT_TESTED**. Not CPU compatible as written for image processing; CPU execution requires a future code/design change and authorized model assets. |

## Registry and invocation-adapter compatibility

`PAPER_TOOL_NAMES` matches the six source `name` constants. `ToolRegistry` can
validate and dispatch injected mocks, but cannot construct real tools. The
Task 14 `ControllerInvocationInput` contains question text and opaque `image_ids`;
the six source tools instead require raw `image_path`/`image_paths` (except
LLaVA-Med's optional image). This is an intentional unresolved integration gap:
the generic controller boundary must not be connected directly to any concrete
tool without a trusted opaque-ID-to-path dispatch layer and explicit tool-policy
authorization. It exposes no answers, explanations, evaluation records,
credentials, local paths, or loader context.

No source inspected contains direct API-key/environment credential reads or an
explicit HTTP client call. `from_pretrained`/model constructors may contact model
hubs when assets are absent; this was not executed and is **NOT_VERIFIED**.
