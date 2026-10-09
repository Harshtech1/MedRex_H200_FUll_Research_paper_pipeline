# Trusted Tool Dispatch Contract

`TrustedToolDispatcher` is the boundary between path-free controller output and
the raw local paths required by the six audited tools. It accepts only a
loader-issued `CaseContext`, an allowlisted `PAPER_TOOL_NAMES` name, opaque image
IDs, and a tool-specific allowlisted argument mapping. It resolves IDs solely
through `CaseImageIdentifierAdapter`; callers cannot submit a path or case
override.

Preparation is non-executing. `PreparedToolCall` hides trusted execution paths
from `repr` and controller-safe metadata. A later separately authorized executor
may read `execution_arguments()` and invoke a preconstructed tool; this module
does neither.

Mappings: report/classification/segmentation use one `image_path`; grounding
uses one path plus `phrase` and `max_new_tokens`; CheXagent uses one-or-more
`image_paths` plus `prompt` and `max_new_tokens`; LLaVA-Med uses `question` plus
zero-or-one `image_path`. Duplicate, unknown, cross-case, missing, symlinked,
or non-regular assets fail closed. Evaluation fields, credentials, raw paths,
and unsupported arguments are rejected.

Remaining work: an explicitly authorized execution adapter must bind a prepared
call to an already-constructed concrete tool, preserve lock/snapshot ordering,
and define tool output validation. No such execution is enabled here.
