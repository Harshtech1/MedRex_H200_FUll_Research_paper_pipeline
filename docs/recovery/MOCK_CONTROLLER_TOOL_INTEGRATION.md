# Mock Controller-to-Tool Integration

This is a synthetic contract path, not medical inference or benchmark
reproduction. `MockControllerToolIntegration` exposes the existing path-free
`ControllerInvocationInput` to an injected fake controller. Its decision holds
only an allowlisted tool name, opaque image IDs, and tool-specific arguments.

The integration verifies those IDs belong to the prepared benchmark question,
then uses `TrustedToolDispatcher` and the loader-issued context to make a
trusted `PreparedToolCall`. It routes that call only to an injected fake tool.
The six source-audited mappings remain unchanged: report, classification, and
segmentation get one `image_path`; grounding gets `image_path`, `phrase`, and a
limit; CheXagent gets one-or-more `image_paths`, `prompt`, and a limit; LLaVA
gets `question` and an optional `image_path`.

Fake outputs must be `MockToolOutput` with a nonempty response and must not echo
trusted paths. Invalid decisions and malformed outputs become explicit
`INVALID_RESPONSE` attempts; fake callable exceptions become `EXECUTION_ERROR`.
There is no fallback, retry, persistence, or real-tool path. The caller must
separately record the returned attempt via `LockedRunOrchestrator.record_outcome`,
which retains its lock and snapshot ordering.

Remaining gaps: an approved real execution adapter, CPU-compatible model assets,
tool-output schemas, and an explicit execution authorization policy are still
required. No concrete tool, provider, model, or external API is enabled here.
