# H200 Error Ledger

Every historical resolution below requires fresh reconstruction validation.

| Finding | Classification | Evidence and lesson |
|---|---|---|
| CheXagent on Transformers 4.51.3 raised `DynamicCache.get_max_length` missing. | FIXED HISTORICALLY | **HISTORICAL RECORD**: a prefill/decode compatibility adapter was developed and real CXR inference completed. **NOT YET VERIFIED**: no adapter is in this checkout. |
| A simple DynamicCache alias could suppress image-prefill encoding. | MITIGATED HISTORICALLY | **HISTORICAL RECORD**: the alias approach was rejected; contract-aware behavior was required. |
| LLaVA-Med nested CLIP loader missed the selected cache path. | FIXED HISTORICALLY | **CURRENTLY VERIFIED**: current CLIP loader still calls `from_pretrained` without `cache_dir`; reconstruction is required. |
| CheXpert report generation required explicit local findings and impression paths. | FIXED HISTORICALLY | **CURRENTLY VERIFIED**: upstream only uses model IDs plus shared `cache_dir`; the historical path behavior is absent. |
| Controller unexpectedly resolved to GPT-4o. | UNRESOLVED | **CURRENTLY VERIFIED**: current entry point uses `ChatOpenAI` and invokes `gpt-4o`; Gemini selection is absent. Gemini remains an experimental controller substitution, not an exact paper reproduction. |
| Arbitrary filesystem paths caused benchmark tool-call rejection. | MITIGATED HISTORICALLY | **HISTORICAL RECORD**: a case-scoped identifier adapter eliminated the observed rejection class; it was not a controlled accuracy improvement. |
| Duplicate B2 workers shared selection/output state. | FIXED HISTORICALLY | **HISTORICAL RECORD**: exclusive locking and atomic writes were introduced. **NOT YET VERIFIED**: implementation is absent here. |
| Gemini free-tier 429 `RESOURCE_EXHAUSTED` stopped B2. | INFRASTRUCTURE LIMITATION | **HISTORICAL RECORD**: B2 stopped after 11/100 cases; no fallback controller and no B2 accuracy estimate were reported. |
| CheXagent CUDA cleanup retained 32 MiB in process. | INFRASTRUCTURE LIMITATION | **HISTORICAL RECORD**: inference passed with a qualified cleanup observation, not an unqualified zero-memory claim. |
