# ChestAgentBench Provenance

- **HISTORICAL RECORD**: dataset `wanglab/chest-agent-bench`, revision `921e60440927f9893228d843c0206b755744252a`.
- **HISTORICAL RECORD**: metadata SHA-256 `45e429e3a3b8e4dbfa064cbdaa3f2960bee90cad9e0f25501e38da67597bc2a`; figures archive SHA-256 `8d5ac156c267046aadf54f20d961896c6e73d31ec59b8f635d1b208fa1444226`.
- **HISTORICAL RECORD**: prepared inventory was 2,500 questions/IDs, 609 case IDs, 4,629 references, and 1,346 distinct resolved image paths.
- **CURRENTLY VERIFIED**: these assets and hashes are absent from this checkout and cannot be validated here.

## Identifier adapter contract

- **HISTORICAL RECORD**: controller-facing schemas used selected image identifiers scoped to the current case.
- **HISTORICAL RECORD**: the adapter validated an identifier then resolved it to a canonical local path immediately before production-tool invocation.
- **HISTORICAL RECORD**: remote images, cross-case IDs, basename fallback, and raw filesystem-path arguments were prohibited; production tools were not rewritten merely for this contract.

## Experiment record

- **HISTORICAL RECORD**: B1 scored 16/25 (64.0%), with 71 tool calls and 71 successful calls; this is not a local reproduction and does not demonstrate an adapter-caused accuracy improvement.
- **HISTORICAL RECORD**: original B2 selected 100 cases but stopped after 11 completed observations because of Gemini 429 responses. No B2 accuracy estimate was reported.
- **HISTORICAL RECORD**: B2 R2 preflight passed without model/API/GPU work, but R2 inference was never launched.
- **NOT YET VERIFIED**: the published 63.1% MedRAX result has not been reproduced by this repository.
