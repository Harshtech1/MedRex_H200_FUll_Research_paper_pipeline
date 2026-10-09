# H200 Reconstruction Plan

## Gate lifecycle

Every gate must follow: `IMPLEMENT -> STATIC CHECKS -> TARGETED VALIDATION -> RECORD EVIDENCE -> REVIEW -> COMMIT -> PUSH -> VERIFY REMOTE COMMIT`.

1. **Baseline preservation** — **CURRENTLY VERIFIED**: branch starts at audited upstream commit; acceptance: recorded immutable identity and clean baseline.
2. **Recovery documentation and first checkpoint** — acceptance: these six ledgers pass review and are committed/pushed without source changes.
3. **Reproducible environment/dependency specification** — acceptance: pinned manifest reconciles historical H200 references with actual compatible packages.
4. **ChestAgentBench restoration and checksum verification** — acceptance: revision, metadata/archive hashes, record counts, and all referenced paths match evidence.
5. **H0 environment validation** — acceptance: non-inference environment evidence is recorded before CUDA/model work.
6. **H2 XRV** — acceptance: production tool produces 18 pathology outputs and a complete artifact hash is recorded.
7. **H3 CheXagent** — provenance, pinned construction, compatibility contract, then isolated real inference; acceptance: image-prefill/decode contract and inference evidence pass.
8. **H4 LLaVA-Med** — acceptance: pinned main/nested cache behavior and targeted inference pass.
9. **H5 MAIRA-2** — acceptance: pinned construction and phrase-grounding inference pass.
10. **H6 ChestX-Det** — acceptance: checkpoint SHA matches and segmentation validation passes.
11. **H7 CheXpert reports** — acceptance: explicit local paths/revisions and report generation pass.
12. **H8 Gemini controller and six-tool integration** — acceptance: fail-closed provider selection, schemas, continuation/error behavior, and six-tool gate pass. Gemini must be labeled a controller substitution.
13. **B1 identifier adapter and 25-case validation** — acceptance: adapter leakage tests pass and outcomes are separately recorded.
14. **B2 exclusive lock and atomic state** — acceptance: live-owner, atomicity, and deterministic-selection tests pass.
15. **R2 preflight** — acceptance: zero API/model/CUDA/network/inference behavior plus expected hashes/namespace is evidenced.
16. **Separate Gemini quota/capacity decision** — acceptance: explicit authorization and capacity evidence before any R2 inference.

**NOT YET VERIFIED**: Later gates must not be merged as equivalent to the historical implementation until their stated evidence has been recorded.
