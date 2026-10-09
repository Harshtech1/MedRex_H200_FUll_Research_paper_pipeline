# Model Provenance

All entries in this table are **HISTORICAL RECORD** except current-workspace absence, which is **CURRENTLY VERIFIED**. No listed model artifact is present in the workspace, and no download or model construction was performed.

| Component | Identifier / integrity record | Recovery requirement |
|---|---|---|
| CheXagent | `StanfordAIMI/CheXagent-2-3b` @ `8f19b53a2eceda4c33b0acec6c81fbc293ad80d0` | Pin acquisition and validate compatibility adapter. |
| XraySigLIP | `StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli` @ `e80904a69fa4c2517670950e444f8c9d626d84c3` | Preserve nested dependency provenance. |
| LLaVA-Med | `microsoft/llava-med-v1.5-mistral-7b` @ `91bb16c122001ddc9cf1fd36ce1dae09448943a2` | Propagate cache paths through main and nested loaders. |
| CLIP | `openai/clip-vit-large-patch14-336` @ `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` | Pin nested vision artifact and verify offline cache behavior. |
| MAIRA-2 | `microsoft/maira-2` @ `795a2b1cd4a310624b4e3d14b5a23e41fd273deb` | Pin before construction/inference. |
| CheXpert findings | `IAMJB/chexpert-mimic-cxr-findings-baseline` @ `5bed2b551d7cadfe08f0757d29cc054dc1adb60d` | Support explicit local path and revision. |
| CheXpert impression | `IAMJB/chexpert-mimic-cxr-impression-baseline` @ `dd9e8a79659426980e8197657c81d2b9d443ddd0` | Support explicit local path and revision. |
| ChestX-Det PSPNet | SHA-256 `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc` | Verify downloaded/local checkpoint before use. |
| TorchXRayVision | Historical abbreviated value `56524913...68899` | **NOT YET VERIFIED**: not a valid checksum; recover a complete artifact hash. |

**CURRENTLY VERIFIED**: upstream source has model identifiers for the named runtime tools but no listed revision pins or complete local artifact integrity records.
