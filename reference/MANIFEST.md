# Reference artifact manifest

Public artifacts downloaded for Phase 1 forensics. **This directory is
read-only after acquisition** -- vendor and upstream files are never edited.
The archives are not committed to git (see `.gitignore`); this manifest is,
so any result can be traced back to the exact bytes it came from.

Regenerate with `python scripts/hash_reference.py`.

Generated 2026-09-05 01:04 UTC.

| Artifact | Bytes | SHA-256 | Origin |
|---|---:|---|---|
| `hailo_model_zoo/HAILO8_stereo_depth_estimation.rst` | 1,783 | `9490051ed8eb8535d4d6e959e186c98c...` | https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/docs/public_models/HAILO8/HAILO8_stereo_depth_estimation.rst |
| `hailo_model_zoo/stereonet.alls` | 17,094 | `9928fc4b01c07e6062b47abe62a142df...` | https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/alls/generic/stereonet.alls |
| `hailo_model_zoo/stereonet.yaml` | 1,230 | `9655b95f3884d1c2bca58af1c2642aeb...` | https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/networks/stereonet.yaml |
| `onnx/stereonet.onnx` | 23,690,586 | `b1a01d855bb22663f06dfd29eae11194...` | derived (extracted from an archive above) |
| `stereonet.hef` | 24,057,165 | `541ee6cb841df318f63ad93e76d6048a...` | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/stereonet.hef |
| `stereonet.zip` | 1,617,369 | `591f11d1deaa71d6bda0eade4dc498d3...` | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/DisparityEstimation/stereonet/pretrained/2023-05-31/stereonet.zip |
| `stereonet_profiler_results_compiled_runtime_data.html` | 40,883,621 | `cf69a03f5d359b3716cef784be140fa5...` | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/stereonet_profiler_results_compiled_runtime_data.html |
| `upstream/hailo_apps_main.zip` | 240,596,068 | `b89369fdfbaf073b710005ca50dfba63...` | https://codeload.github.com/hailo-ai/hailo-apps/zip/refs/heads/main |
| `upstream/hailo_model_zoo_master.zip` | 2,508,564 | `dd7ee631b8c7d9a6c67c3e615bc71c40...` | https://codeload.github.com/hailo-ai/hailo_model_zoo/zip/refs/heads/master |
| `upstream/nivosco_StereoNet.zip` | 869,310 | `db1f11d80bc68db087ea1a26d2644751...` | https://codeload.github.com/nivosco/StereoNet/zip/refs/heads/master |

Full hashes are in `reference/manifest.json`.

`upstream/extracted/` holds the unpacked archives and is intentionally not
hashed file-by-file -- it is reproducible from the archives listed above.
