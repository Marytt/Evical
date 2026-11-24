# Third-party components

No third-party source code is vendored in this repository. The following are
runtime dependencies, installed separately and governed by their own licenses.

| Package | Version used for the paper runs | License (as declared by the project) |
|---|---|---|
| torch | 2.11.0+cu128 | BSD-3-Clause |
| transformers | 5.17.0 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| bitsandbytes | 0.50.2 | MIT |
| huggingface_hub | 1.30.0 | Apache-2.0 |
| tokenizers | 0.23.2 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| numpy | 2.5.3 | BSD-3-Clause |
| pandas | 3.0.5 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |


## Models (not distributed)
Backbones are downloaded from the Hugging Face Hub by the user and are subject to
each model's own license and access terms (several are gated):
Qwen/Qwen2.5-7B-Instruct, meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Ministral-8B-Instruct-2410, google/gemma-2-9b-it, microsoft/Phi-4-mini-instruct, meta-llama/Llama-3.1-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3.
See configs/pipeline_config.yaml
for the exact identifiers used.

## Data (not distributed)
- MedAbstain (MedQA-derived NA/NAP pairs): downloaded at run time from the
  MedAbstain repository by src/data/dataset.py.
- MediQ (stellalisy/mediQ, all_dev_good.jsonl): downloaded by the user; NA/NAP
  pairs are built locally by scripts/prepare_mediq.py.
Review each dataset's license and terms before use. No question text is included here.
