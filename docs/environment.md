# Environment

Paper runs: single GPU, Linux, CUDA 12.8, PyTorch 2.11.0+cu128, transformers 5.17.0,
bitsandbytes 0.50.2 (backbones loaded in 4-bit NF4, bf16 compute). Exact pins: requirements.txt.

CPU-only (tests, smoke test, table reproduction): `pip install -r requirements-cpu.txt`.

Hugging Face access: gated backbones need a token supplied through the environment only:

    export HF_TOKEN=...        # never commit tokens; nothing in this repository reads a token file

Default output locations are relative to the working directory (`runs/`, `cache/`,
`runs_transfer/`) and can be changed with `--out_root` / `--cache_root`.
