"""
src/utils/seeding.py
Full seed registration for reproducible runs.

Why every line matters here:
  * random          - Python-level shuffling
  * numpy           - split permutations, batch order, bootstrap CIs
  * torch           - head weight init
  * torch.cuda      - DROPOUT in the ConfidenceHead runs on the CUDA RNG, so
                      without manual_seed_all the head is not reproducible even
                      if every other seed is fixed
  * cudnn flags     - forces deterministic kernel selection
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True,
                  strict: bool = False) -> dict:
    """
    Seed every RNG the pipeline touches.

    Args:
        seed:          the seed
        deterministic: set cudnn deterministic / disable benchmark autotuning
        strict:        additionally call torch.use_deterministic_algorithms.
                       Stricter, but raises on ops with no deterministic
                       implementation, so it is opt-in.

    Returns a record of what was set, for the reproducibility bundle.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as ex:
            print(f"[seed] strict determinism unavailable: {ex}")

    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": True,
        "torch_cuda": cuda_available,
        "cudnn_deterministic": bool(deterministic),
        "cudnn_benchmark": False if deterministic else None,
        "strict_deterministic_algorithms": bool(strict),
    }
